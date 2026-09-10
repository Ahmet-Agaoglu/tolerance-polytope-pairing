"""Evaluate saved uniform decisions and random pairing under all three true w models."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tolerance_pairing.comparison import expected_random_g
from tolerance_pairing.convergence import MODELS

SOURCE=ROOT/'results/convergence_five_n50_seed20260907'
ORACLE=ROOT/'results/reoptimization_no_threshold_b177/batch_results.json'
UNIFORM=ROOT/'results/comparison_r2_wide_b177'
OUT=ROOT/'results/comparison_r2_all_w_b177'
BATCHES=177


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def estimate(values):
    x=np.asarray(values,dtype=float); se=float(x.std(ddof=1)/np.sqrt(len(x)))
    half=float(t.ppf(.975,len(x)-1)*se)
    return dict(mean=float(x.mean()),se=se,ci95=[float(x.mean()-half),float(x.mean()+half)])


def main():
    started=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True)
    oracle=json.loads(ORACLE.read_text())
    if len(oracle)!=BATCHES: raise RuntimeError('Expected 177 oracle batches.')
    rows=[]
    for batch in range(BATCHES):
        uniform_summary=json.loads((UNIFORM/f'batch_{batch:03d}'/'summary.json').read_text())
        pairs=np.array(uniform_summary['uniform_pairs_zero_based'],dtype=int).reshape(-1,2)
        with np.load(SOURCE/f'batch_{batch:03d}'/'probabilities.npz') as z: truth=z['p'][:3]
        evaluations=[]
        for model in range(3):
            p=truth[model]; oracle_result=oracle[batch]['results'][model]
            oracle_pairs=np.array(oracle_result['pairs_zero_based'],dtype=int).reshape(-1,2)
            oracle_g=float(p[oracle_pairs[:,0],oracle_pairs[:,1]].sum())
            uniform_g=float(p[pairs[:,0],pairs[:,1]].sum())
            random_g=expected_random_g(p)
            if abs(oracle_g-oracle_result['G'])>1e-7: raise RuntimeError('Oracle mismatch.')
            if oracle_g+1e-9<max(uniform_g,random_g): raise RuntimeError('Oracle dominance failure.')
            evaluations.append(dict(model=MODELS[model],G_oracle=oracle_g,G_uniform=uniform_g,
                G_random=random_g,value_of_knowing_w=oracle_g-uniform_g,
                value_of_optimization=uniform_g-random_g,
                oracle_gain_over_random=oracle_g-random_g,
                common_oracle_uniform_pairs=len(set(map(tuple,oracle_pairs))&set(map(tuple,pairs)))))
        rows.append(dict(batch=batch,evaluations=evaluations))
    summary_rows=[]
    for model,name in enumerate(MODELS):
        entries=[b['evaluations'][model] for b in rows]
        summary_rows.append(dict(model=name,
            oracle=estimate([e['G_oracle'] for e in entries]),
            uniform_optimizer=estimate([e['G_uniform'] for e in entries]),
            random=estimate([e['G_random'] for e in entries]),
            value_of_knowing_w=estimate([e['value_of_knowing_w'] for e in entries]),
            value_of_optimization=estimate([e['value_of_optimization'] for e in entries]),
            oracle_gain_over_random=estimate([e['oracle_gain_over_random'] for e in entries]),
            mean_common_oracle_uniform_pairs=float(np.mean([e['common_oracle_uniform_pairs'] for e in entries]))))
    summary=dict(batches=BATCHES,radius_um=2.,positive_p_min=None,true_w_models=MODELS,
        results=summary_rows,uniform_geometry_recomputed=False,
        random_permutations_drawn=False,elapsed_seconds=time.perf_counter()-started)
    save(OUT/'batch_results.json',rows); save(OUT/'summary.json',summary)
    save(OUT/'config.json',dict(source_probabilities=str(SOURCE),oracle_decisions=str(ORACLE),
        uniform_decisions=str(UNIFORM),batches=BATCHES,radius_um=2.,positive_p_min=None))
    body=''.join(f"<tr><td>{x['model']}</td><td>{x['oracle']['mean']:.4f}</td><td>{x['uniform_optimizer']['mean']:.4f}</td><td>{x['random']['mean']:.4f}</td><td>{x['value_of_knowing_w']['mean']:.4f}</td><td>{x['value_of_knowing_w']['se']:.4f}</td><td>{x['value_of_optimization']['mean']:.4f}</td><td>{x['value_of_optimization']['se']:.4f}</td><td>{x['mean_common_oracle_uniform_pairs']:.2f}</td></tr>" for x in summary_rows)
    (OUT/'report.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Comparison under all w models</title><style>body{{font:16px system-ui;max-width:1250px;margin:32px auto;line-height:1.6}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}</style><h1>Comparison under all true w models</h1><p>r=2 um; 177 common batches; no positive p_min. One saved w=1 decision per batch is evaluated under each of the three true Gaussian models. Known-w oracle decisions and Gaussian probabilities are reused. Random perfect-matching expectations are analytic.</p><table><tr><th>True w</th><th>Known-w G</th><th>w=1 G</th><th>Random G</th><th>Value of w</th><th>SE</th><th>Value of optimization</th><th>SE</th><th>Common oracle/w=1 pairs</th></tr>{body}</table><p>Value of w = G(known-w optimizer) - G(w=1 optimizer), evaluated under the same true w. Value of optimization = G(w=1 optimizer) - E[G(random)], also under that true w. Approximate paired 95% intervals are available in summary.json. This comparison quantifies information and optimization value; it does not rank the physical plausibility of the w models.</p><p><a href="summary.json">Detailed summary</a> | <a href="config.json">Configuration</a></p></html>''',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
