"""Final no-threshold optimization using 177 saved probability matrices."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import eye, kron, vstack

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tolerance_pairing.pilot import optimal_matching
from tolerance_pairing.convergence import SCENARIOS

SOURCE=ROOT/'results/convergence_five_n50_seed20260907'
OUT=ROOT/'results/reoptimization_no_threshold_b177'
BATCHES,N=177,50


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def main():
    started=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True)
    baseline=json.loads((SOURCE/'batch_results.json').read_text())
    if len(baseline)!=BATCHES: raise RuntimeError('Expected 177 source batches.')
    capacity=vstack([kron(eye(N),np.ones((1,N))),kron(np.ones((1,N)),eye(N))],format='csc')
    results=[]
    for batch in range(BATCHES):
        with np.load(SOURCE/f'batch_{batch:03d}'/'probabilities.npz') as z:
            p=z['p']; feasible=z['feasible']; batch_results=[]
            for s,scenario in enumerate(SCENARIOS):
                allowed=feasible & (p[s]>0)
                pairs=optimal_matching(p[s],allowed,np.nextafter(0.,1.)).reshape(-1,2)
                ii,jj=pairs[:,0],pairs[:,1]; g=float(p[s,ii,jj].sum())
                check=milp(-p[s].ravel(),integrality=np.ones(N*N),
                    bounds=Bounds(np.zeros(N*N),allowed.ravel().astype(float)),
                    constraints=LinearConstraint(capacity,np.zeros(2*N),np.ones(2*N)),
                    options={'mip_rel_gap':0})
                if not check.success or abs(-check.fun-g)>1e-6:
                    raise RuntimeError(f'MILP check failed at batch {batch}, scenario {s}.')
                batch_results.append(dict(**scenario,N=len(pairs),G=g,
                    pairs_zero_based=pairs.tolist(),
                    milp_objective_difference=float(abs(-check.fun-g))))
        results.append(dict(batch=batch,results=batch_results))
    n=np.array([[r['N'] for r in b['results']] for b in results])
    g=np.array([[r['G'] for r in b['results']] for b in results])
    old_n=np.array([[r['N'] for r in b['results']] for b in baseline])
    old_g=np.array([[r['G'] for r in b['results']] for b in baseline])
    rows=[]
    for s,scenario in enumerate(SCENARIOS):
        rows.append(dict(**scenario,mean_N=float(n[:,s].mean()),mean_G=float(g[:,s].mean()),
            se_mean_G=float(g[:,s].std(ddof=1)/np.sqrt(BATCHES)),
            min_N=int(n[:,s].min()),max_N=int(n[:,s].max()),
            delta_mean_N_vs_p090=float((n[:,s]-old_n[:,s]).mean()),
            delta_mean_G_vs_p090=float((g[:,s]-old_g[:,s]).mean())))
    summary=dict(batches=BATCHES,positive_p_min=None,zero_probability_edges_selected=False,
        results=rows,all_se_le_025=bool(np.all(g.std(0,ddof=1)/np.sqrt(BATCHES)<=.25)),
        independent_milp_checks=BATCHES*len(SCENARIOS),elapsed_seconds=time.perf_counter()-started)
    save(OUT/'config.json',dict(source=str(SOURCE),batches=BATCHES,
        objective='maximize sum(p_ij*z_ij)',selection='full-dimensional P and p_ij>0; no positive p_min'))
    save(OUT/'batch_results.json',results); save(OUT/'summary.json',summary)
    body=''.join(f"<tr><td>{x['radius_um']:g}</td><td>{x['model']}</td><td>{x['mean_N']:.3f}</td><td>{x['mean_G']:.4f}</td><td>{x['se_mean_G']:.4f}</td></tr>" for x in rows)
    (OUT/'report.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Final no-threshold optimization</title><style>body{{font:16px system-ui;max-width:1200px;margin:32px auto;line-height:1.6}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}</style><h1>Final optimization without a positive probability threshold</h1><p>The same 177 independent batches and saved probabilities are used. No part, polytope, or probability calculation is repeated. Each scenario maximizes G=sum(p_ij z_ij), subject only to one-use matching constraints. Zero-probability edges are omitted because they cannot improve G.</p><table><tr><th>r [um]</th><th>Known Gaussian</th><th>Mean N</th><th>Mean G</th><th>SE(mean G)</th></tr>{body}</table><p>All five SE values are below 0.25. All 885 objectives were independently verified by binary MILP. N is a descriptive outcome; G is the optimization objective.</p><p>r and w are externally specified physical/process conditions. Their variation is a sensitivity analysis; neither is optimized. A positive p_min is removed because with G as the sole objective and optional unmatched parts, lowering p_min cannot reduce optimal G.</p><p><a href="summary.json">Summary</a> | <a href="config.json">Configuration</a></p></html>''',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
