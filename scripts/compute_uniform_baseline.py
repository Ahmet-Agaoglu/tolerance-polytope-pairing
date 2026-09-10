"""Compare known-w optimization, uniform-w optimization, and random pairing."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import eye,kron,vstack

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tolerance_pairing.core import CaseStudy,build_polytope,functional_constraints,pair_constraints
from tolerance_pairing.pilot import optimal_matching
from tolerance_pairing.comparison import expected_random_g

SOURCE=ROOT/'results/convergence_five_n50_seed20260907'
ORACLE=ROOT/'results/reoptimization_no_threshold_b177/batch_results.json'
OUT=ROOT/'results/comparison_r2_wide_b177'
BATCHES,N,RADIUS_UM,M,WORKERS=177,50,2.,256,12
SCALE=np.array([.001,.001,.001/200])
AF,BF=functional_constraints(RADIUS_UM/1000,M)


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def uniform_probability(poly):
    if poly.status!='full_dimensional': return 0.
    if np.max(np.linalg.norm(poly.physical_vertices[:,:2],axis=1))<=RADIUS_UM/1000:
        return 1.
    intersection=build_polytope(np.vstack((poly.A,AF)),np.r_[poly.b,BF],poly.scale)
    if intersection.status!='full_dimensional': return 0.
    value=intersection.volume/poly.volume
    if value < -1e-9 or value > 1+1e-9: raise RuntimeError('Uniform probability range failure.')
    return float(np.clip(value,0,1))


def run_row(task):
    i,pin,holes,nominal=task; values=[]
    for j,hole in enumerate(holes):
        poly=build_polytope(*pair_constraints(pin,hole,nominal),SCALE)
        if poly.status=='lower_dimensional':
            raise RuntimeError(f'Lower-dimensional P at pair {(i,j)}.')
        values.append(uniform_probability(poly))
    return dict(row=i,p_uniform=values)


def diagnostics(results):
    if not results: return None
    oracle=np.array([x['G_oracle_known_w'] for x in results])
    uniform=np.array([x['G_uniform_optimizer_true_w'] for x in results])
    random=np.array([x['G_random_expected_true_w'] for x in results])
    def stats(x): return dict(mean=float(x.mean()),se=float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None)
    knowledge=oracle-uniform; optimization=uniform-random
    return dict(batches=len(results),oracle=stats(oracle),uniform_optimizer=stats(uniform),random=stats(random),
        knowledge_gain=stats(knowledge),optimization_gain=stats(optimization),
        oracle_minus_random=stats(oracle-random),
        batches_order_violations=int(np.sum((oracle+1e-9<uniform)|(oracle+1e-9<random))))


def report(results,status,active_batch=None,rows=None,error=None):
    d=diagnostics(results)
    state=dict(status=status,completed_batches=len(results),active_batch=active_batch,
               completed_rows=rows,diagnostics=d,error=error,
               updated_local=time.strftime('%Y-%m-%d %H:%M:%S'))
    save(OUT/'progress.json',state)
    cells='Pending'
    if d:
        f=lambda x:'Pending' if x is None else f'{x:.4f}'
        cells=f'''<table><tr><th>Method / contrast</th><th>Mean G</th><th>SE</th></tr>
<tr><td>Optimizer, known wide-centered w</td><td>{d['oracle']['mean']:.4f}</td><td>{f(d['oracle']['se'])}</td></tr>
<tr><td>Optimizer, unknown w (w=1)</td><td>{d['uniform_optimizer']['mean']:.4f}</td><td>{f(d['uniform_optimizer']['se'])}</td></tr>
<tr><td>Random perfect matching</td><td>{d['random']['mean']:.4f}</td><td>{f(d['random']['se'])}</td></tr>
<tr><td>Value of knowing w</td><td>{d['knowledge_gain']['mean']:.4f}</td><td>{f(d['knowledge_gain']['se'])}</td></tr>
<tr><td>Value of optimization under ignorance</td><td>{d['optimization_gain']['mean']:.4f}</td><td>{f(d['optimization_gain']['se'])}</td></tr></table>'''
    document=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Comparison at r=2 um</title><style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;line-height:1.6}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}</style><h1>Comparison: r=2 um, wide-centered Gaussian truth</h1><p>Status: <strong>{html.escape(status)}</strong>. Completed batches: {len(results)}/177. Active batch: {active_batch}; rows: {rows}/50.</p>{cells}<p>All methods are evaluated with the same saved wide-centered Gaussian probabilities. The proposed method optimizes those probabilities. The unknown-w optimizer uses uniform polytope mass ratios (w=1) for its decisions, then is evaluated under the Gaussian truth. Random pairing is a uniformly random perfect matching; its expected G is computed analytically as sum(p_ij)/50, so it adds no permutation Monte Carlo error.</p><p>No positive p_min is imposed. M=256 is fixed based on the completed resolution audit. Common manufacturing batches support paired contrasts. Random pairing attempts all 50 assemblies; zero-probability pairs contribute zero.</p><p>{html.escape(error or '')}</p><p><a href="progress.json">Progress</a> | <a href="summary.json">Final summary</a> | <a href="config.json">Configuration</a></p></html>'''
    temporary=OUT/'report.html.tmp'; temporary.write_text(document,encoding='utf-8'); temporary.replace(OUT/'report.html')


def summarize_batch(batch,p_uniform,p_true,oracle_result):
    allowed=p_uniform>0
    pairs=optimal_matching(p_uniform,allowed,np.nextafter(0.,1.)).reshape(-1,2)
    ii,jj=pairs[:,0],pairs[:,1]
    uniform_objective=float(p_uniform[ii,jj].sum())
    true_g=float(p_true[ii,jj].sum())
    capacity=vstack([kron(eye(N),np.ones((1,N))),kron(np.ones((1,N)),eye(N))],format='csc')
    check=milp(-p_uniform.ravel(),integrality=np.ones(N*N),
        bounds=Bounds(np.zeros(N*N),allowed.ravel().astype(float)),
        constraints=LinearConstraint(capacity,np.zeros(2*N),np.ones(2*N)),options={'mip_rel_gap':0})
    if not check.success or abs(-check.fun-uniform_objective)>1e-6: raise RuntimeError('MILP check failed.')
    oracle_pairs=np.array(oracle_result['pairs_zero_based'],dtype=int).reshape(-1,2)
    oracle_g=float(p_true[oracle_pairs[:,0],oracle_pairs[:,1]].sum())
    if abs(oracle_g-oracle_result['G'])>1e-7: raise RuntimeError('Oracle source mismatch.')
    random_g=expected_random_g(p_true)
    if oracle_g+1e-9<max(true_g,random_g): raise RuntimeError('Known-w oracle dominance failed.')
    return dict(batch=batch,N_oracle=oracle_result['N'],G_oracle_known_w=oracle_g,
        N_uniform_optimizer=len(pairs),G_uniform_objective=uniform_objective,
        G_uniform_optimizer_true_w=true_g,G_random_expected_true_w=random_g,
        value_of_knowing_w=oracle_g-true_g,
        value_of_optimization_under_ignorance=true_g-random_g,
        oracle_gain_over_random=oracle_g-random_g,
        common_oracle_uniform_pairs=len(set(map(tuple,oracle_pairs))&set(map(tuple,pairs))),
        uniform_pairs_zero_based=pairs.tolist(),milp_objective_difference=float(abs(-check.fun-uniform_objective)))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sources=['src/tolerance_pairing/core.py','src/tolerance_pairing/pilot.py',
             'src/tolerance_pairing/comparison.py','examples/comparison_r2_wide.py']
    config=dict(version=1,batches=BATCHES,n=N,radius_um=RADIUS_UM,fc_polygon_sides=M,
        truth='wide-centered Gaussian',competitors=['uniform optimizer','random perfect matching'],
        positive_p_min=None,source=str(SOURCE),oracle=str(ORACLE),part_settings=asdict(CaseStudy()),
        source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources})
    if (OUT/'config.json').exists() and json.loads((OUT/'config.json').read_text())!=config:
        raise RuntimeError('Cached comparison configuration differs.')
    save(OUT/'config.json',config)
    oracle=json.loads(ORACLE.read_text())
    pool=ProcessPoolExecutor(max_workers=WORKERS); results=[]
    try:
        for batch in range(BATCHES):
            folder=OUT/f'batch_{batch:03d}'; folder.mkdir(exist_ok=True); (folder/'rows').mkdir(exist_ok=True)
            summary_path=folder/'summary.json'
            if summary_path.exists(): result=json.loads(summary_path.read_text())
            else:
                with np.load(SOURCE/f'batch_{batch:03d}'/'parts.npz') as parts:
                    pins,holes,nominal=parts['pins'],parts['holes'],parts['nominal']
                rows=[]; futures=[]
                for i in range(N):
                    path=folder/'rows'/f'{i:02d}.json'
                    if path.exists(): rows.append(json.loads(path.read_text()))
                    else: futures.append(pool.submit(run_row,(i,pins[i],holes,nominal)))
                report(results,'Running',batch+1,len(rows))
                for future in as_completed(futures):
                    row=future.result(); save(folder/'rows'/f"{row['row']:02d}.json",row); rows.append(row)
                    report(results,'Running',batch+1,len(rows))
                p_uniform=np.zeros((N,N))
                for row in rows: p_uniform[row['row']]=row['p_uniform']
                np.savez_compressed(folder/'uniform_probabilities.npz',p_uniform=p_uniform)
                with np.load(SOURCE/f'batch_{batch:03d}'/'probabilities.npz') as z: p_true=z['p'][0]
                result=summarize_batch(batch,p_uniform,p_true,oracle[batch]['results'][0]); save(summary_path,result)
            results.append(result); save(OUT/'batch_results.json',results); report(results,'Running')
            print(f'Completed batch {batch+1}/{BATCHES}',flush=True)
        d=diagnostics(results); summary=dict(**d,design=config)
        save(OUT/'summary.json',summary); report(results,'Completed')
    except BaseException:
        report(results,'Failed: review required',error=traceback.format_exc())
        pool.terminate_workers(); raise
    else: pool.shutdown()


if __name__=='__main__': main()
