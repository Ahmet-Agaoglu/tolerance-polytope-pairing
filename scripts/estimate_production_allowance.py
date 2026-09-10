"""Extend saved 50x50 batches and estimate production allowance for 50 outputs."""

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import html
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tolerance_pairing.core import CaseStudy, build_polytope, generate_pair, pair_constraints
from tolerance_pairing.convergence import MEANS, SIGMAS
from tolerance_pairing.gaussian_screen import evaluate
from tolerance_pairing.pilot import optimal_matching


SOURCE = ROOT / "results" / "convergence_five_n50_seed20260907"
OUT = ROOT / "results" / "production_allowance_r2_wide"
MASTER_SEED = 20260907
BASE_N = 50
MAX_N = 80
TARGET_OUTPUT = 50
RADIUS_UM = 2.0
WORKERS = 12
SCALE = np.array([0.001, 0.001, 0.001 / 200])
WIDE_MEAN = MEANS[:1]
WIDE_SIGMA = SIGMAS[:1]


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def gaussian_probability(pin, hole, nominal):
    poly = build_polytope(*pair_constraints(pin, hole, nominal), SCALE)
    if poly.status == "lower_dimensional":
        raise RuntimeError("Lower-dimensional P requires separate measure handling.")
    if poly.status != "full_dimensional":
        return 0.0
    result = evaluate(poly, WIDE_MEAN, WIDE_SIGMA, radius_um=RADIUS_UM, p_min=0.9)
    probability = float(result["p"][0])
    if not 0 <= probability <= 1:
        raise RuntimeError("Probability is outside [0, 1].")
    return probability


def run_row(task):
    i, start_j, pin, holes, nominal = task
    values = [gaussian_probability(pin, hole, nominal) for hole in holes[start_j:]]
    return dict(i=i, start_j=start_j, values=values)


def poisson_binomial_tail(probabilities, target):
    """Exact conditional probability of at least target independent successes."""
    distribution = np.array([1.0])
    for probability in probabilities:
        distribution = np.convolve(distribution, [1 - probability, probability])
    return float(distribution[target:].sum()) if len(distribution) > target else 0.0


def make_extended_parts(batch, folder):
    path = folder / "parts_n80.npz"
    if path.exists():
        with np.load(path) as data:
            return data["pins"], data["holes"], data["nominal"]
    children = np.random.SeedSequence(MASTER_SEED, spawn_key=(batch,)).spawn(MAX_N)
    generated = [generate_pair(seed, CaseStudy()) for seed in children]
    pins = np.array([part[0] for part in generated])
    holes = np.array([part[1] for part in generated])
    nominal = generated[0][2]
    with np.load(SOURCE / f"batch_{batch:03d}" / "parts.npz") as saved:
        if not np.array_equal(pins[:BASE_N], saved["pins"]):
            raise RuntimeError("Regenerated base pins do not match the saved batch.")
        if not np.array_equal(holes[:BASE_N], saved["holes"]):
            raise RuntimeError("Regenerated base holes do not match the saved batch.")
        if not np.array_equal(nominal, saved["nominal"]):
            raise RuntimeError("Regenerated nominal geometry does not match.")
    np.savez_compressed(path, pins=pins, holes=holes, nominal=nominal)
    return pins, holes, nominal


def analyze_matrix(probability):
    results = []
    for n in range(BASE_N, MAX_N + 1):
        matrix = probability[:n, :n]
        pairs = optimal_matching(
            matrix, matrix > 0, np.nextafter(0.0, 1.0)
        ).reshape(-1, 2)
        selected = matrix[pairs[:, 0], pairs[:, 1]]
        results.append(
            dict(
                n=n,
                N=len(pairs),
                G=float(selected.sum()),
                probability_at_least_50=poisson_binomial_tail(selected, TARGET_OUTPUT),
                pairs_zero_based=pairs.tolist(),
            )
        )
    return results


def extend_batch(batch, pool):
    folder = OUT / f"batch_{batch:03d}"
    rows_folder = folder / "rows"
    folder.mkdir(exist_ok=True)
    rows_folder.mkdir(exist_ok=True)
    summary_path = folder / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    pins, holes, nominal = make_extended_parts(batch, folder)
    tasks = []
    for i in range(MAX_N):
        start_j = BASE_N if i < BASE_N else 0
        row_path = rows_folder / f"{i:02d}.json"
        if not row_path.exists():
            tasks.append((i, start_j, pins[i], holes, nominal))
    futures = [pool.submit(run_row, task) for task in tasks]
    for future in as_completed(futures):
        row = future.result()
        save_json(rows_folder / f"{row['i']:02d}.json", row)

    with np.load(SOURCE / f"batch_{batch:03d}" / "probabilities.npz") as saved:
        probability = np.full((MAX_N, MAX_N), np.nan)
        probability[:BASE_N, :BASE_N] = saved["p"][0]
    for i in range(MAX_N):
        row = json.loads((rows_folder / f"{i:02d}.json").read_text(encoding="utf-8"))
        probability[i, row["start_j"]:] = row["values"]
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise RuntimeError("Extended probability matrix validation failed.")
    np.savez_compressed(folder / "probabilities_n80.npz", p=probability)
    result = dict(batch=batch, results=analyze_matrix(probability))
    save_json(summary_path, result)
    return result


def confidence_summary(batches):
    one_sided = float(t.ppf(0.95, len(batches) - 1)) if len(batches) > 1 else None
    rows = []
    for index, n in enumerate(range(BASE_N, MAX_N + 1)):
        g = np.array([batch["results"][index]["G"] for batch in batches])
        reliability = np.array([
            batch["results"][index]["probability_at_least_50"] for batch in batches
        ])
        if len(batches) > 1:
            g_se = float(g.std(ddof=1) / np.sqrt(len(g)))
            r_se = float(reliability.std(ddof=1) / np.sqrt(len(reliability)))
            r_lower = max(0.0, float(reliability.mean() - one_sided * r_se))
            r_upper = min(1.0, float(reliability.mean() + one_sided * r_se))
        else:
            g_se = r_se = None
            r_lower = r_upper = None
        rows.append(
            dict(
                n=n,
                overproduction=n - TARGET_OUTPUT,
                cost_multiplier=n / TARGET_OUTPUT,
                mean_G=float(g.mean()),
                se_mean_G=g_se,
                mean_probability_at_least_50=float(reliability.mean()),
                se_probability_at_least_50=r_se,
                one_sided_95_lower=r_lower,
                one_sided_95_upper=r_upper,
            )
        )
    certified = next(
        (row["n"] for row in rows
         if row["one_sided_95_lower"] is not None and row["one_sided_95_lower"] >= 0.95),
        None,
    )
    return dict(
        completed_batches=len(batches),
        criterion=(
            "Smallest n whose one-sided 95% Student-t lower confidence bound for "
            "the unconditional probability of at least 50 conforming assemblies is >=0.95"
        ),
        certified_minimum_n=certified,
        results=rows,
    )


def write_report(summary, status, error=None):
    candidate = summary["certified_minimum_n"]
    if candidate is None:
        shown = summary["results"]
    else:
        shown = [row for row in summary["results"] if abs(row["n"] - candidate) <= 3]
    body = "".join(
        f"<tr><td>{row['n']}</td><td>{row['mean_G']:.3f}</td>"
        f"<td>{row['mean_probability_at_least_50']:.4f}</td>"
        f"<td>{'Pending' if row['one_sided_95_lower'] is None else f'{row['one_sided_95_lower']:.4f}'}</td>"
        f"<td>{row['overproduction']}</td><td>{row['cost_multiplier']:.2f}</td></tr>"
        for row in shown
    )
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Production allowance</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;line-height:1.55}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}</style>
<h1>Production allowance for 50 conforming assemblies</h1>
<p>Status: <strong>{html.escape(status)}</strong>. Completed extended batches: {summary['completed_batches']}.</p>
<p>Fixed conditions: r=2 um, wide-centered trivariate Gaussian w, M starting at 256, and no positive pair-probability threshold. Each saved 50x50 probability matrix is preserved and extended to 80x80 by computing only 3900 new entries.</p>
<p>For each n, the optimizer maximizes G. Conditional on the selected pairs and independent assembly outcomes, the probability of at least 50 conforming assemblies is computed exactly with the Poisson-binomial distribution. The reported service probability averages over random production batches.</p>
<table><tr><th>Produced n+n</th><th>Mean G</th><th>Mean Pr(S>=50)</th><th>One-sided 95% lower bound</th><th>Extra sets</th><th>Cost multiplier</th></tr>{body}</table>
<p>Current certified minimum n: <strong>{candidate if candidate is not None else 'Pending'}</strong>. The cost multiplier assumes equal per-set production cost and excludes fixed, inspection, and rework costs.</p>
<p>This is a synthetic-model planning result, not a guarantee for an unvalidated physical process. Independence of assembly outcomes is an explicit assumption.</p>
<p>{html.escape(error or '')}</p><p><a href="summary.json">Full summary</a> | <a href="batch_results.json">Batch results</a> | <a href="config.json">Configuration</a> | <a href="progress.json">Progress</a></p>
</html>"""
    temporary = OUT / "report.html.tmp"
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(OUT / "report.html")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-batches", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.target_batches <= 177:
        raise ValueError("target-batches must lie in [1, 177].")
    OUT.mkdir(parents=True, exist_ok=True)
    import msvcrt
    with (OUT / "run.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        execute(args.target_batches)


def execute(target_batches):
    config = dict(
        source=str(SOURCE),
        master_seed=MASTER_SEED,
        base_n=BASE_N,
        max_n=MAX_N,
        newly_computed_entries=MAX_N * MAX_N - BASE_N * BASE_N,
        target_output=TARGET_OUTPUT,
        radius_um=RADIUS_UM,
        model="Wide centered",
        mean_mm_mm_rad=WIDE_MEAN[0].tolist(),
        sigma_mm_mm_rad=WIDE_SIGMA[0].tolist(),
        workers=WORKERS,
        part_settings=asdict(CaseStudy()),
        assembly_outcomes_assumed_independent=True,
    )
    config_path = OUT / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise RuntimeError("Existing scientific configuration differs.")
    save_json(config_path, config)
    completed = []
    pool = ProcessPoolExecutor(max_workers=WORKERS)
    try:
        for batch in range(target_batches):
            started = time.perf_counter()
            result = extend_batch(batch, pool)
            completed.append(result)
            summary = confidence_summary(completed)
            save_json(OUT / "batch_results.json", completed)
            save_json(OUT / "summary.json", summary)
            progress = dict(
                status="Running" if batch + 1 < target_batches else "Completed requested batches",
                target_batches=target_batches,
                completed_batches=batch + 1,
                last_batch_seconds=time.perf_counter() - started,
                certified_minimum_n=summary["certified_minimum_n"],
                updated_local=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            save_json(OUT / "progress.json", progress)
            write_report(summary, progress["status"])
            print(
                f"Batch {batch + 1}/{target_batches} completed; "
                f"candidate={summary['certified_minimum_n']}; "
                f"seconds={progress['last_batch_seconds']:.1f}",
                flush=True,
            )
    except BaseException:
        error = traceback.format_exc()
        summary = confidence_summary(completed) if completed else dict(
            completed_batches=0, certified_minimum_n=None, results=[]
        )
        save_json(
            OUT / "progress.json",
            dict(status="Failed: review required", target_batches=target_batches,
                 completed_batches=len(completed), error=error),
        )
        write_report(summary, "Failed: review required", error)
        pool.terminate_workers()
        raise
    else:
        pool.shutdown()


if __name__ == "__main__":
    main()
