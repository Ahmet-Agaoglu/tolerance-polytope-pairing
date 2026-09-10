"""Final numerical-integration and one-batch computational-scaling checks."""

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tolerance_pairing.core import (
    CaseStudy,
    build_polytope,
    functional_constraints,
    gaussian_weight,
    generate_pair,
    integrate_mass,
    pair_constraints,
)
from tolerance_pairing.convergence import MEANS, MODELS, SIGMAS
from tolerance_pairing.pilot import integrate_gaussians, optimal_matching


SOURCE = ROOT / "results" / "convergence_five_n50_seed20260907"
OUT = ROOT / "results" / "final_validation_checks"
SCALE = np.array([0.001, 0.001, 0.001 / 200])
RADIUS_UM = 2.0
SIDES = 256
WORKERS = 12
SIZES = (25, 50, 100)
SEED = 20260908
BASE_RTOL = 1e-5
STRICT_RTOL = 1e-8
AF, BF = functional_constraints(RADIUS_UM / 1000, SIDES)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def pair_probabilities(pin, hole, nominal):
    """Evaluate the three fixed Gaussian models with no threshold refinement."""
    poly = build_polytope(*pair_constraints(pin, hole, nominal), SCALE)
    if poly.status == "lower_dimensional":
        raise RuntimeError("Lower-dimensional P requires separate measure handling.")
    if poly.status != "full_dimensional":
        return np.zeros(3)
    if np.max(np.linalg.norm(poly.physical_vertices[:, :2], axis=1)) <= RADIUS_UM / 1000:
        return np.ones(3)
    clipped = build_polytope(np.vstack((poly.A, AF)), np.r_[poly.b, BF], SCALE)
    if clipped.status == "lower_dimensional":
        raise RuntimeError("Lower-dimensional P intersection requires separate measure handling.")
    if clipped.status != "full_dimensional":
        return np.zeros(3)
    denominator, _, _ = integrate_gaussians(poly, MEANS, SIGMAS, rtol=BASE_RTOL)
    numerator, _, _ = integrate_gaussians(clipped, MEANS, SIGMAS, rtol=BASE_RTOL)
    if np.any(denominator <= 0):
        raise RuntimeError("Nonpositive Gaussian normalization mass.")
    probability = numerator / denominator
    if np.any((probability < -1e-7) | (probability > 1 + 1e-7)):
        raise RuntimeError("Probability is outside [0, 1].")
    return np.clip(probability, 0, 1)


def run_row(task):
    i, pin, holes, nominal = task
    started = time.perf_counter()
    row = np.array([pair_probabilities(pin, hole, nominal) for hole in holes])
    return i, row, time.perf_counter() - started


def prepare_parts():
    path = OUT / "scale_parts_n100.npz"
    if not path.exists():
        children = np.random.SeedSequence(SEED).spawn(max(SIZES))
        parts = [generate_pair(seed, CaseStudy()) for seed in children]
        np.savez_compressed(
            path,
            pins=np.array([part[0] for part in parts]),
            holes=np.array([part[1] for part in parts]),
            nominal=parts[0][2],
        )
    with np.load(path) as data:
        return data["pins"], data["holes"], data["nominal"]


def run_scaling_check():
    pins, holes, nominal = prepare_parts()
    results = []
    for n in SIZES:
        folder = OUT / f"scale_n{n:03d}"
        folder.mkdir(exist_ok=True)
        probability_path = folder / "probabilities.npz"
        timing_path = folder / "timing.json"
        if probability_path.exists() and timing_path.exists():
            results.append(json.loads(timing_path.read_text(encoding="utf-8")))
            continue

        started = time.perf_counter()
        probabilities = np.zeros((3, n, n))
        child_cpu_seconds = 0.0
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            futures = [
                pool.submit(run_row, (i, pins[i], holes[:n], nominal))
                for i in range(n)
            ]
            for future in as_completed(futures):
                i, row, row_seconds = future.result()
                probabilities[:, i, :] = row.T
                child_cpu_seconds += row_seconds
        probability_seconds = time.perf_counter() - started

        matching_started = time.perf_counter()
        model_results = []
        for model, matrix in zip(MODELS, probabilities):
            allowed = matrix > 0
            pairs = optimal_matching(matrix, allowed, np.nextafter(0.0, 1.0)).reshape(-1, 2)
            ii, jj = pairs[:, 0], pairs[:, 1]
            model_results.append(
                dict(model=model, N=len(pairs), G=float(matrix[ii, jj].sum()))
            )
        matching_seconds = time.perf_counter() - matching_started
        total_seconds = probability_seconds + matching_seconds
        np.savez_compressed(probability_path, p=probabilities)
        result = dict(
            n=n,
            candidate_pairs=n * n,
            workers=WORKERS,
            probability_seconds=probability_seconds,
            matching_seconds=matching_seconds,
            total_seconds=total_seconds,
            milliseconds_per_pair=1000 * probability_seconds / (n * n),
            summed_worker_seconds=child_cpu_seconds,
            model_results=model_results,
        )
        save_json(timing_path, result)
        results.append(result)
        print(f"Scaling n={n} completed in {total_seconds:.1f} s", flush=True)
    return results


def representative_pairs(saved):
    """Select deterministic low, middle, and high nontrivial probabilities."""
    selected = []
    used = set()
    for model in range(3):
        matrix = saved[model]
        candidates = np.argwhere((matrix > 1e-6) & (matrix < 1 - 1e-6))
        for target in (0.1, 0.5, 0.9):
            order = np.argsort(np.abs(matrix[candidates[:, 0], candidates[:, 1]] - target))
            for index in order:
                i, j = map(int, candidates[index])
                if (i, j) not in used:
                    used.add((i, j))
                    selected.append((model, target, i, j))
                    break
    return selected


def run_integration_check():
    path = OUT / "integration_audit.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    with np.load(SOURCE / "batch_000" / "probabilities.npz") as data:
        saved = data["p"][:3]
    with np.load(SOURCE / "batch_000" / "parts.npz") as data:
        pins, holes, nominal = data["pins"], data["holes"], data["nominal"]

    records = []
    for model, target, i, j in representative_pairs(saved):
        poly = build_polytope(*pair_constraints(pins[i], holes[j], nominal), SCALE)
        clipped = build_polytope(np.vstack((poly.A, AF)), np.r_[poly.b, BF], SCALE)
        if poly.status != "full_dimensional" or clipped.status != "full_dimensional":
            raise RuntimeError("A representative pair unexpectedly lacks positive volume.")

        baseline_den, baseline_den_error, baseline_den_sub = integrate_gaussians(
            poly, MEANS[model:model + 1], SIGMAS[model:model + 1], rtol=BASE_RTOL
        )
        baseline_num, baseline_num_error, baseline_num_sub = integrate_gaussians(
            clipped, MEANS[model:model + 1], SIGMAS[model:model + 1], rtol=BASE_RTOL
        )
        weight = gaussian_weight(MEANS[model], SIGMAS[model])
        strict_den = integrate_mass(poly, weight, rtol=STRICT_RTOL, atol=0)
        strict_num = integrate_mass(clipped, weight, rtol=STRICT_RTOL, atol=0)
        baseline_probability = float(baseline_num[0] / baseline_den[0])
        strict_probability = float(strict_num["mass"] / strict_den["mass"])
        records.append(
            dict(
                model=MODELS[model],
                target_probability=target,
                pair_zero_based=[i, j],
                saved_probability=float(saved[model, i, j]),
                baseline_probability=baseline_probability,
                strict_probability=strict_probability,
                absolute_change=abs(strict_probability - baseline_probability),
                baseline_numerator_error_estimate=float(baseline_num_error[0]),
                baseline_denominator_error_estimate=float(baseline_den_error[0]),
                baseline_subdivisions=int(baseline_num_sub + baseline_den_sub),
                strict_numerator_error_estimate=strict_num["estimated_error"],
                strict_denominator_error_estimate=strict_den["estimated_error"],
                strict_subdivisions=int(strict_num["subdivisions"] + strict_den["subdivisions"]),
            )
        )
        print(f"Integral audit {len(records)}/9 completed", flush=True)

    result = dict(
        batch=0,
        radius_um=RADIUS_UM,
        polygon_sides=SIDES,
        baseline_rtol=BASE_RTOL,
        strict_rtol=STRICT_RTOL,
        selection="Nearest nontrivial pair to p=0.1, 0.5, and 0.9 for each model",
        checked_probabilities=len(records),
        max_absolute_change=max(record["absolute_change"] for record in records),
        mean_absolute_change=float(np.mean([record["absolute_change"] for record in records])),
        records=records,
        interpretation=(
            "This isolates integration tolerance at fixed M=256. Polygon resolution was "
            "validated separately by the completed M-versus-2M audit."
        ),
    )
    save_json(path, result)
    return result


def write_report(integration, scaling, elapsed_seconds):
    integral_rows = "".join(
        f"<tr><td>{row['model']}</td><td>{row['target_probability']:.1f}</td>"
        f"<td>({row['pair_zero_based'][0]}, {row['pair_zero_based'][1]})</td>"
        f"<td>{row['baseline_probability']:.8f}</td><td>{row['strict_probability']:.8f}</td>"
        f"<td>{row['absolute_change']:.2e}</td></tr>"
        for row in integration["records"]
    )
    scaling_rows = "".join(
        f"<tr><td>{row['n']}</td><td>{row['candidate_pairs']}</td>"
        f"<td>{row['probability_seconds']:.2f}</td><td>{row['matching_seconds']:.4f}</td>"
        f"<td>{row['total_seconds']:.2f}</td><td>{row['milliseconds_per_pair']:.2f}</td></tr>"
        for row in scaling
    )
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Final numerical validation checks</title>
<style>body{{font:16px system-ui;max-width:1150px;margin:32px auto;line-height:1.55}}table{{border-collapse:collapse;width:100%;margin-bottom:28px}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}code{{background:#f4f4f4;padding:2px 4px}}</style>
<h1>Final numerical validation checks</h1>
<p>Fixed conditions: r=2 um, M=256, three trivariate Gaussian process models, and no positive pair-probability threshold.</p>
<h2>Gaussian integration tolerance</h2>
<p>Nine deterministic representative probabilities were recomputed. The baseline uses shared vector cubature at relative tolerance 1e-5; the strict calculation uses the independent per-tetrahedron implementation at 1e-8. Geometry is held fixed to isolate integration error.</p>
<table><tr><th>Model</th><th>Target p</th><th>Pair</th><th>Baseline p</th><th>Strict p</th><th>|change|</th></tr>{integral_rows}</table>
<p>Maximum absolute probability change: <strong>{integration['max_absolute_change']:.3e}</strong>; mean absolute change: {integration['mean_absolute_change']:.3e}. This is a targeted numerical audit, not a certified global error bound or empirical validation.</p>
<h2>One-batch computational scaling</h2>
<p>Each size performs all n^2 polytope and three-model probability calculations from scratch on the same computer with {WORKERS} worker processes. Assignment is then solved separately for all three models. The nested synthetic part sample is fixed by seed {SEED}. Timings are descriptive hardware-dependent measurements, not inferential estimates.</p>
<table><tr><th>n</th><th>Candidate pairs</th><th>Probability time [s]</th><th>Three matchings [s]</th><th>Total [s]</th><th>Probability ms/pair</th></tr>{scaling_rows}</table>
<p>The quadratic growth in candidate pairs dominates total time; assignment time is comparatively negligible. Total script elapsed time: {elapsed_seconds:.2f} s.</p>
<p><a href="integration_audit.json">Integration details</a> | <a href="scaling_summary.json">Scaling details</a> | <a href="config.json">Configuration</a></p>
</html>"""
    (OUT / "report.html").write_text(document, encoding="utf-8")


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    config = dict(
        radius_um=RADIUS_UM,
        polygon_sides=SIDES,
        models=MODELS,
        means_mm_mm_rad=MEANS.tolist(),
        sigmas_mm_mm_rad=SIGMAS.tolist(),
        baseline_rtol=BASE_RTOL,
        strict_rtol=STRICT_RTOL,
        scale_sizes=SIZES,
        scale_seed=SEED,
        workers=WORKERS,
        part_settings=asdict(CaseStudy()),
        python=sys.executable,
        python_version=sys.version,
        platform=platform.platform(),
        processor=platform.processor(),
    )
    save_json(OUT / "config.json", config)
    integration = run_integration_check()
    scaling = run_scaling_check()
    save_json(OUT / "scaling_summary.json", dict(results=scaling))
    write_report(integration, scaling, time.perf_counter() - started)
    print(json.dumps(dict(integration=integration, scaling=scaling), indent=2), flush=True)


if __name__ == "__main__":
    main()
