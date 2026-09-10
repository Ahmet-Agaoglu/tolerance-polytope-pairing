# Tolerance-Polytope Part Pairing

This repository contains the Python implementation and numerical results for process-informed optimal pairing of manufactured parts. Candidate assemblies are represented by three-dimensional tolerance polytopes. A realization density defined on each feasible polytope determines its functional-compliance probability, and a maximum-weight bipartite matching maximizes the expected number of conforming assemblies.

The repository is publication-venue independent. Manuscript source files and copyrighted source figures are intentionally excluded.

## Contents

- `src/tolerance_pairing/`: geometry, probability integration, matching, convergence, and comparison functions.
- `scripts/`: analysis and validation entry points used for the reported experiments.
- `tests/`: unit tests for geometry, integration, matching, and numerical controls.
- `results/`: machine-readable final summaries and batch-level outputs.

The internal computational coordinate order is `(translation_x, translation_y, rotation)`. Coordinate scaling is applied before geometric computations.

## Installation

Python 3.14 was used for the numerical study.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Add `src` to `PYTHONPATH`, or insert it into `sys.path` when running a script directly.

## Tests

From the repository root:

```bash
python -c "import sys,unittest; sys.path.insert(0,'src'); result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests')); sys.exit(not result.wasSuccessful())"
```

## Analysis workflow

The scripts document the final analysis sequence:

1. `reoptimize_saved_probabilities.py` maximizes expected conforming output without a positive probability threshold.
2. `compute_uniform_baseline.py` computes decisions under the uniform benchmark.
3. `compare_pairing_policies.py` compares process-informed, uniform-benchmark, and random pairing policies.
4. `estimate_production_allowance.py` estimates the production quantity required for a target conforming output.
5. `validate_numerics.py` performs integration and scale checks.
6. `plot_convergence.py` generates the convergence plot.

The complete checkpoint directory contains thousands of intermediate matrices and is not included because of its size. Final batch-level decisions, summaries, configurations, and validation outputs are retained under `results/`. Fixed random seeds and numerical settings are recorded in the scripts and JSON files.

## Core quantities

For candidate pair `(i,j)`, `p_ij` is the probability of satisfying the functional condition under the specified assembly-process realization model. If `z_ij` indicates a selected pair, the optimization maximizes

```text
G = sum_i sum_j p_ij z_ij
```

subject to one-to-one matching constraints. `N` is the number of selected assemblies, while `G` is their expected conforming output.

