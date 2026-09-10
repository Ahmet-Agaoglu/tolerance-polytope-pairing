"""Fixed reduced design and practical Monte Carlo stopping diagnostics."""

import numpy as np
from scipy.optimize import linear_sum_assignment

MODELS = ['Wide centered', 'Narrow centered', 'Narrow shifted']
MEANS = np.array([[0., 0., 0.], [0., 0., 0.], [.004, 0., 0.]])
SIGMAS = np.array([[.006, .006, 25e-6], [.0015, .0015, 25e-6],
                   [.0015, .0015, 25e-6]])
SCENARIOS = [dict(radius_um=r, model=MODELS[m], model_index=m)
             for r, m in [(2., 0), (2., 1), (2., 2), (2.5, 1), (3., 1)]]


def threshold_robust_matching(probabilities, feasible, lower, upper, p_min=.9,
                              objective_tolerance=1e-9):
    """Return the conservative optimum if ambiguous edges cannot improve it."""
    p = np.asarray(probabilities, dtype=float)
    feasible = np.asarray(feasible, dtype=bool)
    conservative = feasible & (np.asarray(lower) >= p_min)
    permissive = feasible & (np.asarray(upper) >= p_min)

    def solve(mask):
        n, m = p.shape
        weights = np.concatenate((np.where(mask, p, -1e6), np.zeros((n, n))), axis=1)
        row, col = linear_sum_assignment(weights, maximize=True)
        real = col < m
        pairs = np.column_stack((row[real], col[real]))
        return pairs, float(p[pairs[:, 0], pairs[:, 1]].sum())

    certain_pairs, certain_g = solve(conservative)
    _, permissive_g = solve(permissive)
    if permissive_g > certain_g + objective_tolerance:
        raise RuntimeError('Numerically ambiguous threshold edge can change the optimum.')
    return certain_pairs, int(np.sum(permissive & ~conservative))


def diagnostics(values):
    """Rows are independent batches; columns are the fixed scenarios.

    Repeated checks are practical diagnostics, not confidence sequences.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError('Expected a nonempty finite batch-by-scenario array.')
    b = len(x)
    checkpoints = list(range(10, b+1, 10))
    changes = [np.abs(x[:right].mean(0)-x[:left].mean(0)).tolist()
               for left, right in zip(checkpoints[:-1], checkpoints[1:])]
    se = x.std(axis=0, ddof=1)/np.sqrt(b) if b > 1 else None
    stable = (np.max(changes[-3:], axis=0) <= .10 if len(changes) >= 3
              else np.zeros(x.shape[1], dtype=bool))
    passed = (stable & (se <= .25) if se is not None else stable)
    return dict(batches=b, mean=x.mean(0).tolist(),
                standard_error=None if se is None else se.tolist(),
                checkpoints=checkpoints, last_three_changes=changes[-3:],
                se_criterion_passed=(np.zeros(x.shape[1], dtype=bool) if se is None
                                     else se <= .25).tolist(),
                stability_diagnostic_passed=stable.tolist(),
                scenario_passed=passed.tolist(),
                stop=bool(b >= 30 and b % 10 == 0 and np.all(passed)))
