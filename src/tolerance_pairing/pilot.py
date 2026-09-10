"""Batch probability integration and optional maximum-weight matching."""

import numpy as np
from scipy.integrate import cubature
from scipy.optimize import linear_sum_assignment


def integrate_gaussians(poly, means, sigmas, rtol=1e-5):
    """Integrate positive Gaussian kernels with shared adaptive cubature.

    Map every tetrahedron onto the unit cube, sum their transformed integrands,
    then integrate the vector of Gaussian masses. This is the same tetrahedral
    change of variables as core.integrate_mass, with shared adaptive refinement.
    Errors are estimates, not certified bounds. Return masses in physical units.
    """
    means, sigmas = np.asarray(means), np.asarray(sigmas)
    if means.shape != sigmas.shape or means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("Expected matching (n_models, 3) parameter arrays.")
    if np.any(sigmas <= 0):
        raise ValueError("Standard deviations must be positive.")
    if poly.status != "full_dimensional":
        return np.zeros(len(means)), np.zeros(len(means)), 0
    tetra = np.asarray(list(poly.tetrahedra()))
    bases = tetra[:, 1:] - tetra[:, :1]
    determinants = np.abs(np.linalg.det(bases))

    def integrand(points):
        u, v, t = points.T
        bary = np.column_stack((u, (1-u)*v, (1-u)*(1-v)*t))
        physical = (tetra[None, :, 0, :] + np.einsum("nk,tkd->ntd", bary, bases)) * poly.scale
        values = np.stack([
            np.exp(-.5*np.sum(((physical-mean)/sigma)**2, axis=2)) @ determinants
            for mean, sigma in zip(means, sigmas)
        ], axis=1)
        return values * ((1-u)**2*(1-v))[:, None]

    integral = cubature(integrand, np.zeros(3), np.ones(3), rule="genz-malik",
                        rtol=rtol, atol=0, max_subdivisions=10000)
    if integral.status != "converged":
        raise RuntimeError("Shared cubature did not converge.")
    jacobian = float(np.prod(poly.scale))
    return integral.estimate*jacobian, integral.error*jacobian, integral.subdivisions


def optimal_matching(probabilities, feasible, p_min=None):
    """Solve max sum(p*z) with optional unmatched parts and forbidden edges.

    Dummy columns represent unmatched pin parts. Real columns are hole parts;
    rectangular linear assignment enforces each real part's capacity of one.
    No secondary cardinality objective is added.
    """
    p = np.asarray(probabilities)
    allowed = np.asarray(feasible, dtype=bool) & (p > 0.0)
    if p_min is not None:
        allowed &= p >= p_min
    if p.ndim != 2 or allowed.shape != p.shape or np.any(~np.isfinite(p)):
        raise ValueError("Invalid probability matrix.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must lie in [0,1].")
    n, m = p.shape
    weights = np.concatenate((np.where(allowed, p, -1e6), np.zeros((n, n))), axis=1)
    row, col = linear_sum_assignment(weights, maximize=True)
    real = col < m
    pairs = np.column_stack((row[real], col[real]))
    if len(pairs) and not np.all(allowed[pairs[:, 0], pairs[:, 1]]):
        raise RuntimeError("A forbidden edge was selected.")
    return pairs
