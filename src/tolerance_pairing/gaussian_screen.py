"""Gaussian-only process grid and threshold-aware FC integration."""

import numpy as np
from .core import build_polytope, functional_constraints
from .pilot import integrate_gaussians
from .repeated import ratio_envelope


def parameter_grid():
    models = []
    for sigma in (1.5, 3., 6.):
        for mean in (0., 2., 4.):
            models.append(dict(mu_x_um=mean, mu_theta_urad=0.,
                               sigma_xy_um=sigma, sigma_theta_urad=25.))
    for sigma in (10., 25., 50.):
        for mean in (0., 12.5, 25.):
            model = dict(mu_x_um=0., mu_theta_urad=mean,
                         sigma_xy_um=3., sigma_theta_urad=sigma)
            if model not in models:
                models.append(model)
    means = np.array([[x['mu_x_um']/1000, 0., x['mu_theta_urad']/1e6] for x in models])
    sigmas = np.array([[x['sigma_xy_um']/1000]*2+[x['sigma_theta_urad']/1e6] for x in models])
    return models, means, sigmas


def masses(poly, means, sigmas, tol):
    """Small vector groups prevent one sharp density driving all refinements."""
    values, errors = [], []
    for start in range(0, len(means), 3):
        value, error, _ = integrate_gaussians(poly, means[start:start+3], sigmas[start:start+3], rtol=tol)
        values.extend(value); errors.extend(error)
    return np.array(values), np.array(errors)


def evaluate(poly, means, sigmas, radius_um=2., p_min=.9):
    count = len(means)
    if poly.status != 'full_dimensional':
        return None
    if np.max(np.linalg.norm(poly.physical_vertices[:, :2], axis=1)) <= radius_um/1000:
        return dict(p=[1.]*count, lower=[1.]*count, upper=[1.]*count,
                    unresolved=[False]*count, sides=0, early_exit='containment')
    den = de = None
    tight_den = np.zeros(count, bool)
    # A bounding box encloses P. The Gaussian maximum on this box bounds
    # its maximum in the outer-minus-inner shell, often much more tightly than 1.
    low, high = poly.physical_vertices.min(axis=0), poly.physical_vertices.max(axis=0)
    distance = np.maximum(np.maximum(low-means, means-high), 0.)/sigmas
    kernel_ceiling = np.exp(-.5*np.sum(distance**2, axis=1))
    for sides in (256, 512, 1024, 2048, 4096):
        AF, bf = functional_constraints(radius_um/1000, sides)
        inner = build_polytope(np.vstack((poly.A, AF)), np.r_[poly.b, bf], poly.scale)
        outer = build_polytope(np.vstack((poly.A, AF)),
            np.r_[poly.b, np.full(sides, radius_um/1000)], poly.scale)
        if outer.status != 'full_dimensional':
            return dict(p=[0.]*count, lower=[0.]*count, upper=[0.]*count,
                        unresolved=[False]*count, sides=sides, early_exit='zero_mass')
        if den is None:
            den, de = masses(poly, means, sigmas, 1e-5)
        if np.any(den <= 0) or np.any(~np.isfinite(den)):
            raise RuntimeError('Gaussian normalization is not numerically positive.')
        num, ne = masses(inner, means, sigmas, 1e-5)
        shell = max(0., outer.volume-inner.volume)*kernel_ceiling
        p = num/den
        lower, upper = ratio_envelope(num, ne, den, de, shell)
        needs = ((lower<p_min)&(upper>=p_min)) | (p<0) | (p>1)
        if np.any(needs):
            update = needs & ~tight_den
            if np.any(update):
                den[update], de[update] = masses(poly, means[update], sigmas[update], 1e-8)
                tight_den[update] = True
            num[needs], ne[needs] = masses(inner, means[needs], sigmas[needs], 1e-8)
            lower, upper = ratio_envelope(num, ne, den, de, shell)
            p = num/den
        ambiguous = (lower<p_min)&(upper>=p_min)
        if np.any(ambiguous):
            # Direct outer mass resolves overly conservative shell bounds before
            # paying for a finer polygon. These remain estimated, not certified, bounds.
            om, oe = masses(outer, means[ambiguous], sigmas[ambiguous], 1e-8)
            upper[ambiguous] = np.minimum(upper[ambiguous],
                np.clip((om+10*oe)/(den[ambiguous]-10*de[ambiguous]), 0, 1))
        unresolved = (lower<p_min)&(upper>=p_min)
        if np.any((p < -1e-7) | (p > 1+1e-7)):
            raise RuntimeError('Gaussian probability range validation failed.')
        if np.any(lower > upper+1e-7):
            raise RuntimeError('Probability envelopes are inconsistent.')
        if not np.any(unresolved):
            break
    return dict(p=np.clip(p, 0, 1).tolist(), lower=lower.tolist(), upper=upper.tolist(),
                unresolved=unresolved.tolist(), sides=sides, early_exit='none')
