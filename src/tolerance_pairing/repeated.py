"""Threshold-aware circular-FC probability evaluation for repeated experiments."""

import numpy as np
from .core import build_polytope, functional_constraints
from .pilot import integrate_gaussians

RADII_UM = [2.5, 5.0]
MODEL_NAMES = ["uniform", "gaussian_centered_sigma2p5", "gaussian_centered_sigma5",
               "gaussian_shifted_x2p5_sigma5"]
MEANS = np.array([[0.,0.,0.], [0.,0.,0.], [.0025,0.,0.]])
SIGMAS = np.array([[.0025,.0025,12.5e-6], [.005,.005,25e-6], [.005,.005,25e-6]])


def ratio_envelope(numerator, numerator_error, denominator, denominator_error,
                   shell_volume, error_factor=10.):
    """Estimated envelope for the exact circular-FC Gaussian probability.

    Every unnormalized kernel is at most one, so its mass in the geometric
    shell is at most shell_volume. Cubature errors are inflated estimates,
    not rigorous enclosures; consequently the returned envelope is not certified.
    """
    denominator = np.asarray(denominator)
    lower_den = denominator-error_factor*np.asarray(denominator_error)
    if np.any(lower_den <= 0):
        raise RuntimeError("Denominator uncertainty is too large.")
    lower = np.maximum(0.,(numerator-error_factor*numerator_error)/
                       (denominator+error_factor*denominator_error))
    upper = np.minimum(1.,(numerator+error_factor*numerator_error+shell_volume)/lower_den)
    if np.any(lower>upper+1e-9) or np.any(lower>1+1e-8):
        raise RuntimeError("Inconsistent probability envelope.")
    return np.clip(lower,0,1),np.clip(upper,0,1)


def evaluate_radii(poly, radii_um=RADII_UM, p_min=.9, base_sides=256, max_sides=4096):
    """Refine FC polygons until threshold classification is numerically resolved.

    The point estimate is the inscribed-polygon probability. The circular-FC
    envelope uses an outer polygon and a Gaussian shell-mass upper estimate.
    Exact containment is checked against the circle, not just the inner polygon.
    Unresolved cases at max_sides are retained and explicitly marked.
    """
    if poly.status != "full_dimensional":
        return []
    if base_sides<3 or max_sides<base_sides or not 0<p_min<1:
        raise ValueError("Invalid refinement settings.")
    denominator_cache = {}

    def denominator(tol):
        if tol not in denominator_cache:
            denominator_cache[tol] = integrate_gaussians(poly,MEANS,SIGMAS,rtol=tol)[:2]
        return denominator_cache[tol]

    results = []
    for radius_um in radii_um:
        radius = radius_um/1000
        if np.max(np.linalg.norm(poly.physical_vertices[:,:2],axis=1)) <= radius:
            results.append({"radius_um":radius_um,"p":[1.]*4,"lower":[1.]*4,"upper":[1.]*4,
                "sides":0,"unresolved":[False]*4,"early_exit":"circular_containment","rtol":0.})
            continue
        sides = base_sides
        tol = 1e-5
        while True:
            AF, b_inner = functional_constraints(radius,sides)
            inner = build_polytope(np.vstack((poly.A,AF)),np.r_[poly.b,b_inner],poly.scale)
            # Same facet normals, offset r: a circumscribed polygon contains the circle.
            outer = build_polytope(np.vstack((poly.A,AF)),np.r_[poly.b,np.full(sides,radius)],poly.scale)
            if outer.status != "full_dimensional":
                p = lower = upper = np.zeros(4)
                unresolved = np.zeros(4,bool)
                early = "outer_intersection_zero_mass"
                break
            shell = max(0.,outer.volume-inner.volume)
            den,de = denominator(tol)
            num,ne,_ = integrate_gaussians(inner,MEANS,SIGMAS,rtol=tol)
            raw = num/den
            if tol>1e-8 and np.any((raw<0)|(raw>1)):
                tol=1e-8
                continue
            gl,gu = ratio_envelope(num,ne,den,de,shell)
            lower = np.r_[max(0.,inner.volume/poly.volume-1e-12),gl]
            upper = np.r_[min(1.,outer.volume/poly.volume+1e-12),gu]
            p = np.clip(np.r_[inner.volume/poly.volume,raw],0,1)
            unresolved = (lower<p_min)&(upper>=p_min)
            early = "none"
            if not np.any(unresolved):
                break
            if tol>1e-8:
                tol=1e-8
                continue
            if sides>=max_sides:
                break
            sides=min(2*sides,max_sides)
        results.append({"radius_um":radius_um,"p":p.tolist(),"lower":lower.tolist(),
            "upper":upper.tolist(),"sides":sides,"unresolved":unresolved.tolist(),
            "early_exit":early,"rtol":tol})
    return results
