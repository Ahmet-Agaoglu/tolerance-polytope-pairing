"""Comparison metrics for complete random one-to-one pairing."""
import numpy as np


def expected_random_g(probabilities):
    """Expected G for a uniformly random perfect matching of a square matrix."""
    p=np.asarray(probabilities,dtype=float)
    if p.ndim!=2 or p.shape[0]!=p.shape[1] or np.any(~np.isfinite(p)):
        raise ValueError('Expected a finite square probability matrix.')
    if np.any((p<0)|(p>1)):
        raise ValueError('Probabilities must be in [0,1].')
    return float(p.sum()/p.shape[0])
