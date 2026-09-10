"""Independent tests of vector cubature and partial weighted matching."""

import unittest
import numpy as np
from scipy.special import erf
from tolerance_pairing.core import build_polytope
from tolerance_pairing.pilot import integrate_gaussians, optimal_matching


class PilotTests(unittest.TestCase):
    def test_vector_gaussian_box(self):
        box = build_polytope(np.vstack((np.eye(3), -np.eye(3))), np.ones(6), np.ones(3))
        means = np.array([[0., 0, 0], [.3, -.2, .1]])
        sigmas = np.array([[1., 1, 1], [.8, 1.2, 2.]])
        value, error, _ = integrate_gaussians(box, means, sigmas, rtol=1e-7)
        exact = np.prod(sigmas*np.sqrt(np.pi/2)*(erf((1-means)/(np.sqrt(2)*sigmas))-
                         erf((-1-means)/(np.sqrt(2)*sigmas))), axis=1)
        np.testing.assert_allclose(value, exact, rtol=2e-7)

    def test_partial_rectangular_matching(self):
        p = np.array([[.95, .99, .2], [.94, .1, .4]])
        pairs = optimal_matching(p, np.ones_like(p, dtype=bool), .90)
        self.assertEqual(set(map(tuple, pairs)), {(0, 1), (1, 0)})

    def test_no_allowed_edges(self):
        self.assertEqual(len(optimal_matching(np.zeros((2,3)), np.ones((2,3), bool), .9)), 0)

    def test_feasibility_exclusion(self):
        pairs = optimal_matching(np.array([[1., .95]]), np.array([[False, True]]), .9)
        self.assertEqual(pairs.tolist(), [[0, 1]])
