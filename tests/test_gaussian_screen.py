"""Regression checks for the Gaussian-only exploratory evaluator."""

import unittest
import numpy as np
from scipy.special import erf
from tolerance_pairing.core import build_polytope
from tolerance_pairing.gaussian_screen import parameter_grid, evaluate


class GaussianScreenTests(unittest.TestCase):
    def test_distinct_parameter_grid(self):
        models, means, sigmas = parameter_grid()
        self.assertEqual(len(models), 17)
        self.assertEqual(len({tuple(x.values()) for x in models}), 17)
        self.assertEqual(means.shape, sigmas.shape)

    def test_analytic_circular_gaussian(self):
        poly = build_polytope(np.vstack((np.eye(3), -np.eye(3))),
                             np.array([.005, .005, 2e-5]*2), np.array([.001, .001, 1e-5]))
        result = evaluate(poly, np.zeros((1, 3)), np.array([[.0025, .0025, 25e-6]]))
        exact = (1-np.exp(-.002**2/(2*.0025**2)))/erf(.005/(np.sqrt(2)*.0025))**2
        self.assertLessEqual(result['lower'][0], exact)
        self.assertGreaterEqual(result['upper'][0], exact)

    def test_containment_model_count(self):
        poly = build_polytope(np.vstack((np.eye(3), -np.eye(3))),
                             np.array([.001, .001, 2e-5]*2), np.array([.001, .001, 1e-5]))
        _, means, sigmas = parameter_grid()
        self.assertEqual(evaluate(poly, means, sigmas)['p'], [1.]*17)

    def test_theta_factor_cancels_for_cartesian_box(self):
        poly = build_polytope(np.vstack((np.eye(3), -np.eye(3))),
                             np.array([.005, .005, 2e-5]*2), np.array([.001, .001, 1e-5]))
        means = np.array([[0., 0., x] for x in (0., 12.5e-6, 25e-6)])
        sigmas = np.array([[.0025, .0025, x] for x in (10e-6, 25e-6, 50e-6)])
        result = evaluate(poly, means, sigmas)
        np.testing.assert_allclose(result['p'], result['p'][0], atol=2e-6)
