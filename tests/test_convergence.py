"""Tests for the fixed design and stopping schedule."""
import unittest
import numpy as np
from tolerance_pairing.convergence import diagnostics, SCENARIOS, threshold_robust_matching


class ConvergenceTests(unittest.TestCase):
    def test_design(self):
        self.assertEqual(len(SCENARIOS), 5)
        self.assertEqual([s['model_index'] for s in SCENARIOS], [0, 1, 2, 1, 1])

    def test_earliest_stop(self):
        self.assertFalse(diagnostics(np.full((30, 5), 42.))['stop'])
        self.assertTrue(diagnostics(np.full((40, 5), 42.))['stop'])
        self.assertFalse(diagnostics(np.full((41, 5), 42.))['stop'])

    def test_precision(self):
        x = np.tile([0., 50.], 20)[:, None]
        self.assertFalse(diagnostics(x)['stop'])

    def test_instability(self):
        x = np.r_[np.full(30, 40.), np.full(10, 41.)][:, None]
        result = diagnostics(x)
        self.assertLess(result['standard_error'][0], .25)
        self.assertFalse(result['stop'])

    def test_one_batch(self):
        self.assertIsNone(diagnostics([[40.]])['standard_error'])

    def test_immaterial_ambiguous_edge(self):
        p = np.array([[.95, .899999], [.94, .93]])
        lower = p.copy(); lower[0, 1] = .899998
        upper = p.copy(); upper[0, 1] = .900001
        pairs, ambiguous = threshold_robust_matching(p, np.ones((2, 2), bool), lower, upper)
        self.assertEqual(ambiguous, 1)
        self.assertEqual(set(map(tuple, pairs)), {(0, 0), (1, 1)})

    def test_material_ambiguous_edge(self):
        p = np.array([[.91, .899999], [.899999, .20]])
        lower = p.copy(); lower[0, 1] = lower[1, 0] = .899998
        upper = p.copy(); upper[0, 1] = upper[1, 0] = .900001
        with self.assertRaises(RuntimeError):
            threshold_robust_matching(p, np.ones((2, 2), bool), lower, upper)
