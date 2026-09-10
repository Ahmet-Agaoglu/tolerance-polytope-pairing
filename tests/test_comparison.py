"""Tests for random-pairing comparison metrics."""
import unittest
import numpy as np
from tolerance_pairing.comparison import expected_random_g


class ComparisonTests(unittest.TestCase):
    def test_expected_random_g(self):
        p=np.array([[1.,0.],[.25,.75]])
        self.assertAlmostEqual(expected_random_g(p),1.)

    def test_identity_matrix(self):
        self.assertAlmostEqual(expected_random_g(np.eye(5)),1.)

    def test_rejects_nonsquare(self):
        with self.assertRaises(ValueError): expected_random_g(np.ones((2,3)))
