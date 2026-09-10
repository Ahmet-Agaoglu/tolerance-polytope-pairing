"""Analytical checks independent of the flange example."""

import unittest
import numpy as np
from scipy.special import erf
from tolerance_pairing.core import (
    build_polytope, functional_constraints, gaussian_weight, integrate_mass,
)


class GeometryAndIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.A = np.vstack((np.eye(3), -np.eye(3)))
        self.b = np.ones(6)
        self.box = build_polytope(self.A, self.b, np.ones(3))

    def test_box_volume_and_constant_integral(self):
        self.assertAlmostEqual(self.box.volume, 8.0, places=10)
        result = integrate_mass(self.box, lambda x: np.ones(len(x)))
        self.assertAlmostEqual(result["mass"], 8.0, places=8)

    def test_gaussian_against_separable_analytic_integral(self):
        actual = integrate_mass(self.box, gaussian_weight(np.zeros(3), np.ones(3)))["mass"]
        expected = (np.sqrt(2*np.pi) * erf(1/np.sqrt(2)))**3
        self.assertAlmostEqual(actual, expected, places=7)

    def test_all_vertices_outside_does_not_imply_empty_intersection(self):
        AF, bF = functional_constraints(0.5, 32)
        clipped = build_polytope(np.vstack((self.A, AF)), np.r_[self.b, bF], np.ones(3))
        self.assertTrue(np.all(np.linalg.norm(self.box.vertices[:, :2], axis=1) > 0.5))
        expected = 32 * 0.5**2 * np.sin(2*np.pi/32)
        self.assertAlmostEqual(clipped.volume, expected, places=9)

    def test_empty_is_not_a_solver_success(self):
        empty = build_polytope(np.vstack((self.A, [1,0,0])), np.r_[self.b, -2], np.ones(3))
        self.assertEqual(empty.status, "empty")

    def test_lower_dimensional_is_distinct(self):
        flat = build_polytope(self.A, np.array([1,1,0,1,1,0]), np.ones(3))
        self.assertEqual(flat.status, "lower_dimensional")


if __name__ == "__main__":
    unittest.main()
