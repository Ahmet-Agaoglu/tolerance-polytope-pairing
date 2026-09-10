"""Tests of threshold envelopes and circular-FC exits."""

import unittest
import numpy as np
from scipy.special import erf
from tolerance_pairing.core import build_polytope
from tolerance_pairing.repeated import evaluate_radii, ratio_envelope


class RepeatedTests(unittest.TestCase):
    def test_shell_envelope(self):
        lo,hi=ratio_envelope(np.array([.8]),np.zeros(1),np.ones(1),np.zeros(1),.15)
        self.assertAlmostEqual(lo[0],.8)
        self.assertAlmostEqual(hi[0],.95)

    def test_exact_containment(self):
        poly=build_polytope(np.vstack((np.eye(3),-np.eye(3))),np.array([.001,.001,1e-5]*2),np.array([.001,.001,1e-5]))
        result=evaluate_radii(poly,[2.5])[0]
        self.assertEqual(result["p"],[1.]*4)
        self.assertEqual(result["early_exit"],"circular_containment")

    def test_inner_outer_enclose_disk_volume(self):
        poly=build_polytope(np.vstack((np.eye(3),-np.eye(3))),np.array([.005,.005,2e-5]*2),np.array([.001,.001,1e-5]))
        result=evaluate_radii(poly,[2.5],base_sides=32,max_sides=64)[0]
        exact=np.pi*.0025**2/(4*.005**2)
        self.assertLess(result["lower"][0],exact)
        self.assertGreater(result["upper"][0],exact)

    def test_unresolved_is_not_silently_accepted(self):
        poly=build_polytope(np.vstack((np.eye(3),-np.eye(3))),np.array([.005,.005,2e-5]*2),np.array([.001,.001,1e-5]))
        exact=np.pi*.0025**2/(4*.005**2)
        result=evaluate_radii(poly,[2.5],p_min=exact,base_sides=8,max_sides=8)[0]
        self.assertTrue(result["unresolved"][0])

    def test_gaussian_circular_probability_against_analytic_result(self):
        poly=build_polytope(np.vstack((np.eye(3),-np.eye(3))),np.array([.005,.005,2e-5]*2),np.array([.001,.001,1e-5]))
        result=evaluate_radii(poly,[2.5],base_sides=64,max_sides=64)[0]
        # The independent theta factor cancels. Integrate the isotropic xy
        # Gaussian analytically over the disk and its bounding square.
        sigma=.0025
        exact=(1-np.exp(-.0025**2/(2*sigma**2)))/erf(.005/(np.sqrt(2)*sigma))**2
        self.assertLessEqual(result["lower"][1],exact)
        self.assertGreaterEqual(result["upper"][1],exact)

    def test_outside_polytope_has_zero_probability(self):
        # x in [9,11] µm; y in [-1,1] µm. The FC disk has radius 2.5 µm.
        poly=build_polytope(np.vstack((np.eye(3),-np.eye(3))),np.array([.011,.001,2e-5,-.009,.001,2e-5]),np.array([.001,.001,1e-5]))
        result=evaluate_radii(poly,[2.5])[0]
        self.assertEqual(result["p"],[0.]*4)
        self.assertEqual(result["early_exit"],"outer_intersection_zero_mass")
