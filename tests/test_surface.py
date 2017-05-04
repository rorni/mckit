# -*- coding: utf-8 -*-

import unittest

import numpy as np

from mckit.transformation import Transformation
from mckit.surface import Plane


tr_glob = Transformation(translation=[1, 2, -3], indegrees=True,
                         rotation=[30, 60, 90, 120, 30, 90, 90, 90, 0])


class TestPlaneSurface(unittest.TestCase):
    def test_plane_creation(self):
        for i, (v, k, tr, v_ref, k_ref) in enumerate(plane_creation_cases):
            with self.subTest(i=i):
                p = Plane(v, k, transform=tr)
                self.assertAlmostEqual(p._k, k_ref)
                for j in range(3):
                    self.assertAlmostEqual(p._v[j], v_ref[j])


plane_creation_cases = [
    (np.array([0, 0, 1]), -2, None, np.array([0, 0, 1]), -2),
    (np.array([0, 0, 1]), -2, tr_glob, np.array([0, 0, 1]), 1),
    (np.array([1, 0, 0]), -2, None, np.array([1, 0, 0]), -2),
    (np.array([1, 0, 0]), -2, tr_glob, np.array([np.sqrt(3) / 2, 0.5, 0]), -3 - 0.5 * np.sqrt(3)),
    (np.array([0, 1, 0]), -2, None, np.array([0, 1, 0]), -2),
    (np.array([0, 1, 0]), -2, tr_glob, np.array([-0.5, np.sqrt(3) / 2, 0]), -1.5 - np.sqrt(3))
]