# -*- coding: utf-8 -*-

import unittest

import numpy as np

from mckit.surface import Plane, GQuadratic, Torus
from mckit.transformation import Transformation
from mckit.constants import *


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

    def test_point_test(self):
        plane = Plane(EX, 0, tr_glob)
        for i, (p, ans) in enumerate(plane_point_test_cases):
            with self.subTest(i=i):
                p1 = tr_glob.apply2point(p)
                sense = plane.test_point(p1)
                if isinstance(sense, np.ndarray):
                    for s, a in zip(sense, ans):
                        self.assertEqual(s, a)
                else:
                    self.assertEqual(sense, ans)

    def test_region_test(self):
        for i, (v, k, ans) in enumerate(plane_region_test_cases):
            with self.subTest(i=i):
                plane = Plane(v, k)
                result = plane.test_region(region)
                self.assertEqual(result, ans)


class TestGQuadraticSurface(unittest.TestCase):
    def test_gq_creation(self):
        for i, (m, v, k, tr) in enumerate(gq_creation_cases):
            with self.subTest(i=i):
                p = GQuadratic(m, v, k, transform=tr)
                if tr is None:
                    m_ref, v_ref, k_ref = np.array(m), np.array(v), k
                else:
                    m_ref, v_ref, k_ref = tr.apply2gq(m, v, k)
                self.assertAlmostEqual(p._k, k_ref)
                for j in range(3):
                    self.assertAlmostEqual(p._v[j], v_ref[j])
                    for l in range(3):
                        self.assertAlmostEqual(p._m[j, l], m_ref[j, l])

    def test_gq_point_test(self):
        gq = GQuadratic(np.diag([1, 1, 1]), [0, 0, 0], -1, tr_glob)
        for i, (p, ans) in enumerate(gq_point_test_cases):
            with self.subTest(i=i):
                p1 = tr_glob.apply2point(p)
                sense = gq.test_point(p1)
                if isinstance(sense, np.ndarray):
                    for s, a in zip(sense, ans):
                        self.assertEqual(s, a)
                else:
                    self.assertEqual(sense, ans)

    def test_region_test(self):
        for i, (m, v, k, ans) in enumerate(gq_region_test_cases):
            with self.subTest(i=i):
                gq = GQuadratic(m, v, k)
                result = gq.test_region(region)
                self.assertEqual(result, ans)


class TestTorusSurface(unittest.TestCase):
    def test_torus_creation(self):
        for i, (center, axis, R, a, b, tr) in enumerate(torus_creation_cases):
            with self.subTest(i=i):
                p = Torus(center, axis, R, a, b, transform=tr)
                if tr is None:
                    c_ref = center
                    a_ref = axis
                else:
                    c_ref = tr.apply2point(center)
                    a_ref = tr.apply2vector(axis)
                self.assertAlmostEqual(p._R, R)
                self.assertAlmostEqual(p._a, a)
                self.assertAlmostEqual(p._b, b)
                for j in range(3):
                    self.assertAlmostEqual(p._center[j], c_ref[j])
                    self.assertAlmostEqual(p._axis[j], a_ref[j])

    def test_torus_point_test(self):
        tor = Torus([0, 0, 0], EX, 4, 2, 1, tr_glob)
        for i, (p, ans) in enumerate(torus_point_test_cases):
            with self.subTest(i=i):
                p1 = tr_glob.apply2point(p)
                sense = tor.test_point(p1)
                if isinstance(sense, np.ndarray):
                    for s, a in zip(sense, ans):
                        self.assertEqual(s, a)
                else:
                    self.assertEqual(sense, ans)

    def test_region_test(self):
        pass


plane_creation_cases = [
    (EZ, -2, None, np.array([0, 0, 1]), -2),
    (EZ, -2, tr_glob, np.array([0, 0, 1]), 1),
    (EX, -2, None, np.array([1, 0, 0]), -2),
    (EX, -2, tr_glob, np.array([np.sqrt(3) / 2, 0.5, 0]), -3 - 0.5 * np.sqrt(3)),
    (EY, -2, None, np.array([0, 1, 0]), -2),
    (EY, -2, tr_glob, np.array([-0.5, np.sqrt(3) / 2, 0]), -1.5 - np.sqrt(3))
]

plane_point_test_cases = [
    (np.array([1, 0, 0]), +1), (np.array([-1, 0, 0]), -1),
    (np.array([1, 0, 0]), +1), (np.array([-1, 0, 0]), -1),
    (np.array([0.1, 0, 0]), +1), (np.array([-0.1, 0, 0]), -1),
    (np.array([1.e-6, 100, -300]), +1), (np.array([-1.e-6, 200, -500]), -1),
    (np.array([[1, 0, 0], [-1, 0, 0], [0.1, 0, 0], [-0.1, 0, 0], [1.e-6, 100, -300],
               [-1.e-6, 200, -500]]), np.array([1, -1, 1, -1, 1, -1]))
]

region = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                   [1, -1, -1],  [1, -1, 1],  [1, 1, -1],  [1, 1, 1]])
plane_region_test_cases = [
    ([1, 0, 0], 0, 0),
    ([1, 0, 0], 1.001, 1),
    ([1, 0, 0], -1.001, -1),
    ([1, 0, 0], 0.999, 0),
    ([1, 0, 0], -0.999, 0),
    ([0, 1, 0], 0, 0),
    ([0, 1, 0], 1.001, 1),
    ([0, 1, 0], -1.001, -1),
    ([0, 1, 0], 0.999, 0),
    ([0, 1, 0], -0.999, 0),
    ([0, 0, 1], 0, 0),
    ([0, 0, 1], 1.001, 1),
    ([0, 0, 1], -1.001, -1),
    ([0, 0, 1], 0.999, 0),
    ([0, 0, 1], -0.999, 0),
    ([1, 1, 1], -2.999, 0),
    ([1, 1, 1], -3.001, -1),
    ([1, 1, 1], 2.999, 0),
    ([1, 1, 1], 3.001, 1)
]

gq_creation_cases = [
    ([[1, 0, 0], [0, 2, 0], [0, 0, 3]], [1, 2, 3], -4, None),
    ([[1, 0, 0], [0, 2, 0], [0, 0, 3]], [1, 2, 3], -4, tr_glob)
]

gq_point_test_cases = [
    (np.array([0, 0, 0]), -1), (np.array([-0.999, 0, 0]), -1),
    (np.array([0.999, 0, 0]), -1), (np.array([-1.001, 0, 0]), +1),
    (np.array([1.001, 0, 0]), +1), (np.array([0, 0.999, 0]), -1),
    (np.array([0, 1.001, 0]), +1), (np.array([0, 0, -0.999]), -1),
    (np.array(
        [[0, 0, 0], [-0.999, 0, 0], [0.999, 0, 0], [-1.001, 0, 0], [1.001, 0, 0],
         [0, 0.999, 0], [0, 1.001, 0], [0, 0, -0.999]]),
     np.array([-1, -1, -1, 1, 1, -1, 1, -1]))
]

gq_region_test_cases = [
    # spheres
    (np.diag([1, 1, 1]), [0, 0, 0], -1, 0),
    (np.diag([1, 1, 1]), [0, 0, 0], -0.1, 0),
    (np.diag([1, 1, 1]), [0, 0, 0], -3.01, -1),
    (np.diag([1, 1, 1]), -2 * np.array([1, 1, 1]), 3 - 0.1, 0),
    (np.diag([1, 1, 1]), -2 * np.array([2, 2, 2]), 12 - 3.01, 0),
    (np.diag([1, 1, 1]), -2 * np.array([2, 2, 2]), 12 - 2.99, +1),
    (np.diag([1, 1, 1]), -2 * np.array([2, 0, 0]), 4 - 1.01, 0),
    (np.diag([1, 1, 1]), -2 * np.array([2, 0, 0]), 4 - 0.99, +1),
    (np.diag([1, 1, 1]), -2 * np.array([100, 0, 100]), 20000 - 2, +1)
]

torus_creation_cases = [
    ([1, 2, 3], [0, 0, 1], 4, 2, 1, None),
    ([1, 2, 3], [0, 0, 1], 4, 2, 1, tr_glob)
]

torus_point_test_cases = [
    ([0, 0, 0], 1), ([0, 0, 2.99], 1), ([0, 0, 5.01], 1), ([0, 0, 3.01], -1),
    ([0, 0, 4.99], -1), ([0, 2.99, 0], 1), ([0, 5.01, 0], 1),
    ([0, 3.01, 0], -1), ([0, 4.99, 0], -1), ([2.01, 0, 4], 1), ([1.99, 0, 4], -1),
    ([2.01, 4, 0], 1), ([1.99, 4, 0], -1),
    ([[0, 0, 0], [0, 0, 2.99], [0, 0, 5.01], [0, 0, 3.01], [0, 0, 4.99], [0, 2.99, 0],
      [0, 5.01, 0], [0, 3.01, 0], [0, 4.99, 0], [2.01, 0, 4], [1.99, 0, 4],
      [2.01, 4, 0], [1.99, 4, 0]], [1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1])
]
