# -*- coding: utf-8 -*-

import unittest

import numpy as np

from mckit.constants import *
from mckit.surface import Plane, GQuadratic, Torus, create_surface
from mckit.transformation import Transformation


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


class TestSurfaceCreation(unittest.TestCase):
    def test_plane_creation(self):
        for i, (kind, params, v, k) in enumerate(create_surface_plane_cases):
            with self.subTest(i=i):
                surf = create_surface(kind, *params)
                self.assertAlmostEqual(surf._k, k)
                for j in range(3):
                    self.assertAlmostEqual(surf._v[j], v[j])

    def test_gq_creation(self):
        for i, (kind, params, m, v, k) in enumerate(create_surface_gq_cases):
            with self.subTest(i=i):
                surf = create_surface(kind, *params)
                self.assertAlmostEqual(surf._k, k)
                for j in range(3):
                    self.assertAlmostEqual(surf._v[j], v[j])
                    for l in range(3):
                        self.assertAlmostEqual(surf._m[j, l], m[j][l])

    def test_torus_creation(self):
        for i, (kind, params, r0, ax, R, a, b) in enumerate(
                create_surface_torus_cases):
            with self.subTest(i=i):
                surf = create_surface(kind, *params)
                self.assertAlmostEqual(surf._R, R)
                self.assertAlmostEqual(surf._a, a)
                self.assertAlmostEqual(surf._b, b)
                for j in range(3):
                    self.assertAlmostEqual(surf._center[j], r0[j])
                    self.assertAlmostEqual(surf._axis[j], ax[j])


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

create_surface_plane_cases = [
    ('PX', [5.3], np.array([1, 0, 0]), -5.3),
    ('PY', [5.4], np.array([0, 1, 0]), -5.4),
    ('PZ', [5.5], np.array([0, 0, 1]), -5.5),
    ('P', [3.2, -1.4, 5.7, -4.8], np.array([3.2, -1.4, 5.7]), 4.8),
    ('X', [5.6, 6.7], np.array([1, 0, 0]), -5.6),
    ('Y', [5.7, 6.8], np.array([0, 1, 0]), -5.7),
    ('Z', [5.8, -6.9], np.array([0, 0, 1]), -5.8),
    ('X', [5.6, 6.7, 5.6, -7.9], np.array([1, 0, 0]), -5.6),
    ('Y', [5.7, 6.8, 5.7, 6.2], np.array([0, 1, 0]), -5.7),
    ('Z', [5.8, -6.9, 5.8, -9.9], np.array([0, 0, 1]), -5.8)
]

create_surface_gq_cases = [
    ('SO', [6.1], np.eye(3), [0, 0, 0], -6.1**2),
    ('SX', [-3.4, 6.2], np.eye(3), [2 * 3.4, 0, 0], 3.4**2 - 6.2**2),
    ('SY', [3.5, 6.3], np.eye(3), [0, -2 * 3.5, 0], 3.5**2 - 6.3**2),
    ('SZ', [-3.6, 6.4], np.eye(3), [0, 0, 2 * 3.6], 3.6**2 - 6.4**2),
    ('S', [3.7, -3.8, 3.9, 6.5], np.eye(3), -2 * np.array([3.7, -3.8, 3.9]),
     3.7**2 + 3.8**2 + 3.9**2 - 6.5**2),
    ('CX', [6.6], [[0, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0], -6.6**2),
    ('CY', [6.7], [[1, 0, 0], [0, 0, 0], [0, 0, 1]], [0, 0, 0], -6.7**2),
    ('CZ', [6.8], [[1, 0, 0], [0, 1, 0], [0, 0, 0]], [0, 0, 0], -6.8**2),
    ('C/X', [4.0, -4.1, 6.9], [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
     -2 * np.array([0, 4.0, -4.1]), 4.0**2 + 4.1**2 - 6.9**2),
    ('C/Y', [-4.2, 4.3, 7.0], [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
     -2 * np.array([-4.2, 0, 4.3]), 4.2**2 + 4.3**2 - 7.0**2),
    ('C/Z', [4.4, 4.5, 7.1], [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
     -2 * np.array([4.4, 4.5, 0]), 4.4 ** 2 + 4.5 ** 2 - 7.1 ** 2),
    ('KX', [4.6, 0.33], [[-0.33, 0, 0], [0, 1, 0], [0, 0, 1]],
     -2 * np.array([-0.33 * 4.6, 0, 0]), -0.33 * 4.6**2),
    ('KY', [4.7, 0.33], [[1, 0, 0], [0, -0.33, 0], [0, 0, 1]],
     -2 * np.array([0, -0.33 * 4.7, 0]), -0.33 * 4.7**2),
    ('KZ', [-4.8, 0.33], [[1, 0, 0], [0, 1, 0], [0, 0, -0.33]],
     -2 * np.array([0, 0, 4.8 * 0.33]), -0.33 * 4.8**2),
    ('K/X', [4.9, -5.0, 5.1, 0.33], [[-0.33, 0, 0], [0, 1, 0], [0, 0, 1]],
     -2 * np.array([-0.33 * 4.9, -5.0, 5.1]), 5.0**2 + 5.1**2 - 0.33 * 4.9**2),
    ('K/Y', [-5.0, -5.1, 5.2, 0.33], [[1, 0, 0], [0, -0.33, 0], [0, 0, 1]],
     -2 * np.array([-5.0, 0.33 * 5.1, 5.2]), 5.0**2 + 5.2**2 - 0.33 * 5.1**2),
    ('K/Z', [5.3, 5.4, 5.5, 0.33], [[1, 0, 0], [0, 1, 0], [0, 0, -0.33]],
     -2 * np.array([5.3, 5.4, -5.5 * 0.33]), 5.3**2 + 5.4**2 - 0.33 * 5.5**2),
    ('SQ', [0.5, -2.5, 3.0, 1.1, -1.3, -5.4, -7.0, 3.2, -1.7, 8.4],
     np.diag([0.5, -2.5, 3.0]), 2 * np.array([1.1 - 0.5 * 3.2, -1.3 - 2.5 * 1.7,
     -5.4 - 3.0 * 8.4]), 0.5 * 3.2**2 - 2.5 * 1.7**2 + 3.0 * 8.4**2 - 7.0 - 2 *
     (1.1 * 3.2 + 1.3 * 1.7 - 5.4 * 8.4)),
    ('GQ', [1, 2, 3, 4, 5, 6, 7, 8, 9, -10],
     [[1, 2, 3], [2, 2, 2.5], [3, 2.5, 3]], [7, 8, 9], -10),
    ('X', [1, 3.2, -2, 3.2], np.diag([0, 1, 1]), [0, 0, 0], -3.2**2),
    ('Y', [-1, 3.3, -2, 3.3], np.diag([1, 0, 1]), [0, 0, 0], -3.3**2),
    ('Z', [1, 3.4, 2, 3.4], np.diag([1, 1, 0]), [0, 0, 0], -3.4**2),
    ('X', [1, 3, 2, 2], np.diag([-1, 1, 1]), [8, 0, 0], -16),
    ('Y', [1, 1, 2, 2], np.diag([1, -1, 1]), [0, 0, 0], 0),
    ('Z', [0, 1, 1, 2], np.diag([1, 1, -1]), [0, 0, -2], -1)
]

create_surface_torus_cases = [
    ('TX', [1, 2, -3, 5, 0.5, 0.8], [1, 2, -3], EX, 5, 0.5, 0.8),
    ('TY', [-4, 5, -6, 3, 0.9, 0.2], [-4, 5, -6], EY, 3, 0.9, 0.2),
    ('TZ', [0, -3, 5, 1, 0.1, 0.2], [0, -3, 5], EZ, 1, 0.1, 0.2)
]