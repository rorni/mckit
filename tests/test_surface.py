# -*- coding: utf-8 -*-

import unittest

import numpy as np

from mckit.constants import *
from mckit.surface import Plane, GQuadratic, Torus, Sphere, Cylinder, \
    create_surface
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


class TestSphereSurface(unittest.TestCase):
    def test_sphere_creation(self):
        for i, (c, r, tr) in enumerate(sphere_creation_cases):
            with self.subTest(i=i):
                sph = Sphere(c, r, transform=tr)
                if tr is None:
                    c_ref = np.array(c)
                else:
                    c_ref = tr.apply2point(c)
                self.assertAlmostEqual(sph._radius, r)
                for j in range(3):
                    self.assertAlmostEqual(sph._center[j], c_ref[j])

    def test_point_test(self):
        sph = Sphere([1, 2, 3], 5)
        for i, (p, ans) in enumerate(sphere_point_test_cases):
            with self.subTest(i=i):
                sense = sph.test_point(p)
                if isinstance(sense, np.ndarray):
                    for s, a in zip(sense, ans):
                        self.assertEqual(s, a)
                else:
                    self.assertEqual(sense, ans)

    def test_region_test(self):
        for i, (c, r, ans) in enumerate(sphere_region_test_cases):
            with self.subTest(i=i):
                sph = Sphere(c, r)
                result = sph.test_region(region)
                self.assertEqual(result, ans)


class TestCylinderSurface(unittest.TestCase):
    def test_creation(self):
        for i, (pt, ax, r, tr) in enumerate(cylinder_creation_cases):
            with self.subTest(i=i):
                cyl = Cylinder(pt, ax, r, transform=tr)
                if tr is None:
                    pt_ref = np.array(pt)
                    ax_ref = np.array(ax) / np.linalg.norm(ax)
                else:
                    pt_ref = tr.apply2point(pt)
                    ax_ref = tr.apply2vector(ax) / np.linalg.norm(ax)
                self.assertAlmostEqual(cyl._radius, r)
                for j in range(3):
                    self.assertAlmostEqual(cyl._pt[j], pt_ref[j])
                    self.assertAlmostEqual(cyl._axis[j], ax_ref[j])

    def test_point_test(self):
        cyl = Cylinder([-1, 3, 4], [1, 0, 0], 2)
        for i, (p, ans) in enumerate(cylinder_point_test_cases):
            with self.subTest(i=i):
                sense = cyl.test_point(p)
                if isinstance(sense, np.ndarray):
                    for s, a in zip(sense, ans):
                        self.assertEqual(s, a)
                else:
                    self.assertEqual(sense, ans)

    def test_region_test(self):
        for i, (pt, ax, r, ans) in enumerate(cylinder_region_test_cases):
            with self.subTest(i=i):
                cyl = Cylinder(pt, ax, r)
                result = cyl.test_region(region)
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

    @unittest.expectedFailure
    def test_transform(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_region_test(self):
        raise NotImplementedError


class TestSurfaceCreation(unittest.TestCase):
    def test_plane_creation(self):
        for i, (kind, params, v, k) in enumerate(create_surface_plane_cases):
            with self.subTest(i=i):
                surf = create_surface(kind, *params)
                self.assertAlmostEqual(surf._k, k)
                for j in range(3):
                    self.assertAlmostEqual(surf._v[j], v[j])

    def test_sphere_creation(self):
        for i, (kind, params, c, r) in enumerate(create_surface_sphere_cases):
            with self.subTest(i=i):
                surf = create_surface(kind, *params)
                for j in range(3):
                    self.assertAlmostEqual(surf._center[j], c[j])
                self.assertAlmostEqual(surf._radius, r)

    def test_cylinder_creation(self):
        for i, (kind, params, pt, ax, r) in enumerate(create_surface_cylinder_cases):
            with self.subTest(i=i):
                surf = create_surface(kind, *params)
                for j in range(3):
                    self.assertAlmostEqual(surf._pt[j], pt[j])
                    self.assertAlmostEqual(surf._axis[j], ax[j])
                self.assertAlmostEqual(surf._radius, r)

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

sphere_creation_cases = [
    ([1, 2, 3], 5, None),
    ([-1, 2, -3], 5, tr_glob)
]

sphere_point_test_cases = [
    (np.array([1, 2, 3]), -1), (np.array([5.999, 2, 3]), -1),
    (np.array([6.001, 2, 3]), +1), (np.array([1, 6.999, 3]), -1),
    (np.array([1, 7.001, 3]), +1), (np.array([1, 2, 7.999]), -1),
    (np.array([1, 2, 8.001]), +1), (np.array([-3.999, 2, 3]), -1),
    (np.array([-4.001, 2, 3]), +1), (np.array([1, 2.999, 3]), -1),
    (np.array([1, -3.001, 3]), +1), (np.array([1, 2, -1.999]), -1),
    (np.array([1, 2, -2.001]), +1),
    (np.array([[1, 2, 3], [5.999, 2, 3], [6.001, 2, 3], [1, 6.999, 3],
               [1, 7.001, 3], [1, 2, 7.999], [1, 2, 8.001], [-3.999, 2, 3],
               [-4.001, 2, 3], [1, 2.999, 3], [1, -3.001, 3], [1, 2, -1.999],
               [1, 2, -2.001]]),
     np.array([-1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1]))
]

sphere_region_test_cases = [
    (np.array([0, 0, 0]), 1.8, -1), (np.array([0, 0, 0]), 1.7, 0),
    (np.array([0, 0, -2]), 0.999, +1), (np.array([0, 0, -2]), 1.001, 0),
    (np.array([-2, -2, -2]), 1.7, +1), (np.array([-2, -2, -2]), 1.8, 0),
    (np.array([-2, -2, -2]), 5.1, 0),  (np.array([-2, -2, -2]), 5.2, -1),
    (np.array([-2, 0, -2]), 1.4, +1), (np.array([-2, 0, -2]), 1.5, 0),
    (np.array([-2, 0, -2]), 4.3, 0), (np.array([-2, 0, -2]), 4.4, -1)
]

cylinder_creation_cases = [
    ([1, 2, 3], [1, 2, 3], 5, None),
    ([-1, 2, -3], [1, 2, 3], 5, tr_glob)
]

cylinder_point_test_cases = [
    ([0, 3, 4], -1), ([-2, 3, 2.001], -1), ([-3, 3, 1.999], +1),
    ([2, 3, 5.999], -1), ([3, 3, 6.001], +1), ([4, 1.001, 4], -1),
    ([-4, 0.999, 4], +1), ([-5, 4.999, 4], -1), ([5, 5.001, 4], +1),
    (np.array([[0, 3, 4], [-2, 3, 2.001], [-3, 3, 1.999], [2, 3, 5.999],
               [3, 3, 6.001], [4, 1.001, 4], [-4, 0.999, 4], [-5, 4.999, 4],
               [5, 5.001, 4]]), [-1, -1, +1, -1, +1, -1, +1, -1, +1])
]

cylinder_region_test_cases = [
    ([0, 0, 0], [1, 0, 0], 0.5, 0), ([0, 0, 0], [1, 0, 0], 1.4, 0),
    ([0, 0, 0], [1, 0, 0], 1.5, -1), ([0, 1, 1], [1, 0, 0], 0.001, 0),
    ([0, 1, 1], [1, 0, 0], 2.8, 0), ([0, 1, 1], [1, 0, 0], 2.9, -1),
    ([0, 2, 0], [1, 0, 0], 0.999, +1), ([0, 2, 0], [1, 0, 0], 1.001, 0),
    ([0, 2, 0], [1, 0, 0], 3.1, 0), ([0, 2, 0], [1, 0, 0], 3.2, -1),
    ([0, 0, 0], [0, 1, 0], 0.5, 0), ([0, 0, 0], [0, 1, 0], 1.4, 0),
    ([0, 0, 0], [0, 1, 0], 1.5, -1), ([1, 0, 1], [0, 1, 0], 0.001, 0),
    ([1, 0, 1], [0, 1, 0], 2.8, 0), ([1, 0, 1], [0, 1, 0], 2.9, -1),
    ([2, 0, 0], [0, 1, 0], 0.999, +1), ([2, 0, 0], [0, 1, 0], 1.001, 0),
    ([2, 0, 0], [0, 1, 0], 3.1, 0), ([2, 0, 0], [0, 1, 0], 3.2, -1)
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

create_surface_sphere_cases = [
    ('SO', [6.1], [0, 0, 0], 6.1),
    ('SX', [-3.4, 6.2], [-3.4, 0, 0], 6.2),
    ('SY', [3.5, 6.3], [0, 3.5, 0], 6.3),
    ('SZ', [-3.6, 6.4], [0, 0, -3.6], 6.4),
    ('S', [3.7, -3.8, 3.9, 6.5], np.array([3.7, -3.8, 3.9]), 6.5)
]

create_surface_cylinder_cases = [
    ('CX', [6.6], [0, 0, 0], [1, 0, 0], 6.6),
    ('CY', [6.7], [0, 0, 0], [0, 1, 0], 6.7),
    ('CZ', [6.8], [0, 0, 0], [0, 0, 1], 6.8),
    ('C/X', [4.0, -4.1, 6.9], [0, 4.0, -4.1], [1, 0, 0], 6.9),
    ('C/Y', [-4.2, 4.3, 7.0], [-4.2, 0, 4.3], [0, 1, 0], 7.0),
    ('C/Z', [4.4, 4.5, 7.1], [4.4, 4.5, 0], [0, 0, 1], 7.1)
]

create_surface_gq_cases = [
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


if __name__ == '__main__':
    unittest.main()
