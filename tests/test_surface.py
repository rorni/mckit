# -*- coding: utf-8 -*-

import unittest

import numpy as np

from mckit.constants import *
from mckit.surface import Plane, GQuadratic, Torus, Sphere, Cylinder, Cone, \
    create_surface
from mckit.transformation import Transformation
from tests.surface_test_data import surface_test_point_data
from tests.surface_test_data import surface_test_region_data


tr_glob = Transformation(translation=[1, 2, -3], indegrees=True,
                         rotation=[30, 60, 90, 120, 30, 90, 90, 90, 0])

class_apply = {'Plane': Plane, 'Sphere': Sphere, 'Cylinder': Cylinder,
               'Cone': Cone, 'Torus': Torus, 'GQuadratic': GQuadratic}


class TestSurfaceMethods(unittest.TestCase):
    def test_point_test(self):
        for class_name, test_cases in surface_test_point_data.data.items():
            TClass = class_apply[class_name]
            for surf_params, test_points in test_cases:
                surf = TClass(*surf_params)
                for i, (p, ans) in enumerate(test_points):
                    msg = class_name + ' params: {0}, case {1}'.format(surf_params, i)
                    with self.subTest(msg=msg):
                        sense = surf.test_point(p)
                        if isinstance(sense, np.ndarray):
                            for s, a in zip(sense, ans):
                                self.assertEqual(s, a)
                        else:
                            self.assertEqual(sense, ans)

    def test_region_test(self):
        for class_name, test_cases in surface_test_region_data.data.items():
            TClass = class_apply[class_name]
            for i, (*params, ans) in enumerate(test_cases):
                msg = class_name + ' case {0}'.format(i)
                with self.subTest(msg=msg):
                    surf = TClass(*params)
                    result = surf.test_region(surface_test_region_data.region)
                    self.assertEqual(result, ans)

class TestPlaneSurface(unittest.TestCase):
    def test_plane_creation(self):
        for i, (v, k, tr, v_ref, k_ref) in enumerate(plane_creation_cases):
            with self.subTest(i=i):
                p = Plane(v, k, transform=tr)
                self.assertAlmostEqual(p._k, k_ref)
                for j in range(3):
                    self.assertAlmostEqual(p._v[j], v_ref[j])






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






class TestConeSurface(unittest.TestCase):
    def test_creation(self):
        for i, (ap, ax, ta, tr) in enumerate(cone_creation_cases):
            with self.subTest(i=i):
                cone = Cone(ap, ax, ta, transform=tr)
                if tr is None:
                    ap_ref = np.array(ap)
                    ax_ref = np.array(ax) / np.linalg.norm(ax)
                else:
                    ap_ref = tr.apply2point(ap)
                    ax_ref = tr.apply2vector(ax) / np.linalg.norm(ax)
                self.assertAlmostEqual(cone._t2, ta**2)
                for j in range(3):
                    self.assertAlmostEqual(cone._apex[j], ap_ref[j])
                    self.assertAlmostEqual(cone._axis[j], ax_ref[j])






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

    def test_cone_creation(self):
        for i, (kind, params, ap, ax, t2) in enumerate(create_surface_cone_cases):
            with self.subTest(i=i):
                surf = create_surface(kind, *params)
                for j in range(3):
                    self.assertAlmostEqual(surf._apex[j], ap[j])
                    self.assertAlmostEqual(surf._axis[j], ax[j])
                self.assertAlmostEqual(surf._t2, t2)

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



sphere_creation_cases = [
    ([1, 2, 3], 5, None),
    ([-1, 2, -3], 5, tr_glob)
]



cylinder_creation_cases = [
    ([1, 2, 3], [1, 2, 3], 5, None),
    ([-1, 2, -3], [1, 2, 3], 5, tr_glob)
]



cone_creation_cases = [
    ([1, 2, 3], [1, 2, 3], 0.5, None),
    ([-1, 2, -3], [1, 2, 3], 0.5, tr_glob)
]



gq_creation_cases = [
    ([[1, 0, 0], [0, 2, 0], [0, 0, 3]], [1, 2, 3], -4, None),
    ([[1, 0, 0], [0, 2, 0], [0, 0, 3]], [1, 2, 3], -4, tr_glob)
]


torus_creation_cases = [
    ([1, 2, 3], [0, 0, 1], 4, 2, 1, None),
    ([1, 2, 3], [0, 0, 1], 4, 2, 1, tr_glob)
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

create_surface_cone_cases = [
    ('KX', [4.6, 0.33], [4.6, 0, 0], [1, 0, 0], 0.33),
    ('KY', [4.7, 0.33], [0, 4.7, 0], [0, 1, 0], 0.33),
    ('KZ', [-4.8, 0.33], [0, 0, -4.8], [0, 0, 1], 0.33),
    ('K/X', [4.9, -5.0, 5.1, 0.33], [4.9, -5.0, 5.1], [1, 0, 0], 0.33),
    ('K/Y', [-5.0, -5.1, 5.2, 0.33], [-5.0, -5.1, 5.2], [0, 1, 0], 0.33),
    ('K/Z', [5.3, 5.4, 5.5, 0.33], [5.3, 5.4, 5.5], [0, 0, 1], 0.33)
]

create_surface_gq_cases = [
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
