# -*- coding: utf-8 -*-

import unittest

import numpy as np

from mckit.constants import *
from mckit.transformation import Transformation

places = 10


class TestTransformationCreation(unittest.TestCase):
    """Test creation of transformation instance."""

    def test_normal_creation(self):
        for i, test_item in enumerate(normal_creation_cases):
            with self.subTest(i=i):
                param_dict, u, t = test_item
                transform = Transformation(**param_dict)
                for j in range(3):
                    self.assertAlmostEqual(t[j], transform._t[j], places=places)
                    for k in range(3):
                        self.assertAlmostEqual(u[j, k], transform._u[j, k],
                                               places=places)

    def test_exception_creation(self):
        for i, test_item in enumerate(exception_creation_cases):
            with self.subTest(i=i):
                self.assertRaises(ValueError, Transformation, **test_item)

    def test_orthogonalization(self):
        for i, test_item in enumerate(orthogonalization_creation_cases):
            with self.subTest(i=i):
                tr = Transformation(**test_item)
                result = np.dot(tr._u.transpose(), tr._u)
                for j in range(3):
                    for k in range(3):
                        self.assertAlmostEqual(result[j, k],
                                               IDENTITY_ROTATION[j, k], places)


class TestTransformationMethods(unittest.TestCase):
    def test_point_transformation(self):
        for i, (p1, p) in enumerate(point_transformation_cases):
            with self.subTest(i=i):
                result = tr_glob.apply2point(p1)
                for j in range(p.shape[0]):
                    if len(p.shape) > 1:
                        for k in range(p.shape[1]):
                            self.assertAlmostEqual(result[j, k], p[j, k])
                    else:
                        self.assertAlmostEqual(result[j], p[j])

    def test_vector_transformation(self):
        for i, (v1, v) in enumerate(vector_transformation_cases):
            with self.subTest(i=i):
                result = tr_glob.apply2vector(v1)
                for j in range(v.shape[0]):
                    if len(v.shape) > 1:
                        for k in range(v.shape[1]):
                            self.assertAlmostEqual(result[j, k], v[j, k])
                    else:
                        self.assertAlmostEqual(result[j], v[j])

    def test_plane_transformation(self):
        for i, (v1, p1) in enumerate(plane_transformation_cases):
            with self.subTest(i=i):
                k1 = -np.dot(v1, p1)
                v, k = tr_glob.apply2plane(v1, k1)
                v_ref = tr_glob.apply2vector(v1)
                k_ref = -np.dot(v_ref, tr_glob.apply2point(p1))

                self.assertAlmostEqual(k, k_ref)
                for j in range(3):
                    self.assertAlmostEqual(v[j], v_ref[j])

    def test_gq_transformation(self):
        # Sphere transformation example
        p0 = np.array([3, -4, 7])
        R = 4
        m1 = np.eye(3)
        v1 = -2 * p0
        k1 = np.linalg.norm(p0)**2 - R**2
        m, v, k = tr_glob.apply2gq(m1, v1, k1)
        v_ref = -2 * tr_glob.apply2point(p0)
        k_ref = np.linalg.norm(-0.5 * v_ref)**2 - R**2
        self.assertAlmostEqual(k, k_ref)
        for i in range(3):
            self.assertAlmostEqual(v_ref[i], v[i])
            for j in range(3):
                self.assertAlmostEqual(m1[i, j], m[i, j])

    def test_reverse(self):
        tr_inv = tr_glob.reverse()
        for i, (p1, p) in enumerate(point_transformation_cases):
            with self.subTest(i=i):
                result = tr_inv.apply2point(p)
                for j in range(p.shape[0]):
                    if len(p.shape) > 1:
                        for k in range(p.shape[1]):
                            self.assertAlmostEqual(result[j, k], p1[j, k])
                    else:
                        self.assertAlmostEqual(result[j], p1[j])


normal_creation_cases = [
    ({}, IDENTITY_ROTATION, ORIGIN),
    ({'indegrees': True}, IDENTITY_ROTATION, ORIGIN),
    ({'indegrees': False}, IDENTITY_ROTATION, ORIGIN),
    ({'inverted': True}, IDENTITY_ROTATION, ORIGIN),
    ({'inverted': False}, IDENTITY_ROTATION, ORIGIN),
    ({'translation': [1, 0, 0]}, IDENTITY_ROTATION, np.array([1, 0, 0])),
    ({'translation': [1, 0, 0], 'inverted': True}, IDENTITY_ROTATION, np.array([-1, 0, 0])),
    ({'translation': [1, 2, 3], 'inverted': True}, IDENTITY_ROTATION, np.array([-1, -2, -3])),
    ({'rotation': np.cos(np.array([30, 60, 90, 120, 30, 90, 90, 90, 0]) * np.pi / 180)},
     np.cos(np.array([[30, 120, 90], [60, 30, 90], [90, 90, 0]]) * np.pi / 180), ORIGIN),
    ({'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0], 'indegrees': True},
     np.cos(np.array([[30, 120, 90], [60, 30, 90], [90, 90, 0]]) * np.pi / 180), ORIGIN),
    ({'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0], 'indegrees': True, 'translation': [1, 2, 3]},
     np.cos(np.array([[30, 120, 90], [60, 30, 90], [90, 90, 0]]) * np.pi / 180), np.array([1, 2, 3])),
    ({'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0], 'indegrees': True,
      'translation': [1, 2, 3], 'inverted': True},
     np.cos(np.array([[30, 120, 90], [60, 30, 90], [90, 90, 0]]) * np.pi / 180),
     np.array([-(np.sqrt(3) - 2) / 2, -(2 * np.sqrt(3) + 1) / 2, -3])),
]

exception_creation_cases = [
    {'translation': [0]},
    {'translation': [0, 1]},
    {'translation': [0, 1, 2, 3]},
    {'translation': [0, 1, 2, 3, 4]},
    {'rotation': [0, 1, 2]},
    {'rotation': [1, 2, 3, 4, 5, 6, 7, 8]},
    {'rotation': [30.058, 59.942, 90, 120, 30, 90, 90, 90, 0], 'indegrees': True},
    {'rotation': [29.942, 60.058, 90, 120, 30, 90, 90, 90, 0], 'indegrees': True},
    {'rotation': [30, 60, 90, 120.058, 30.058, 90, 90, 90, 0], 'indegrees': True},
    {'rotation': [30, 60, 90, 119.942, 29.942, 90, 90, 90, 0], 'indegrees': True},
    {'rotation': [30, 60, 90, 120, 30, 90, 90.058, 90.058, 0.058], 'indegrees': True},
    {'rotation': [30, 60, 90, 120, 30, 90, 89.942, 89.942, 0.058], 'indegrees': True},
]

orthogonalization_creation_cases = [
    {'rotation': [30.057, 59.943, 90, 120, 30, 90, 90, 90, 0], 'indegrees': True},
    {'rotation': [29.943, 60.057, 90, 120, 30, 90, 90, 90, 0], 'indegrees': True},
    {'rotation': [30, 60, 90, 120.057, 30.057, 90, 90, 90, 0], 'indegrees': True},
    {'rotation': [30, 60, 90, 119.943, 29.943, 90, 90, 90, 0], 'indegrees': True},
    {'rotation': [30, 60, 90, 120, 30, 90, 90.057, 90, 0.057], 'indegrees': True},
    {'rotation': [30, 60, 90, 120, 30, 90, 89.943, 90, 0.057], 'indegrees': True},
]

tr_glob = Transformation(translation=[1, 2, -3], indegrees=True,
                         rotation=[30, 60, 90, 120, 30, 90, 90, 90, 0])

point_transformation_cases = [
    (np.array([1, 0, 0]), np.array([np.sqrt(3) / 2 + 1, 2.5, -3])),
    (np.array([2, 0, 0]), np.array([np.sqrt(3) + 1, 3, -3])),
    (np.array([0, 1, 0]), np.array([0.5, np.sqrt(3) / 2 + 2, -3])),
    (np.array([0, 2, 0]), np.array([0, np.sqrt(3) + 2, -3])),
    (np.array([0, 0, 1]), np.array([1, 2, -2])),
    (np.array([[1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0], [0, 0, 1]]),
     np.array([[np.sqrt(3) / 2 + 1, 2.5, -3], [np.sqrt(3) + 1, 3, -3],
        [0.5, np.sqrt(3) / 2 + 2, -3], [0, np.sqrt(3) + 2, -3], [1, 2, -2]]))
]

vector_transformation_cases = [
    (np.array([1, 0, 0]), np.array([np.sqrt(3) / 2, 0.5, 0])),
    (np.array([2, 0, 0]), np.array([np.sqrt(3), 1, 0])),
    (np.array([0, 1, 0]), np.array([-0.5, np.sqrt(3) / 2, 0])),
    (np.array([0, 2, 0]), np.array([-1, np.sqrt(3), 0])),
    (np.array([0, 0, 1]), np.array([0, 0, 1])),
    (np.array([[1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0], [0, 0, 1]]),
     np.array([[np.sqrt(3) / 2, 0.5, 0], [np.sqrt(3), 1, 0],
               [-0.5, np.sqrt(3) / 2, 0], [-1, np.sqrt(3), 0], [0, 0, 1]]))
]

plane_transformation_cases = [
    (np.array([1, 0, 0]), np.array([2, 3, 5])),
    (np.array([0, 1, 0]), np.array([2, 3, 5])),
    (np.array([0, 0, 1]), np.array([2, 3, 5])),
    (np.array([1, -2, 3]), np.array([2, 3, 5])),
    (np.array([-4, 0, 5]), np.array([2, 3, 5]))
]

