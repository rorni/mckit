# -*- coding: utf-8 -*-

import unittest

import numpy as np

from mckit.transformation import Transformation
from mckit.constants import *

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