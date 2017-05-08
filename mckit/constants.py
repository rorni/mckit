# -*- coding: utf-8 -*-

import numpy as np


__all__ = [
    'EX', 'EY', 'EZ',
    'ORIGIN', 'IDENTITY_ROTATION',
    'ANGLE_TOLERANCE', 'RESOLUTION'
]


# basis vectors
EX = np.array([1, 0, 0])
EY = np.array([0, 1, 0])
EZ = np.array([0, 0, 1])

# the origin of coordinate system - vector of zeros.
ORIGIN = np.zeros((3,))

# identity rotation matrix
IDENTITY_ROTATION = np.eye(3)

# angle tolerance
ANGLE_TOLERANCE = 0.001

RESOLUTION = np.finfo(float).resolution
