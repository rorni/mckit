# -*- coding: utf-8 -*-

import numpy as np

from .geometry import MIN_VOLUME

__all__ = [
    'MIN_BOX_VOLUME',
    'RESOLUTION', 'TIME_UNITS'
]


MIN_BOX_VOLUME = MIN_VOLUME

# Resolution of float number
RESOLUTION = np.finfo(float).resolution


_TOLERANCE = 1.e-3


TIME_UNITS = {'SECS': 1., 'MINS': 60., 'HOURS': 3600., 'DAYS': 3600.*24, 'YEARS': 3600.*24*365}