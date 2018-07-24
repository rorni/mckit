# -*- coding: utf-8 -*-

import numpy as np

from .geometry import MIN_VOLUME

__all__ = [
    'ORIGIN', 'EX', 'EY', 'EZ', 'GLOBAL_BOX', 'MIN_BOX_VOLUME',
    'IDENTITY_ROTATION',
    'ANGLE_TOLERANCE', 'RESOLUTION'
]

# identity rotation matrix
IDENTITY_ROTATION = np.eye(3)

MIN_BOX_VOLUME = MIN_VOLUME

# angle tolerance
ANGLE_TOLERANCE = 0.001

# Resolution of float number
RESOLUTION = np.finfo(float).resolution

# Natural presence of isotopes

# ------------------------------------------------------------------------------




# if __name__ == '__main__':
#    print(_NAME_TO_CHARGE)
#    print(_ISOTOPE_MASS)
#    print(_NATURAL_ABUNDANCE)

TIME_UNITS = {'SECS': 1., 'MINS': 60., 'HOURS': 3600., 'DAYS': 3600.*24, 'YEARS': 3600.*24*365}