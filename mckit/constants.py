# -*- coding: utf-8 -*-

import numpy as np

# from mckit.geometry import MIN_VOLUME

__all__ = [
    "MIN_BOX_VOLUME",
    "RESOLUTION",
    "TIME_UNITS",
    "FLOAT_TOLERANCE",
    "MCNP_ENCODING",
    "DROP_OPTIONS",
]

MIN_BOX_VOLUME = 0.001

# Resolution of float number
RESOLUTION = np.finfo(float).resolution

FLOAT_TOLERANCE = 1.0e-12

# TODO dvp: should be enum and change 365 to 365.25
TIME_UNITS = {
    "SECS": 1.0,
    "MINS": 60.0,
    "HOURS": 3600.0,
    "DAYS": 3600.0 * 24,
    "YEARS": 3600.0 * 24 * 365,
}

MCNP_ENCODING = "cp1251"

DROP_OPTIONS = frozenset(
    ["original", "transform", "comment", "trailing_comment", "comment_above"]
)
