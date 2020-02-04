import itertools
import math
from typing import Union, Iterable, Callable, Optional

import numpy as np
from numpy import ndarray

from mckit.constants import FLOAT_TOLERANCE

ComparableType = Optional[Union[Iterable, ndarray, float]]


def tolerance_estimator(
    rtol: float = FLOAT_TOLERANCE,
    atol: float = FLOAT_TOLERANCE,
    equal_nan: bool = False,
) -> Callable[[ComparableType, ComparableType], bool]:
    """
    Estimates "closeness of numpy arrays and float scalars using math.isclose and numpy.allclose methods
    """
    def _call(a: ComparableType, b: ComparableType) -> bool:
        if a is b:
            return True
        if a is None:
            return False  # b is not None here
        if b is None:
            return False
        if isinstance(a, float) and isinstance(b, float):
            return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
        elif isinstance(a, ndarray) and isinstance(b, ndarray):
            return np.allclose(a, b, rtol, atol, equal_nan)
        elif issubclass(type(a), Iterable) and issubclass(type(b), Iterable):
            for ai, bi in itertools.zip_longest(a, b):
                if not _call(ai, bi):
                    return False
            return True
        elif isinstance(a, int) and isinstance(b, int):
            return a == b
        else:
            raise TypeError(f"Not implemented for {type(a).__name__} and {type(b).__name__}")

    return _call