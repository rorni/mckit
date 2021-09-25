from typing import Any, Callable, Iterable, Optional, Union

import itertools
import math

from abc import ABC, abstractmethod

import numpy as np

from mckit.constants import FLOAT_TOLERANCE
from numpy import ndarray


class MaybeClose(ABC):
    @abstractmethod
    def is_close_to(self, other: Any, estimator: "EstimatorType") -> bool:
        """Objects can be estimated as close with some estimator"""


ComparableType = Optional[Union[Iterable, ndarray, float, MaybeClose]]
EstimatorType = Callable[[ComparableType, ComparableType], bool]


def tolerance_estimator(
    rtol: float = FLOAT_TOLERANCE,
    atol: float = FLOAT_TOLERANCE,
    equal_nan: bool = False,
) -> EstimatorType:
    """
    Estimates "closeness of numpy arrays and float scalars using math.isclose and numpy.allclose methods
    """

    def _call(a: ComparableType, b: ComparableType) -> bool:
        if a is b:
            return True
        if a is None:
            return False  # b is not None here
        if b is None:
            return True  # in our use cases absent optional objects (like Transformations) are equal
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
        elif isinstance(a, MaybeClose) and isinstance(b, MaybeClose):
            return a.is_close_to(b, _call)
        else:
            raise TypeError(
                f"Not implemented for {type(a).__name__} and {type(b).__name__}"
            )

    return _call
