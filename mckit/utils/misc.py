"""Generic utitility methods."""
from __future__ import annotations

from typing import Any, Dict, Tuple, cast

import collections
import collections.abc
import functools
import itertools

from copy import deepcopy

import numpy as np

from numpy import ndarray

from ..constants import FLOAT_TOLERANCE

MAX_DIGITS = np.finfo(float).precision


def significant_digits(
    value: float, reltol: float = FLOAT_TOLERANCE, resolution: float = None
) -> int:
    """The minimum number of significant digits to provide relative tolerance.

    Args:
        value:  The value to be checked.
        reltol: Relative tolerance needed to represent the value.
        resolution:  The threshold value, below which numbers are believed to be zero, optional.

    Returns:
        The number of significant digits.
    """
    if value == 0.0 or resolution and abs(value) < resolution:
        return 0
    dec = get_decades(value)
    low = min(dec, 0)
    high = MAX_DIGITS
    d = d0 = abs(value)
    while high - low > 1:
        p = round(0.5 * (high + low))
        v = round(value, p)
        d = max(d0, abs(v))
        if abs(value - v) > reltol * d:
            low = p
        else:
            high = p
    v = round(value, low)
    if abs(value - v) < reltol * d:
        return low
    else:
        return high


# LG2 = math.log10(2.0)


def get_decades(value: int | float) -> int:
    """Compute number of digits needed to represent integer part of 'value' in fixed format.

    Args:
        value: ... to check

    Returns:
        Number of decades.
    """
    # TODO dvp: check if math.frexp is applicable,
    #           this mostly works but some test for pretty_print fail.
    # if value == 0.0:
    #     return 0
    # mantissa, exponent = math.frexp(value)  # type: float, int
    # if -0.5 <= mantissa <= 0.5:
    #     decades: int = math.floor(LG2 * (exponent - 1))
    # else:
    #     decades: int = math.floor(LG2 * exponent)
    # return decades

    if value != 0:
        decimal_power = np.log10(abs(value))
    else:
        decimal_power = 0
    decades = np.trunc(decimal_power)
    if decimal_power < 0:
        decades -= 1
    return int(decades)


def significant_array(
    array: ndarray, reltol: float = FLOAT_TOLERANCE, resolution: float = None
) -> ndarray:
    """Compute the minimum numbers of significant digits to achieve desired tolerance."""
    result: ndarray = np.empty_like(array, dtype=int)
    for index in zip(*map(np.ravel, np.indices(array.shape))):
        result[index] = significant_digits(array[index], reltol, resolution)
    return result


def round_scalar(value: float, digits: int = None) -> float:
    """Rounds scalar value to represent the value in minimal form.

    Args:
        value: The value to be rounded.
        digits: The number of significant digits, optional.

    Returns:
        Rounded value.
    """
    if digits is None:
        digits = significant_digits(value, FLOAT_TOLERANCE, FLOAT_TOLERANCE)
    return round(value, digits)


def round_array(array: ndarray, digits_array: ndarray = None) -> ndarray:
    """Rounds array to desired precision.

    Args:
        array:   Array of values.
        digits_array:   Array of corresponding significant digits.

    Returns:
        Rounded array.
    """
    if digits_array is None:
        digits_array = significant_array(array, FLOAT_TOLERANCE, FLOAT_TOLERANCE)
    result: ndarray = np.empty_like(array)
    for index in zip(*map(np.ravel, np.indices(array.shape))):
        result[index] = round_scalar(array[index], digits_array[index])
    return result


@functools.singledispatch
def are_equal(a, b) -> bool:
    """Check if objects are equal dispatching method."""
    return a is b or a == b


@are_equal.register
def _(a: str, b: str) -> bool:
    return a is b or a == b


@are_equal.register
def _(a: ndarray, b: ndarray) -> bool:
    return np.array_equal(a, b)


@are_equal.register
def _(a: collections.abc.Iterable, b) -> bool:  # : ignore=ANN001
    if not issubclass(type(b), collections.abc.Iterable):
        return False
    for ai, bi in itertools.zip_longest(a, b):
        if not are_equal(ai, bi):
            return False
    return True


@functools.singledispatch
def is_in(where, x) -> bool:  # : ignore=ANN001
    if where is None:
        return False
    return x is where or x == where


@is_in.register
def _(where: str, x) -> bool:  # : ignore=ANN001
    return x is where or x == where


@is_in.register
def _(where: tuple, x) -> bool:  # : ignore=ANN001
    for i in where:
        if is_in(i, x):
            return True
    return False


@is_in.register
def _(where: collections.abc.Callable, x) -> bool:  # : ignore=ANN001
    return where(x)  # type: ignore


@is_in.register
def _(where: collections.abc.Container, x) -> bool:  # : ignore=ANN001
    return x in where


def filter_dict(
    a: Dict[Any, Any],
    *drop_items,  # : ignore=ANN001
) -> Dict[Any, Any]:  # : ignore=ANN002
    """Create copy of a dictionary omitting some keys."""
    res = {}
    for k, v in a.items():
        if drop_items and is_in(drop_items, k):
            pass
        else:
            if isinstance(v, dict):
                res[k] = filter_dict(v, *drop_items)
            elif issubclass(type(v), collections.abc.Collection):
                res[k] = deepcopy(v)
            else:
                res[k] = v
    return res


@functools.singledispatch
def make_hashable(x):  # : ignore=ANN002
    raise TypeError(f"Don't know how to make {type(x).__name__} objects hashable")


@make_hashable.register
def _(x: collections.abc.Hashable):  # : ignore=ANN002
    return x


@make_hashable.register
def _(x: str):  # : ignore=ANN002
    return x


@make_hashable.register
def _(x: collections.abc.Mapping) -> Tuple:
    return tuple((k, make_hashable(v)) for k, v in x.items())


@make_hashable.register
def _(x: collections.abc.Iterable) -> Tuple:
    return tuple(map(make_hashable, x))


def make_hash(*items) -> int:  # : ignore=ANN001
    if 1 < len(items):
        return make_hash(tuple(map(make_hash, items)))
    return hash(make_hashable(items[0]))


def is_sorted(a: np.ndarray) -> bool:
    """Check if an array is sorted."""
    return cast(bool, np.all(np.diff(a) > 0))


def mids(a: np.ndarray) -> np.ndarray:
    """Get centers of bins presented with array 'a'."""
    result: ndarray = 0.5 * (a[1:] + a[:-1])
    return result


def prettify_float(x: float, fmt: str = "{:.13g}") -> str:
    """Format float in uniform way."""
    if x.is_integer():
        return str(int(x))
    else:
        return fmt.format(x)
