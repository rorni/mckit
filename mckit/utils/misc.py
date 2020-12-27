from typing import Any, Dict, Tuple, cast

import collections
import collections.abc as abc
import functools
import itertools

from copy import deepcopy

import numpy as np

from numpy import ndarray

from ..constants import FLOAT_TOLERANCE

MAX_DIGITS = np.finfo(float).precision

# def digits_in_fraction_for_str(
#         value: float,
#         reltol: float = FLOAT_TOLERANCE,
#         resolution: float = None
# ) -> int:
#     if value == 0.0:
#         return 0
#     if value < 0.0:
#         value = -value
#     if resolution and value < resolution:
#         return 0
#     max_remainder = value * reltol
#     s_value = str(value)
#     _, s_rem = s_value.split('.')
#
#     def _iter():
#         ord0 = ord('0')
#         m = 0.1
#         for c in s_rem:
#             yield m * (ord(c) - ord0)
#             m *= 0.1
#
#     rem = np.flip(np.fromiter(_iter(), np.float))
#     n = np.searchsorted(rem, max_remainder)
#     return rem.size - n


def significant_digits(value, reltol=FLOAT_TOLERANCE, resolution=None):
    """The minimum number of significant digits to provide relative tolerance.

    Parameters
    ----------
    value : float
        The value to be checked.
    reltol : float
        Relative tolerance needed to represent the value.
    resolution : float
        The threshold value, below which numbers are believed to be zero.
        Default: None - not applied.

    Returns
    -------
    digits : int
        The number of digits.
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


def get_decades(value):
    # TODO dvp: check if math.frexp is applicable, this mostly works but some test for pretty_print fail.
    # if value == 0.0:
    #     return 0
    # mantissa, exponent = math.frexp(value)  # type: float, int
    # if -0.5 <= mantissa <= 0.5:
    #     decades: int = math.floor(LG2 * (exponent - 1))
    # else:
    #     decades: int = math.floor(LG2 * exponent)
    # return decades

    if value != 0:
        decimal_power = np.log10(
            abs(value)
        )  # TODO dvp: log10 will be called billion times on C-model
    else:
        decimal_power = 0
    decades = np.trunc(decimal_power)
    if decimal_power < 0:
        decades -= 1
    return int(decades)


def significant_array(
    array: ndarray, reltol: float = FLOAT_TOLERANCE, resolution: float = None
) -> ndarray:
    """The minimum number of significant digits to provide the desired relative and absolute tolerances."""
    result = np.empty_like(array, dtype=int)
    for index in zip(*map(np.ravel, np.indices(array.shape))):
        result[index] = significant_digits(array[index], reltol, resolution)
    return result


def round_scalar(value, digits=None):
    """Rounds scalar value to represent the value in minimal form.

    Parameters
    ----------
    value : float
        The value to be rounded.
    digits : int
        The number of significant digits.

    Returns
    -------
    result : float
        Rounded value.
    """
    if digits is None:
        digits = significant_digits(value, FLOAT_TOLERANCE, FLOAT_TOLERANCE)
    return round(value, digits)


def round_array(array: ndarray, digits_array: ndarray = None) -> ndarray:
    """Rounds array to desired precision.

    Parameters
    ----------
    array :
        Array of values.
    digits_array :
        Array of corresponding significant digits.

    Returns
    -------
    result :
        Rounded array.
    """
    if digits_array is None:
        digits_array = significant_array(array, FLOAT_TOLERANCE, FLOAT_TOLERANCE)
    result = np.empty_like(array)
    for index in zip(*map(np.ravel, np.indices(array.shape))):
        result[index] = round_scalar(array[index], digits_array[index])
    return result


@functools.singledispatch
def are_equal(a, b) -> bool:
    return a is b or a == b


@are_equal.register
def _(a: str, b) -> bool:  # nomypy
    return a is b or a == b


@are_equal.register  # type: ignore
def _(a: ndarray, b) -> bool:
    return np.array_equal(a, b)


@are_equal.register  # type: ignore
def _(a: collections.abc.Iterable, b) -> bool:
    if not issubclass(type(b), collections.abc.Iterable):
        return False
    for ai, bi in itertools.zip_longest(a, b):
        if not are_equal(ai, bi):
            return False
    return True


@functools.singledispatch
def is_in(where, x) -> bool:
    if where is None:
        return False
    return x is where or x == where


@is_in.register  # type: ignore
def _(where: str, x) -> bool:
    return x is where or x == where


@is_in.register  # type: ignore
def _(where: tuple, x) -> bool:
    for i in where:
        if is_in(i, x):
            return True
    return False


@is_in.register  # type: ignore
def _(where: collections.abc.Callable, x) -> bool:
    return where(x)


@is_in.register  # type: ignore
def _(where: collections.abc.Container, x) -> bool:
    return x in where


def filter_dict(a: Dict[Any, Any], *drop_items) -> Dict[Any, Any]:
    res = {}
    for k, v in a.items():
        # if drop_items is None or not check_if_is_in(k, *drop_items):
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
def make_hashable(x):
    raise TypeError(f"Don't know how to make {type(x).__name__} objects hashable")


@make_hashable.register  # type: ignore
def _(x: collections.abc.Hashable):
    return x


@make_hashable.register  # type: ignore
def _(x: str):
    return x


@make_hashable.register  # type: ignore
def _(x: collections.abc.Mapping) -> Tuple:
    return tuple(map(lambda i: (i[0], make_hashable(i[1])), x.items()))


@make_hashable.register  # type: ignore
def _(x: collections.abc.Iterable) -> Tuple:
    return tuple(map(make_hashable, x))


def make_hash(*items) -> int:
    if 1 < len(items):
        return make_hash(tuple(map(make_hash, items)))
    return hash(make_hashable(items[0]))


def is_sorted(a: np.ndarray) -> bool:
    return cast(bool, np.all(np.diff(a) > 0))


def mids(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a[1:] + a[:-1])


def prettify_float(x: float, fmt: str = "{:.13g}") -> str:
    if x.is_integer():
        return str(int(x))
    else:
        return fmt.format(x)
