import os
from pathlib import Path
from typing import Any, Optional, Set, Dict
import numpy as np
from numpy import ndarray

MAX_DIGITS = np.finfo(float).precision


def digits_in_fraction_for_str(value: float, reltol: float = 1e-12, resolution: float = None) -> int:
    if value == 0.0:
        return 0
    if value < 0.0:
        value = -value
    if resolution and value < resolution:
        return 0
    max_remainder = value * reltol
    s_value = str(value)
    _, s_rem = s_value.split('.')

    def _iter():
        ord0 = ord('0')
        m = 0.1
        for c in s_rem:
            yield m * (ord(c) - ord0)
            m *= 0.1

    rem = np.flip(np.fromiter(_iter(), np.float))
    n = np.searchsorted(rem, max_remainder)
    return rem.size - n


def significant_digits(value, reltol=1.e-12, resolution=None):
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
    while high - low > 1:
        p = round(0.5 * (high + low))
        v = round(value, p)
        d = max(abs(value), abs(v))
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
        decpow = np.log10(abs(value))  # TODO dvp: log10 will be called billion times on C-model
    else:
        decpow = 0
    decades = np.trunc(decpow)
    if decpow < 0:
        decades -= 1
    return int(decades)


def significant_array(
        array: ndarray,
        reltol: float = 1.e-12,
        resolution: float = None,
) -> ndarray:
    """The minimum number of significant digits to provide relative tolerance.
    """
    result = np.empty_like(array, dtype=int)
    for index in zip(*map(np.ravel, np.indices(array.shape))):
        result[index] = significant_digits(array[index], reltol, resolution)
    return result


def round_scalar(value, digits):
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
    return round(value, digits)


def round_array(array: ndarray, digarr: ndarray):
    """Rounds array to desired precision.

    Parameters
    ----------
    array : numpy.ndarray
        Array of values.
    digarr : arraylike[int]
        Array of corresponding significant digits.

    Returns
    -------
    result : numpy.ndarray
        Rounded array.
    """
    result = np.empty_like(array)
    for index in zip(*map(np.ravel, np.indices(array.shape))):
        result[index] = round_scalar(array[index], digarr[index])
    return result


def deep_copy_dict(
        a: Dict[Any, Any],
        drop_item: Optional[Any] = None,
        drop_set: Optional[Set[Any]] = None
) -> Dict[Any, Any]:
    res = {}
    for k, v in a.items():
        if drop_item is None or k != drop_item:
            if drop_set is None or k not in drop_set:
                if isinstance(v, dict):
                    v = deep_copy_dict(v)
                res[k] = v
    return res


def assert_all_paths_exist(*paths):
    def apply(p: Path):
        assert p.exists(), "Path \"{}\" doesn't exist".format(p)

    map(apply, paths)


def make_dirs(*dirs):
    def apply(d: Path):
        d.mkdir(parents=True, exist_ok=True)

    map(apply, dirs)


def get_root_dir(environment_variable_name, default):
    return Path(os.getenv(environment_variable_name, default)).expanduser()


def is_sorted(a: np.ndarray) -> bool:
    return np.all(np.diff(a) > 0)
