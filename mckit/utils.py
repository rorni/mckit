import numpy as np


MAX_DIGITS = np.finfo(float).precision


def significant_digits(value, reltol=1.e-12):
    """The minimum number of significant digits to provide relative tolerance.

    Parameters
    ----------
    value : float
        The value to be checked.
    reltol : float
        Relative tolerance needed to represent the value.

    Returns
    -------
    digits : int
        The number of digits.
    """
    low = 0
    high = MAX_DIGITS
    while high - low > 1:
        p = round(0.5 * (high + low))
        v = np.round(value, p)
        d = max(abs(value), abs(v))
        if d != 0 and abs(value - v) / d > reltol:
            low = p
        else:
            high = p
    return high


def round_scalar(value, reltol=1.e-12, resolution=None):
    """Rounds scalar value to represent the value in minimal form.

    Parameters
    ----------
    value : float
        The value to be rounded.
    reltol : float
        The relative tolerance.
    resolution : float
        The threshold value, below which numbers are believed to be zero.
        Default: None - not applied.

    Returns
    -------
    result : float
        Rounded value.
    """
    if resolution and abs(value) < resolution:
        return 0
    prec = significant_digits(value, reltol)
    return round(value, prec)


def round_array(array, reltol=1.e-12, resolution=None):
    """Rounds array to desired precision.

    Parameters
    ----------
    array : numpy.ndarray
        Array of values.
    reltol : float
        The relative tolerance.
    resolution : float
        The threshold value, below which numbers are believed to be zero.
        Default: None - not applied.

    Returns
    -------
    result : numpy.ndarray
        Rounded array.
    """
    result = np.empty_like(array)
    for index in zip(*map(np.ravel, np.indices(array.shape))):
        result[index] = round_scalar(array[index], reltol, resolution)
    return result
