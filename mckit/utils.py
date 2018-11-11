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
