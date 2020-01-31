import pytest
import numpy as np
from mckit.utils import digits_in_fraction_for_str, significant_digits, get_decades, make_hash


@pytest.mark.parametrize("value,expected", [
    (15.0, 1),
    (10.0, 1),
    (1.5, 0),
    (1.0,  0),
    (0.5, -1),
    (0.0011, -3),    # TODO dvp: check this strange values
    (0.001, -4),     #
    (0.01, -3),      #
    (0.1, -2),
    (1e-12, -13),
    (0.0, 0),
    (-1e-12, -13),
    (-0.01, -3),
    (-0.001, -4),
    (-0.0011, -3),
    (-0.5, -1),
    (-1.0, 0),
    (-1.5, 0),
    (-10.0, 1),
    (-15.0, 1),
])
def test_get_decades(value, expected):
    assert get_decades(value) == expected


@pytest.mark.parametrize("value, reltol, resolution, expected", [
    (1.0, 1e-12, None, 0),
    (1.1, 1e-12, None, 1),
    (11.0, 1e-12, None, 0),
    (11.11, 1e-12, None, 2),  # TODO dvp: looks like the function should be called `digits_in_fraction`.
])
def test_significant_digits(value, reltol, resolution, expected):
    actual = significant_digits(value, reltol, resolution)
    assert actual == expected


@pytest.mark.parametrize("value, reltol, resolution, expected", [
    (1.0, 1e-12, None, 0),
    (1.1, 1e-12, None, 1),
    (11.0, 1e-12, None, 0),
    (11.12, 1e-12, None, 2),
    (11.123, 1e-12, None, 3),
    (1.123456789123456, 1e-12, None, 12),
    (0.0001, 1e-12, None, 0),
    (0.000123456, 1e-12, None, 9),
])
def test_digits_in_fraction(value, reltol, resolution, expected):
    actual = digits_in_fraction_for_str(value, reltol, resolution)
    assert actual == expected


@pytest.mark.parametrize("values", [
    ('abc',),
    ({1: {'a': 2}}, np.arange(10)),
])
def test_make_hash(values):
    make_hash(*values)
