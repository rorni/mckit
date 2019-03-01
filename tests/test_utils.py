import pytest

from mckit.utils import *


@pytest.mark.parametrize('res', [None, 1.e-2, 1.e-4, 1.e-6, 1.e-8])
@pytest.mark.parametrize('tol', [1.e-2, 1.e-4, 1.e-6, 1.e-8])
@pytest.mark.parametrize('value', [
    5.0000000000001, 3.45, -6.986574864, -4, 8, 0.0, 0.00000000000,
    3.48e-8, 4.870001e+6, 0, 0.00000000000001, 1.e-12, -1.e-12, -1.e-3,
    4.872342e+35
])
def test_significant_digits(value, tol, res):
    prec = significant_digits(value, tol, res)
    approx = round(value, prec)
    if res and abs(value) < res:
        assert approx == 0
    elif value != approx:
        rel = abs(value - approx) / max(abs(value), abs(approx))
        assert rel <= tol
        wrong = round(value, prec - 1)
        print(approx, wrong, prec)
        rel = abs(value - wrong) / max(abs(value), abs(wrong))
        assert rel > tol


@pytest.mark.parametrize('value, reltol, resolution, answer', [
    (5.4320000, 1.e-4, None, 5.4320),
    (5.4320001, 1.e-4, None, 5.4320),
    (1.e-12, 1.e-4, None, 1.e-12),
    (1.e-12, 1.e-4, 1.e-14, 1.e-12),
    (1.e-15, 1.e-4, 1.e-14, 0),
    (-5.4320000, 1.e-4, None, -5.4320),
    (-5.4320001, 1.e-4, None, -5.4320),
    (-1.e-12, 1.e-4, None, -1.e-12),
    (-1.e-12, 1.e-4, 1.e-14, -1.e-12),
    (-1.e-15, 1.e-4, 1.e-14, 0),
])
def test_round_scalar(value, reltol, resolution, answer):
    p = significant_digits(value, reltol, resolution)
    result = round_scalar(value, p)
    assert result == answer


@pytest.mark.parametrize('array, reltol, resolution, answer', [
    (np.array([5.4320000]), 1.e-4, None, np.array([5.4320])),
    (np.array([5.4320001]), 1.e-4, None, np.array([5.4320])),
    (np.array([1.e-12]), 1.e-4, None, np.array([1.e-12])),
    (np.array([1.e-12]), 1.e-4, 1.e-14, np.array([1.e-12])),
    (np.array([1.e-15]), 1.e-4, 1.e-14, np.array([0])),
    (np.array([-5.4320000]), 1.e-4, None, np.array([-5.4320])),
    (np.array([-5.4320001]), 1.e-4, None, np.array([-5.4320])),
    (np.array([-1.e-12]), 1.e-4, None, np.array([-1.e-12])),
    (np.array([-1.e-12]), 1.e-4, 1.e-14, np.array([-1.e-12])),
    (np.array([-1.e-15]), 1.e-4, 1.e-14, np.array([0])),
    (np.array([5.4320000, 5.4320001, 1.e-12, -5.4320000, -5.4320001, -1.e-12]),
     1.e-4, None, np.array([5.4320, 5.4320, 1.e-12, -5.4320, -5.4320, -1.e-12])),
    (np.array([5.4320000, 5.4320001, 1.e-12, -5.4320000, -5.4320001, -1.e-12]),
     1.e-4, 1.e-14,
     np.array([5.4320, 5.4320, 1.e-12, -5.4320, -5.4320, -1.e-12])),
    (np.array([[5.4320000, 5.4320001, 1.e-12], [-5.4320000, -5.4320001, -1.e-12]]),
     1.e-4, None,
     np.array([[5.4320, 5.4320, 1.e-12], [-5.4320, -5.4320, -1.e-12]])),
    (np.array([[5.4320000, 5.4320001], [1.e-12, -5.4320000], [-5.4320001, -1.e-12]]),
     1.e-4, 1.e-14,
     np.array([[5.4320, 5.4320], [1.e-12, -5.4320], [-5.4320, -1.e-12]])),
    (np.array([5.4320000, 1.e-12, 1.e-15, -5.4320000, -1.e-12, -1.e-15]),
     1.e-4, 1.e-14,
     np.array([5.4320, 1.e-12, 0, -5.4320, -1.e-12, -0])),
    (np.array([[5.4320000, 1.e-12, 1.e-15], [-5.4320000, -1.e-12, -1.e-15]]),
     1.e-4, 1.e-14,
     np.array([[5.4320, 1.e-12, 0], [-5.4320, -1.e-12, -0]])),
])
def test_round_array(array, reltol, resolution, answer):
    digarr = significant_array(array, reltol, resolution)
    result = round_array(array, digarr)
    assert result.shape == answer.shape
    assert np.all(result == answer)
