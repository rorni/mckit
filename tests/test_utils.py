import pytest

from mckit.utils import significant_digits


@pytest.mark.parametrize('tol', [
    1.e-2, 1.e-4, 1.e-6, 1.e-8
])
@pytest.mark.parametrize('value', [
    5.0000000000001, 3.45, -6.986574864, -4, 8, 0.0, 0.00000000000,
    3.48e-8, 4.8700001e+6, 0, 0.00000000000001, 1.e-12, -1.e-12, -1.e-3
])
def test_significant_digits(value, tol):
    prec = significant_digits(value, tol)
    approx = round(value, prec)
    if value != approx:
        rel = abs(value - approx) / max(abs(value), abs(approx))
        assert rel <= tol
        if prec > 1:
            wrong = round(value, prec - 1)
            print(approx, wrong)
            rel = abs(value - wrong) / max(abs(value), abs(wrong))
            assert rel > tol
