from __future__ import annotations

import numpy as np

from mckit.utils import get_decades, significant_digits

# def run_digits_in_fraction(a: np.ndarray) -> None:
#     map(digits_in_fraction_for_str, a)
#
#
# def test_digits_in_fraction(benchmark):
#     values = (np.random.rand(1000) - 0.5) * 1000.0
#     benchmark(run_digits_in_fraction, values)

# Name (time in ns)                Min       Max      Mean   StdDev    Median     IQR    Outliers  OPS (Mops/s)  Rounds  Iterations
# test_digits_in_fraction     186.9630  720.5936  194.0950  26.6555  190.4080  1.2957  3901;10481        5.1521  192753          27


def run_significant_digits(a: np.ndarray) -> None:
    map(significant_digits, a)


def test_significant_digits(benchmark):
    values = (np.random.rand(1000) - 0.5) * 1000.0
    benchmark(run_significant_digits, values)


# Name (time in ns)                Min       Max      Mean   StdDev    Median      IQR   Outliers  OPS (Mops/s)  Rounds  Iterations
# test_significant_digits     185.8118  811.3717  200.8212  30.4897  190.1493  21.4075  2785;2766        4.9796  193839          27


def run_get_decades(a: np.ndarray) -> None:
    return np.fromiter(map(get_decades, a), np.int16)


def test_get_decades(benchmark):
    values = (np.random.rand(1000) - 0.5) * 1000.0
    benchmark(run_get_decades, values)


# Name (time in ms)        Min     Max    Mean  StdDev  Median     IQR  Outliers       OPS  Rounds  Iterations
# test_get_decades      2.8199  5.4146  2.9415  0.2459  2.8766  0.0728     12;45  339.9577     331           1
