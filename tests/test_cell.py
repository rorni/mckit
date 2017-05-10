import unittest

import numpy as np

from mckit.cell import _complement, _intersection, _union


class TestCell(unittest.TestCase):
    def test_creation(self):
        pass

    def test_test_point(self):
        pass

    def test_test_region(self):
        pass

    def test_transform(self):
        pass

    def test_get_surfaces(self):
        pass


class TestOperations(unittest.TestCase):
    def test_complement(self):
        for i, (arg, ans) in enumerate(complement_cases):
            with self.subTest(i=i):
                comp = _complement(arg)
                if isinstance(comp, np.ndarray):
                    for x, y in zip(ans, comp):
                        self.assertEqual(x, y)
                else:
                    self.assertEqual(comp, ans)

    def test_intersection(self):
        for i, (arg1, arg2, ans) in enumerate(intersection_cases):
            with self.subTest(i=i):
                res = _intersection(arg1, arg2)
                if isinstance(res, np.ndarray):
                    for x, y in zip(ans, res):
                        self.assertEqual(x, y)
                else:
                    self.assertEqual(res, ans)

    def test_union(self):
        for i, (arg1, arg2, ans) in enumerate(union_cases):
            with self.subTest(i=i):
                res = _union(arg1, arg2)
                if isinstance(res, np.ndarray):
                    for x, y in zip(ans, res):
                        self.assertEqual(x, y)
                else:
                    self.assertEqual(res, ans)


complement_cases = [
    (0, 0), (-1, +1), (+1, -1), (np.array([-1, 0, 1]), [1, 0, -1])
]

intersection_cases = [
    (-1, -1, -1), (-1, 0, -1), (-1, 1, -1), (0, -1, -1), (0, 0, 0), (0, 1, 0),
    (1, -1, -1), (1, 0, 0), (1, 1, 1),
    (np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1]),
     np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1]), [-1, -1, -1, -1, 0, 0, -1, 0, 1])
]

union_cases = [
    (-1, -1, -1), (-1, 0, 0), (-1, 1, 1), (0, -1, 0), (0, 0, 0), (0, 1, 1),
    (1, -1, 1), (1, 0, 1), (1, 1, 1),
    (np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1]),
     np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1]), [-1, 0, 1, 0, 0, 1, 1, 1, 1])
]

if __name__ == '__main__':
    unittest.main()
