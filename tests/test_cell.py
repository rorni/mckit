import unittest

import numpy as np

from mckit.cell import _complement, _intersection, _union, Cell
from mckit.constants import *
from mckit.surface import Plane, create_surface, Surface
from mckit.transformation import Transformation
from mckit.fmesh import Box


class TestCell(unittest.TestCase):
    def test_creation(self):
        for i, (geom, opt) in enumerate(cell_creation_cases):
            with self.subTest(i=i):
                c = Cell(geom, **opt)
                for op1, op2 in zip(geom, c._expression):
                    self.assertEqual(op1, op2)
                for k, v in opt.items():
                    self.assertAlmostEqual(v, c[k])

    def test_test_point(self):
        for i, (geom, points, ans) in enumerate(cell_test_point_cases):
            with self.subTest(i=i):
                c = Cell(geom)
                results = c.test_point(points)
                for j in range(len(ans)):
                    self.assertEqual(results[j], ans[j])

    def test_test_region(self):
        for i, (geom, ans) in enumerate(cell_test_region_cases):
            with self.subTest(i=i):
                c = Cell(geom)
                r = c.test_box(region)
                self.assertEqual(r, ans)

    def test_transform(self):
        tr = Transformation(translation=[1, 3, -2])
        geom = [create_surface('SO', 5), 'C', create_surface('SX', 5, 5), 'C',
                'I', create_surface('SY', 5, 5), 'C', 'U']
        c = Cell(geom, OPT1=1, OPT2=2)
        c_tr = c.transform(tr)
        for op1, op2 in zip(c._expression, c_tr._expression):
            if isinstance(op1, Surface):
                sur1 = op1.transform(tr)
                np.testing.assert_almost_equal(sur1._center, op2._center)
        for k, v in c.items():
            self.assertEqual(v, c_tr[k])

    def test_get_surfaces(self):
        s1 = Plane(EX, -5)
        s2 = Plane(EY, -5)
        s3 = Plane(EZ, -5)
        c = Cell([s1, 'C', s2, 'I', s3, 'U', s2, 'C', 'I', s3, 'I'])
        surfs = c.get_surfaces()
        self.assertEqual(len(surfs.difference(set([s1, s2, s3]))), 0)
        self.assertEqual(len(set([s1, s2, s3]).difference(surfs)), 0)


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

cell_creation_cases = [
    ([Plane(EX, 5), Plane(EX, -5), 'C', 'I', Plane(EY, 5), Plane(EY, -5), 'C',
      'I', Plane(EZ, 5), Plane(EZ, -5), 'C', 'I', 'I', 'I'],
     {'U': 5, 'IMP:N': 1, 'DEN': -4.3})
]

cell_test_point_cases = [
    ([Plane(EX, 5), Plane(EX, -5), 'C', 'I', Plane(EY, 5), Plane(EY, -5), 'C',
      'I', Plane(EZ, 5), Plane(EZ, -5), 'C', 'I', 'I', 'I'],
     [[-6, -6, -6], [-6, -6, -4], [-6, -6, 4], [-6, -6, 6],
      [-6, -4, -6], [-6, -4, -4], [-6, -4, 4], [-6, -4, 6],
      [-6,  4, -6], [-6,  4, -4], [-6,  4, 4], [-6,  4, 6],
      [-6,  6, -6], [-6,  6, -4], [-6,  6, 4], [-6,  6, 6],
      [-4, -6, -6], [-4, -6, -4], [-4, -6, 4], [-4, -6, 6],
      [-4, -4, -6], [-4, -4, -4], [-4, -4, 4], [-4, -4, 6],
      [-4,  4, -6], [-4,  4, -4], [-4,  4, 4], [-4,  4, 6],
      [-4,  6, -6], [-4,  6, -4], [-4,  6, 4], [-4,  6, 6]],
     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1]),

    ([Plane(EX, 5), Plane(EX, -5), 'C', 'I', create_surface('SX', 7, 1), 'C',
      'U'],
     [[-6, 0, 0], [-4, 0, 0], [4, 0, 0], [5.5, 0, 0], [7, 0, 0], [9, 0, 0]],
     [-1, 1, 1, -1, 1, -1])
]

region = Box([-5, -5, -5], [10, 0, 0], [0, 10, 0], [0, 0, 10])

cell_test_region_cases = [
    ([create_surface('SO', 3), 'C'], 0),
    ([Plane(EX, 6), Plane(EX, -6), 'C', 'I', Plane(EY, 6), Plane(EY, -6), 'C',
      'I', Plane(EZ, 6), Plane(EZ, -6), 'C', 'I', 'I', 'I'], 1),
    ([Plane(EX, 10), Plane(EX, 6), 'C', 'I', create_surface('SX', 5.5, 1),
      'C', 'U'], 0),
    ([Plane(EX, 10), Plane(EX, 6), 'C', 'I', create_surface('SX', 4.5, 1),
      'C', 'U'], 0),
    ([Plane(EX, 10), Plane(EX, 6), 'C', 'I', create_surface('SX', 6.5, 1),
      'C', 'U'], -1),
]

if __name__ == '__main__':
    unittest.main()
