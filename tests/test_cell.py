import unittest

import numpy as np

from mckit.cell import _complement, _intersection, _union, Cell, GeometryNode
from mckit.surface import create_surface, Surface
from mckit.constants import *
from mckit.transformation import Transformation
from mckit.fmesh import Box
from mckit.universe import Universe

from tests.cell_test_data.geometry_test_data import *

surfaces = {}
geoms = []


def setUpModule():
    for name, (kind, params) in surface_data.items():
        surfaces[name] = create_surface(kind, *params, name=name)
    for g in create_geom:
        geoms.append(create_node(g[0], g[1]))


def create_node(kind, args):
    new_args = []
    for g in args:
        if isinstance(g, tuple):
            g = create_node(g[0], g[1])
        else:
            g = surfaces[g]
        new_args.append(g)
    return GeometryNode(kind, *new_args)


class TestGeometryNode(unittest.TestCase):
    def test_from_polish(self):
        for i, raw_data in enumerate(polish_geoms):
            data = []
            for t in raw_data:
                if isinstance(t, int):
                    data.append(surfaces[t])
                else:
                    data.append(t)
            with self.subTest(i=i):
                g = GeometryNode.from_polish_notation(data)
                self.assertEqual(g, geoms[i])

    def test_complement(self):
        for i, raw_data in enumerate(complement_geom):
            ans = create_node(raw_data[0], raw_data[1])
            with self.subTest(i=i):
                c = geoms[i].complement()
                self.assertEqual(c, ans)

    def test_intersection(self):
        for i, case in enumerate(intersection_geom):
            for j, g in enumerate(case):
                ans = create_node(g[0], g[1]).clean()
                with self.subTest(msg='i={0}, j={1}'.format(i, j)):
                    ind = j if j < i else j + 1
                    test = geoms[i].intersection(geoms[ind])
                    if test != ans:
                        print(test, ' =====> ', ans)
                    self.assertEqual(ans, test)

    def test_union(self):
        for i, case in enumerate(union_geom):
            for j, g in enumerate(case):
                ans = create_node(g[0], g[1]).clean()
                with self.subTest(msg='i={0}, j={1}'.format(i, j)):
                    ind = j if j < i else j + 1
                    test = geoms[i].union(geoms[ind])
                    if test != ans:
                        print(test, ' =====> ', ans)
                    self.assertEqual(ans, test)

    def test_test_point(self):
        for i, g in enumerate(geoms):
            for j, p in enumerate(node_points):
                with self.subTest(msg='geom={0}, point={1}'.format(i, j)):
                    t = g.test_point(p)
                    self.assertEqual(t, node_test_point_ans[i][j])
            t = g.test_point(np.array(node_points))
            self.assertListEqual(list(t), node_test_point_ans[i])

    def test_complexity(self):
        for i, g in enumerate(geoms):
            with self.subTest(i=i):
                c = g.complexity()
                self.assertEqual(c, node_complexity_data[i])

    def test_test_box(self):
        for i, b_data in enumerate(node_boxes_data):
            box = Box(b_data['base'], b_data['ex'], b_data['ey'], b_data['ez'])
            for j, g in enumerate(geoms):
                with self.subTest(msg='box {0}, geom {1}, only result'.format(i, j)):
                    r = g.test_box(box, return_simple=False)
                    self.assertEqual(r, node_box_ans[i][j][0])
                with self.subTest(msg='box {0}, geom {1}, result+geom'.format(i, j)):
                    r, sg = g.test_box(box, return_simple=True)
                    ans = set()
                    for a in node_box_ans[i][j][1]:
                        ans.add(create_node(a[0], a[1]))
                    if sg != ans:
                        print(len(sg))
                        print(i, ' ', j, '=', list(sg)[0], ' ===> ', list(ans)[0])
                        print(i, ' ', j, '=', list(sg)[1], ' ===> ', list(ans)[1])
                    self.assertEqual(r, node_box_ans[i][j][0])
                    self.assertEqual(sg, ans)

    def test_bounding_box(self):
        base = [-10, -10, -10]
        dims = [30, 30, 30]
        gb = Box(base, [dims[0], 0, 0], [0, dims[1], 0], [0, 0, dims[2]])
        tol = 0.2
        for i, (ag, limits) in enumerate(zip(geoms, node_bounding_box)):
            with self.subTest(i=i):
                bb = ag.bounding_box(box=gb, tol=tol)
                for j in range(3):
                    if limits[j][0] is None:
                        limits[j][0] = base[j]
                    if limits[j][1] is None:
                        limits[j][1] = base[j] + dims[j]
                    self.assertLessEqual(bb.base[j], limits[j][0])
                    self.assertGreaterEqual(bb.base[j], limits[j][0] - tol)
                    self.assertGreaterEqual(bb.base[j] + bb.scale[j], limits[j][1])
                    self.assertLessEqual(bb.base[j] + bb.scale[j], limits[j][1] + tol)

    def test_volume(self):
        for i, b_data in enumerate(node_boxes_data):
            box = Box(b_data['base'], b_data['ex'], b_data['ey'], b_data['ez'])
            for j, g in enumerate(geoms):
                with self.subTest(msg='box {0}, geom {1}'.format(i, j)):
                    v = g.volume(box, min_volume=1.e-3)
                    v_ans = node_volume[i][j]
                    self.assertAlmostEqual(v, v_ans, delta=v_ans * 0.001)


class TestCell(unittest.TestCase):
    def test_creation(self):
        for i, pol_data in enumerate(polish_geoms):
            pol_geom = []
            for x in pol_data:
                if isinstance(x, int):
                    pol_geom.append(surfaces[x])
                else:
                    pol_geom.append(x)
            with self.subTest(msg="polish #{0}".format(i)):
                ans = geoms[i]
                cell = Cell(pol_geom)
                # print(str(ag))
                self.assertEqual(cell, ans)
        for i, ag in enumerate(geoms):
            with self.subTest(msg="additive geom #{0}".format(i)):
                cell = Cell(ag)
                self.assertEqual(cell, ag)

    def test_intersection(self):
        for i, ag1 in enumerate(geoms):
            c1 = Cell(ag1, **cell_kwargs)
            for j, ag2 in enumerate(geoms):
                if i == j:
                    continue
                ind = j-1 if j > i else j
                g = intersection_geom[i][ind]
                ans = create_node(g[0], g[1]).clean()
                with self.subTest(msg='additive i={0}, j={1}'.format(i, j)):
                    c2 = Cell(ag2)
                    u = c1.intersection(c2)
                    self.assertEqual(u, ans)
                    self.assertDictEqual(u, c1)

    def test_union(self):
        for i, ag1 in enumerate(geoms):
            c1 = Cell(ag1, **cell_kwargs)
            for j, ag2 in enumerate(geoms):
                if i == j:
                    continue
                ind = j-1 if j > i else j
                g = union_geom[i][ind]
                ans = create_node(g[0], g[1]).clean()
                with self.subTest(msg='additive i={0}, j={1}'.format(i, j)):
                    c2 = Cell(ag2)
                    u = c1.union(c2)
                    self.assertEqual(u, ans)
                    self.assertDictEqual(u, c1)

#     def test_populate(self):
#         for i, ag_out in enumerate(additives):
#             c_out = Cell(ag_out, name='Outer', U=4)
#             cells = []
#             geoms = []
#             for j, ag in enumerate(additives):
#                 if j != i:
#                     cells.append(Cell(ag, name=j))
#                     geoms.append(ag.intersection(ag_out))
#             universe = Universe(cells)
#             with self.subTest(i=i):
#                 output = c_out.populate(universe)
#                 for j in range(len(cells)):
#                     self.assertEqual(output[j], geoms[j])
#                     if 'U' in c_out.keys():
#                         self.assertEqual(c_out['U'], output[j]['U'])
#                         output[j].pop('U')
#                     self.assertDictEqual(output[j], cells[j])
#                 c_out['FILL'] = universe
#                 output = c_out.populate()
#                 for j in range(len(cells)):
#                     self.assertEqual(output[j], geoms[j])
#                     if 'U' in c_out.keys():
#                         self.assertEqual(c_out['U'], output[j]['U'])
#                         output[j].pop('U')
#                     self.assertDictEqual(output[j], cells[j])
#
#     @unittest.expectedFailure  # Surface equality should be developed.
#     def test_transform(self):
#         tr = Transformation(translation=[1, 3, -2])
#         surfaces_tr = {k: s.transform(tr) for k, s in surfaces.items()}
#         for i, (pol_data, ans_data) in enumerate(ag_polish_data):
#             pol_geom = []
#             for x in pol_data:
#                 if isinstance(x, int):
#                     pol_geom.append(surfaces[x])
#                 else:
#                     pol_geom.append(x)
#             with self.subTest(msg="cell transform #{0}".format(i)):
#                 ans = [produce_term(a, surf=surfaces_tr) for a in ans_data]
#                 ans_geom = AdditiveGeometry(*ans)
#                 cell = Cell(pol_geom).transform(tr)
#                 print(str(cell), '->', str(ans_geom), '->', len(cell.terms), '->', len(ans_geom.terms))
#                 self.assertSetEqual(cell.terms, ans_geom.terms)
#
#     # @unittest.skip
    def test_simplify(self):
        for i, ag in enumerate(geoms):
            cell = Cell(ag)
            pol_geom = []
            for x in simple_geoms[i]:
                if isinstance(x, int):
                    pol_geom.append(surfaces[x])
                else:
                    pol_geom.append(x)
            with self.subTest(i=i):
                s = cell.simplify(min_volume=0.1, box=Box([-10, -10, -10], [26, 0, 0], [0, 20, 0], [0, 0, 20]))
                # print(i, len(s))
                # for ss in s:
                #     print(str(ss))
                ans = Cell(pol_geom)
                self.assertEqual(ans, s)


class TestOperations(unittest.TestCase):
    def test_complement(self):
        for i, (arg, ans) in enumerate(cell_complement_cases):
            with self.subTest(i=i):
                comp = _complement(np.array(arg))
                if isinstance(comp, np.ndarray):
                    for x, y in zip(ans, comp):
                        self.assertEqual(x, y)
                else:
                    self.assertEqual(comp, ans)

    def test_intersection(self):
        for i, (arg1, arg2, ans) in enumerate(cell_intersection_cases):
            with self.subTest(i=i):
                res = _intersection(arg1, arg2)
                if isinstance(res, np.ndarray):
                    for x, y in zip(ans, res):
                        self.assertEqual(x, y)
                else:
                    self.assertEqual(res, ans)

    def test_union(self):
        for i, (arg1, arg2, ans) in enumerate(cell_union_cases):
            with self.subTest(i=i):
                res = _union(arg1, arg2)
                if isinstance(res, np.ndarray):
                    for x, y in zip(ans, res):
                        self.assertEqual(x, y)
                else:
                    self.assertEqual(res, ans)


if __name__ == '__main__':
    unittest.main()
