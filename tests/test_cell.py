import unittest

import numpy as np

from mckit.cell import _complement, _intersection, _union, Cell, GeometryTerm, \
    AdditiveGeometry
from mckit.surface import Plane, create_surface
from mckit.constants import *
from mckit.transformation import Transformation
from mckit.fmesh import Box
from mckit.universe import Universe

from tests.cell_test_data.geometry_test_data import *

surfaces = {}
terms = []
additives = []


def setUpModule():
    for name, (kind, params) in surface_data.items():
        surfaces[name] = create_surface(kind, *params, name=name)
    for tdata in term_data:
        terms.append(produce_term(tdata))
    for adata in additive_data:
        params = [terms[i] for i in adata]
        additives.append(AdditiveGeometry(*params))


def produce_term(data, surf=surfaces):
    p = {}
    for k, v in data.items():
        p[k] = {surf[i] for i in v}
    return GeometryTerm(**p)


class TestGeometryTerm(unittest.TestCase):
    def test_intersection(self):
        ans_data = term_intersection_ans
        for i, t1 in enumerate(terms):
            for j, t2 in enumerate(terms[:i] + terms[i+1:]):
                with self.subTest(msg='i={0} j={1}'.format(i, j)):
                    ti = t1.intersection(t2)
                    ans = {}
                    for k, v in ans_data[i][j].items():
                        ans[k] = {surfaces[i] for i in v}
                    ta = GeometryTerm(**ans)
                    self.assertSetEqual(ti.positive, ta.positive)
                    self.assertSetEqual(ti.negative, ta.negative)

    def test_complement(self):
        answers = []
        for tdata in term_complement_ans:
            answers.append(produce_term(tdata))
        for i, t in enumerate(terms):
            with self.subTest(i=i):
                c = t.complement()
                self.assertSetEqual(c.positive, answers[i].positive)
                self.assertSetEqual(c.negative, answers[i].negative)

    def test_is_superset(self):
        ans = is_subset_ans
        for i, t1 in enumerate(terms):
            for j, t2 in enumerate(terms):
                with self.subTest(msg='i={0} j={1}'.format(i, j)):
                    dj = t1.is_superset(t2)
                    self.assertEqual(dj, ans[j][i])

    def test_is_subset(self):
        ans = is_subset_ans
        for i, t1 in enumerate(terms):
            for j, t2 in enumerate(terms):
                with self.subTest(msg='i={0} j={1}'.format(i, j)):
                    dj = t1.is_subset(t2)
                    self.assertEqual(dj, ans[i][j])

    def test_is_empty(self):
        for i, t in enumerate(terms):
            with self.subTest(i=i):
                self.assertEqual(t.is_empty(), is_empty_ans[i])

    def test_box(self):
        box_data = term_box_data
        for box_param, ans_array in box_data:
            box = Box(box_param['base'], box_param['ex'], box_param['ey'],
                      box_param['ez'])
            for i, t in enumerate(terms):
                with self.subTest(msg='result only case={0}'.format(i)):
                    r = t.test_box(box)
                    self.assertEqual(r, ans_array[i][0])
        for box_param, ans_array in box_data:
            box = Box(box_param['base'], box_param['ex'], box_param['ey'],
                      box_param['ez'])
            for i, t in enumerate(terms):
                with self.subTest(msg='result and simple case={0}'.format(i)):
                    r, s = t.test_box(box, return_simple=True)
                    self.assertEqual(r, ans_array[i][0])
                    ans = {produce_term(a) for a in ans_array[i][1]}
                    self.assertEqual(s, ans)

    def test_complexity(self):
        for i, t in enumerate(terms):
            with self.subTest(i=i):
                c = t.complexity()
                self.assertEqual(c, term_complexity_ans[i])


class TestAdditiveGeometry(unittest.TestCase):
    def test_creation(self):
        for i, ag in enumerate(additives):
            ans = {produce_term(a) for a in ag_create[i]}
            self.assertEqual(ans, ag.terms)

    def test_contains(self):
        for i, ag1 in enumerate(additives):
            for j, ag2 in enumerate(additives):
                with self.subTest(msg='i={0}, j={1}'.format(i, j)):
                    c = ag1.contains(ag2)
                    self.assertEqual(c, ag_contains[i][j])

    def test_eq(self):
        for i, ag1 in enumerate(additives):
            for j, ag2 in enumerate(additives):
                with self.subTest(msg='i={0}, j={1}'.format(i, j)):
                    self.assertEqual(ag1 == ag2, ag_equiv[i][j])

    def test_union(self):
        for i, ag1 in enumerate(additives):
            for j, ag2 in enumerate(additives):
                with self.subTest(msg='additive i={0}, j={1}'.format(i, j)):
                    u = ag1.union(ag2)
                    ans = [produce_term(a) for a in ag_union1[i][j]]
                    ans_geom = AdditiveGeometry(*ans)
                    self.assertEqual(u == ans_geom, True)
        for i, ag1 in enumerate(additives):
            for j, t in enumerate(terms):
                with self.subTest(msg='term i={0}, j={1}'.format(i, j)):
                    u = ag1.union(t)
                    ans = [produce_term(a) for a in ag_union2[j][i]]
                    ans_geom = AdditiveGeometry(*ans)
                    self.assertEqual(u == ans_geom, True)

    def test_intersection(self):
        for i, ag1 in enumerate(additives):
            for j, ag2 in enumerate(additives):
                with self.subTest(msg='additive i={0}, j={1}'.format(i, j)):
                    u = ag1.intersection(ag2)
                    ans = [produce_term(a) for a in ag_intersection1[i][j]]
                    ans_geom = AdditiveGeometry(*ans)
                    self.assertEqual(u == ans_geom, True)
        for i, ag1 in enumerate(additives):
            for j, t in enumerate(terms):
                with self.subTest(msg='term i={0}, j={1}'.format(i, j)):
                    u = ag1.intersection(t)
                    ans = [produce_term(a) for a in ag_intersection2[j][i]]
                    ans_geom = AdditiveGeometry(*ans)
                    self.assertEqual(u == ans_geom, True)

    def test_complement(self):
        for i, ag in enumerate(additives):
            with self.subTest(msg='comlement for geom #{0}'.format(i)):
                c = ag.complement()
                ans = [produce_term(a) for a in ag_complement[i]]
                ans_geom = AdditiveGeometry(*ans)
                self.assertEqual(c == ans_geom, True)

    def test_box(self):
        box_data = ag_box_data
        for box_param, ans_array in box_data:
            box = Box(box_param['base'], box_param['ex'], box_param['ey'],
                      box_param['ez'])
            for i, t in enumerate(additives):
                with self.subTest(msg='result only case={0}'.format(i)):
                    r = t.test_box(box)
                    self.assertEqual(r, ans_array[i][0])
        for box_param, ans_array in box_data:
            box = Box(box_param['base'], box_param['ex'], box_param['ey'],
                      box_param['ez'])
            for i, t in enumerate(additives):
                with self.subTest(msg='result and simple case={0}'.format(i)):
                    r, s = t.test_box(box, return_simple=True)
                    self.assertEqual(r, ans_array[i][0])
                    ans = [produce_term(a) for a in ans_array[i][1][0]]
                    ans_geom = AdditiveGeometry(*ans)
                    self.assertEqual(s.pop() == ans_geom, True)

    def test_complexity(self):
        for i, ag in enumerate(additives):
            with self.subTest(i=i):
                c = ag.complexity()
                self.assertEqual(c, ag_complexity_ans[i])

    # @unittest.skip
    def test_bounding_box(self):
        base = [-10, -10, -10]
        dims = [30, 30, 30]
        gb = Box(base, [dims[0], 0, 0], [0, dims[1], 0], [0, 0, dims[2]])
        tol = 0.2
        for i, (ag, limits) in enumerate(zip(additives, ag_bounding_box)):
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

    # @unittest.skip
    def test_volume(self):
        for i, (box_data, vols) in enumerate(ag_volume):
            box = Box(box_data['base'], box_data['ex'], box_data['ey'], box_data['ez'])
            for j, ag in enumerate(additives):
                with self.subTest(msg='box {0}, geom {1}'.format(i, j)):
                    v = ag.volume(box, min_volume=1.e-2)
                    self.assertAlmostEqual(v, vols[j], delta=vols[j] * 0.001)

    def test_get_surfaces(self):
        s1 = Plane(EX, -5)
        s2 = Plane(EY, -5)
        s3 = Plane(EZ, -5)
        c = Cell([s1, 'C', s2, 'I', s3, 'U', s2, 'C', 'I', s3, 'I'])
        surfs = c.get_surfaces()
        self.assertEqual(len(surfs.difference({s2, s3})), 0)
        self.assertEqual(len({s2, s3}.difference(surfs)), 0)

    def test_test_point(self):
        points = ag_test_points
        for i, ans in enumerate(ag_test_point_ans):
            for j, p_ans in enumerate(ans):
                with self.subTest(msg='geom={0}, point={1}'.format(i, j)):
                    result = additives[i].test_point(points[j])
                    self.assertEqual(result, p_ans)
            with self.subTest(msg='geom={0}, total'.format(i)):
                result = list(additives[i].test_point(points))
                self.assertListEqual(ans, result)

    def test_from_polish_notation(self):
        for i, (pol_data, ans_data) in enumerate(ag_polish_data):
            pol_geom = []
            for x in pol_data:
                if isinstance(x, int):
                    pol_geom.append(surfaces[x])
                else:
                    pol_geom.append(x)
            with self.subTest(i=i):
                ans = [produce_term(a) for a in ans_data]
                ans_geom = AdditiveGeometry(*ans)
                ag = AdditiveGeometry.from_polish_notation(pol_geom)
                # print(str(ag))
                self.assertEqual(ag == ans_geom, True)


class TestCell(unittest.TestCase):
    def test_creation(self):
        for i, (pol_data, ans_data) in enumerate(ag_polish_data):
            pol_geom = []
            for x in pol_data:
                if isinstance(x, int):
                    pol_geom.append(surfaces[x])
                else:
                    pol_geom.append(x)
            with self.subTest(msg="polish #{0}".format(i)):
                ans = [produce_term(a) for a in ans_data]
                ans_geom = AdditiveGeometry(*ans)
                cell = Cell(pol_geom)
                # print(str(ag))
                self.assertEqual(cell, ans_geom)
        for i, ag in enumerate(additives):
            with self.subTest(msg="additive geom #{0}".format(i)):
                cell = Cell(ag)
                self.assertEqual(cell, ag)

    def test_intersection(self):
        for i, ag1 in enumerate(additives):
            c1 = Cell(ag1, **cell_kwargs)
            for j, ag2 in enumerate(additives):
                with self.subTest(msg='additive i={0}, j={1}'.format(i, j)):
                    c2 = Cell(ag2)
                    u = c1.intersection(c2)
                    ans = [produce_term(a) for a in ag_intersection1[i][j]]
                    ans_geom = AdditiveGeometry(*ans)
                    self.assertEqual(u, ans_geom)
                    self.assertDictEqual(u, c1)

    def test_union(self):
        for i, ag1 in enumerate(additives):
            c1 = Cell(ag1, **cell_kwargs)
            for j, ag2 in enumerate(additives):
                with self.subTest(msg='additive i={0}, j={1}'.format(i, j)):
                    c2 = Cell(ag2)
                    u = c1.union(c2)
                    ans = [produce_term(a) for a in ag_union1[i][j]]
                    ans_geom = AdditiveGeometry(*ans)
                    self.assertEqual(u, ans_geom)
                    self.assertDictEqual(u, c1)

    def test_populate(self):
        for i, ag_out in enumerate(additives):
            c_out = Cell(ag_out, name='Outer', U=4)
            cells = []
            geoms = []
            for j, ag in enumerate(additives):
                if j != i:
                    cells.append(Cell(ag, name=j))
                    geoms.append(ag.intersection(ag_out))
            universe = Universe(cells)
            with self.subTest(i=i):
                output = c_out.populate(universe)
                for j in range(len(cells)):
                    self.assertEqual(output[j], geoms[j])
                    if 'U' in c_out.keys():
                        self.assertEqual(c_out['U'], output[j]['U'])
                        output[j].pop('U')
                    self.assertDictEqual(output[j], cells[j])
                c_out['FILL'] = universe
                output = c_out.populate()
                for j in range(len(cells)):
                    self.assertEqual(output[j], geoms[j])
                    if 'U' in c_out.keys():
                        self.assertEqual(c_out['U'], output[j]['U'])
                        output[j].pop('U')
                    self.assertDictEqual(output[j], cells[j])

    @unittest.expectedFailure  # Surface equality should be developed.
    def test_transform(self):
        tr = Transformation(translation=[1, 3, -2])
        surfaces_tr = {k: s.transform(tr) for k, s in surfaces.items()}
        for i, (pol_data, ans_data) in enumerate(ag_polish_data):
            pol_geom = []
            for x in pol_data:
                if isinstance(x, int):
                    pol_geom.append(surfaces[x])
                else:
                    pol_geom.append(x)
            with self.subTest(msg="cell transform #{0}".format(i)):
                ans = [produce_term(a, surf=surfaces_tr) for a in ans_data]
                ans_geom = AdditiveGeometry(*ans)
                cell = Cell(pol_geom).transform(tr)
                print(str(cell), '->', str(ans_geom), '->', len(cell.terms), '->', len(ans_geom.terms))
                self.assertSetEqual(cell.terms, ans_geom.terms)

    # @unittest.skip
    def test_simplify(self):
        for i, ag in enumerate(additives):
            cell = Cell(ag)
            with self.subTest(i=i):
                s = cell.simplify(min_volume=0.1, box=Box([-10, -10, -10], [26, 0, 0], [0, 20, 0], [0, 0, 20]))
                # print(i, len(s))
                # for ss in s:
                #     print(str(ss))
                ans = [produce_term(a) for a in ag_simplify[i]]
                ans_geom = AdditiveGeometry(*ans)
                self.assertEqual(ans_geom == s, True)


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
