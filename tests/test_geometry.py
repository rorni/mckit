import unittest

from mckit.cell import GeometryTerm, AdditiveGeometry
from mckit.surface import create_surface
from mckit.fmesh import Box

from tests.cell_test_data import geometry_test_data

surfaces = {}
terms = []


def setUpModule():
    for name, (kind, params) in geometry_test_data.surface_data.items():
        surfaces[name] = create_surface(kind, *params, name=name)
    for tdata in geometry_test_data.terms:
        terms.append(produce_term(tdata))


def produce_term(data):
    p = {}
    for k, v in data.items():
        p[k] = {surfaces[i] for i in v}
    return GeometryTerm(**p)


class TestGeometryTerm(unittest.TestCase):
    def test_intersection(self):
        ans_data = geometry_test_data.term_intersection_ans
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
        for tdata in geometry_test_data.term_complement_ans:
            answers.append(produce_term(tdata))
        for i, t in enumerate(terms):
            with self.subTest(i=i):
                c = t.complement()
                self.assertSetEqual(c.positive, answers[i].positive)
                self.assertSetEqual(c.negative, answers[i].negative)

    def test_is_superset(self):
        ans = geometry_test_data.is_subset_ans
        for i, t1 in enumerate(terms):
            for j, t2 in enumerate(terms):
                with self.subTest(msg='i={0} j={1}'.format(i, j)):
                    dj = t1.is_superset(t2)
                    self.assertEqual(dj, ans[j][i])

    def test_is_subset(self):
        ans = geometry_test_data.is_subset_ans
        for i, t1 in enumerate(terms):
            for j, t2 in enumerate(terms):
                with self.subTest(msg='i={0} j={1}'.format(i, j)):
                    dj = t1.is_subset(t2)
                    self.assertEqual(dj, ans[i][j])

    def test_is_empty(self):
        for i, t in enumerate(terms):
            with self.subTest(i=i):
                self.assertEqual(t.is_empty(), geometry_test_data.is_empty_ans[i])

    def test_box(self):
        box_data = geometry_test_data.term_box_data
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
                    if r != -1:
                        ans = produce_term(ans_array[i][1][0])
                        self.assertSetEqual(s[0].positive, ans.positive)
                        self.assertSetEqual(s[0].negative, ans.negative)
                    else:
                        ans = [produce_term(a) for a in ans_array[i][1]]
                        self.assertEqual(len(s), len(ans))
                        pos_s = {p.positive.pop() for p in s if len(p.positive)}
                        pos_a = {p.positive.pop() for p in ans if len(p.positive)}
                        neg_s = {p.negative.pop() for p in s if len(p.negative)}
                        neg_a = {p.negative.pop() for p in ans if len(p.negative)}
                        self.assertSetEqual(pos_s, pos_a)
                        self.assertSetEqual(neg_s, neg_a)

    def test_complexity(self):
        for i, t in enumerate(terms):
            with self.subTest(i=i):
                c = t.complexity()
                self.assertEqual(c, geometry_test_data.term_complexity_ans[i])


class TestAdditiveGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.additives = []
        for adata in geometry_test_data.additives:
            params = [terms[i] for i in adata]
            cls.additives.append(AdditiveGeometry(*params))

    def test_creation(self):
        for i, ag in enumerate(self.additives):
            ans = [produce_term(a) for a in geometry_test_data.ag_create[i]]
            self.assertEqual(len(ans), len(ag.terms))
            for j, (aa, ao) in enumerate(zip(ans, ag.terms)):
                with self.subTest(msg='geom={0}, term={1}'.format(i, j)):
                    self.assertSetEqual(aa.positive, ao.positive)
                    self.assertSetEqual(aa.negative, ao.negative)

    def test_contains(self):
        for i, ag1 in enumerate(self.additives):
            for j, ag2 in enumerate(self.additives):
                with self.subTest(msg='i={0}, j={1}'.format(i, j)):
                    c = ag1.contains(ag2)
                    self.assertEqual(c, geometry_test_data.ag_contains[i][j])

    def test_equivalent(self):
        for i, ag1 in enumerate(self.additives):
            for j, ag2 in enumerate(self.additives):
                with self.subTest(msg='i={0}, j={1}'.format(i, j)):
                    c = ag1.equivalent(ag2)
                    self.assertEqual(c, geometry_test_data.ag_equiv[i][j])

    def test_union(self):
        for i, ag1 in enumerate(self.additives):
            for j, ag2 in enumerate(self.additives):
                with self.subTest(msg='additive i={0}, j={1}'.format(i, j)):
                    u = ag1.union(ag2)
                    ans = [produce_term(a) for a in
                           geometry_test_data.ag_union1[i][j]]
                    ans_geom = AdditiveGeometry(*ans)
                    self.assertEqual(u.equivalent(ans_geom), True)
        for i, ag1 in enumerate(self.additives):
            for j, t in enumerate(terms):
                with self.subTest(msg='term i={0}, j={1}'.format(i, j)):
                    u = ag1.union(t)
                    ans = [produce_term(a) for a in
                           geometry_test_data.ag_union2[j][i]]
                    ans_geom = AdditiveGeometry(*ans)
                    self.assertEqual(u.equivalent(ans_geom), True)

    def test_intersection(self):
        for i, ag1 in enumerate(self.additives):
            for j, ag2 in enumerate(self.additives):
                with self.subTest(msg='additive i={0}, j={1}'.format(i, j)):
                    u = ag1.intersection(ag2)
                    ans = [produce_term(a) for a in
                           geometry_test_data.ag_intersection1[i][j]]
                    ans_geom = AdditiveGeometry(*ans)
                    self.assertEqual(u.equivalent(ans_geom), True)
        for i, ag1 in enumerate(self.additives):
            for j, t in enumerate(terms):
                with self.subTest(msg='term i={0}, j={1}'.format(i, j)):
                    u = ag1.intersection(t)
                    ans = [produce_term(a) for a in
                           geometry_test_data.ag_intersection2[j][i]]
                    ans_geom = AdditiveGeometry(*ans)
                    self.assertEqual(u.equivalent(ans_geom), True)

    def test_complement(self):
        for i, ag in enumerate(self.additives):
            with self.subTest(msg='comlement for geom #{0}'.format(i)):
                c = ag.complement()
                ans = [produce_term(a) for a in geometry_test_data.ag_complement[i]]
                ans_geom = AdditiveGeometry(*ans)
                self.assertEqual(c.equivalent(ans_geom), True)

    def test_box(self):
        box_data = geometry_test_data.ag_box_data
        for box_param, ans_array in box_data:
            box = Box(box_param['base'], box_param['ex'], box_param['ey'],
                      box_param['ez'])
            for i, t in enumerate(self.additives):
                with self.subTest(msg='result only case={0}'.format(i)):
                    r = t.test_box(box)
                    self.assertEqual(r, ans_array[i][0])
        for box_param, ans_array in box_data:
            box = Box(box_param['base'], box_param['ex'], box_param['ey'],
                      box_param['ez'])
            for i, t in enumerate(self.additives):
                with self.subTest(msg='result and simple case={0}'.format(i)):
                    r, s = t.test_box(box, return_simple=True)
                    self.assertEqual(r, ans_array[i][0])
                    ans = [produce_term(a) for a in ans_array[i][1][0]]
                    ans_geom = AdditiveGeometry(*ans)
                    self.assertEqual(s[0].equivalent(ans_geom), True)

    def test_simplify(self):
        for i, ag in enumerate(self.additives):
            with self.subTest(i=i):
                s = ag.simplify(min_volume=0.1, box=Box([-10, -10, -10], [26, 0, 0], [0, 20, 0], [0, 0, 20]))
                # print(i, len(s))
                # for ss in s:
                #     print(str(ss))
                ans = [produce_term(a) for a in geometry_test_data.ag_simplify[i]]
                ans_geom = AdditiveGeometry(*ans)
                self.assertEqual(ans_geom.equivalent(s[0]), True)

    def test_complexity(self):
        for i, ag in enumerate(self.additives):
            with self.subTest(i=i):
                c = ag.complexity()
                self.assertEqual(c, geometry_test_data.ag_complexity_ans[i])

    # def test_merge_geometries(self):
    #     raise NotImplementedError

    def test_from_polish_notation(self):
        for i, (pol_data, ans_data) in enumerate(geometry_test_data.ag_polish_data):
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
                self.assertEqual(ag.equivalent(ans_geom), True)


if __name__ == '__main__':
    unittest.main()
