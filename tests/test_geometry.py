import unittest

from mckit.cell import GeometryTerm, AdditiveGeometry
from mckit.surface import create_surface
from mckit.fmesh import Box

from tests.cell_test_data import geometry_test_data


class TestGeometryTerm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.surfaces = {}
        cls.terms = []
        for name, (kind, params) in geometry_test_data.surface_data.items():
            cls.surfaces[name] = create_surface(kind, *params, name=name)
        for tdata in geometry_test_data.terms:
            cls.terms.append(cls.produce_term(tdata))

    @classmethod
    def produce_term(cls, data):
        p = {}
        for k, v in data.items():
            p[k] = {cls.surfaces[i] for i in v}
        return GeometryTerm(**p)

    def test_intersection(self):
        ans_data = geometry_test_data.intersection_ans
        for i, t1 in enumerate(self.terms):
            for j, t2 in enumerate(self.terms[:i] + self.terms[i+1:]):
                with self.subTest(msg='i={0} j={1}'.format(i, j)):
                    ti = t1.intersection(t2)
                    ans = {}
                    for k, v in ans_data[i][j].items():
                        ans[k] = {self.surfaces[i] for i in v}
                    ta = GeometryTerm(**ans)
                    self.assertSetEqual(ti.positive, ta.positive)
                    self.assertSetEqual(ti.negative, ta.negative)

    def test_complement(self):
        answers = []
        for tdata in geometry_test_data.complement_ans:
            answers.append(self.produce_term(tdata))
        for i, t in enumerate(self.terms):
            with self.subTest(i=i):
                c = t.complement()
                self.assertSetEqual(c.positive, answers[i].positive)
                self.assertSetEqual(c.negative, answers[i].negative)

    def test_is_superset(self):
        ans = geometry_test_data.is_subset_ans
        for i, t1 in enumerate(self.terms):
            for j, t2 in enumerate(self.terms):
                with self.subTest(msg='i={0} j={1}'.format(i, j)):
                    dj = t1.is_superset(t2)
                    self.assertEqual(dj, ans[j][i])

    def test_is_subset(self):
        ans = geometry_test_data.is_subset_ans
        for i, t1 in enumerate(self.terms):
            for j, t2 in enumerate(self.terms):
                with self.subTest(msg='i={0} j={1}'.format(i, j)):
                    dj = t1.is_subset(t2)
                    self.assertEqual(dj, ans[i][j])

    def test_is_empty(self):
        for i, t in enumerate(self.terms):
            with self.subTest(i=i):
                self.assertEqual(t.is_empty(), geometry_test_data.is_empty_ans[i])

    def test_box(self):
        box_data = geometry_test_data.box_data
        for box_param, ans_array in box_data:
            box = Box(box_param['base'], box_param['ex'], box_param['ey'],
                      box_param['ez'])
            for i, t in enumerate(self.terms):
                with self.subTest(msg='result only case={0}'.format(i)):
                    r = t.test_box(box)
                    print(ans_array[i][0])
                    self.assertEqual(r, ans_array[i][0])
        for box_param, ans_array in box_data:
            box = Box(box_param['base'], box_param['ex'], box_param['ey'],
                      box_param['ez'])
            for i, t in enumerate(self.terms):
                with self.subTest(msg='result and simple case={0}'.format(i)):
                    r, s = t.test_box(box, return_simple=True)
                    self.assertEqual(r, ans_array[i][0])
                    if r != -1:
                        ans = self.produce_term(ans_array[i][1][0])
                        self.assertSetEqual(s[0].positive, ans.positive)
                        self.assertSetEqual(s[0].negative, ans.negative)
                    else:
                        ans = [self.produce_term(a) for a in ans_array[i][1]]
                        self.assertEqual(len(s), len(ans))
                        pos_s = {p.positive.pop() for p in s if len(p.positive)}
                        pos_a = {p.positive.pop() for p in ans  if len(p.positive)}
                        neg_s = {p.negative.pop() for p in s  if len(p.negative)}
                        neg_a = {p.negative.pop() for p in ans  if len(p.negative)}
                        self.assertSetEqual(pos_s, pos_a)
                        self.assertSetEqual(neg_s, neg_a)

    def test_complexity(self):
        for i, t in enumerate(self.terms):
            with self.subTest(i=i):
                c = t.complexity()
                self.assertEqual(c, geometry_test_data.complexity_ans[i])


class TestAdditiveGeometry(unittest.TestCase):
    def test_creation(self):
        raise NotImplementedError

    def test_union(self):
        raise NotImplementedError

    def test_intersection(self):
        raise NotImplementedError

    def test_complement(self):
        raise NotImplementedError

    def test_box(self):
        raise NotImplementedError

    def test_simplify(self):
        raise NotImplementedError

    def test_complexity(self):
        raise NotImplementedError

    def test_merge_geometries(self):
        raise NotImplementedError

    def test_from_polish_notation(self):
        raise NotImplementedError


if __name__ == '__main__':
    unittest.main()
