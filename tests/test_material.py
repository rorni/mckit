import unittest

from mckit.material import Element, Material, Composition
from tests.material_test_data import element_test_data as el_data
from tests.material_test_data import composition_test_data as com_data
from tests.material_test_data import material_test_data as mat_data


class TestElement(unittest.TestCase):
    def test_creation(self):
        for i, (data, ans) in enumerate(zip(el_data.element_data, el_data.creation_cases)):
            name, options = data
            with self.subTest(i=i):
                elem = Element(name, **options)
                self.assertEqual(elem.charge, ans['charge'])
                self.assertEqual(elem.mass_number, ans['mass_number'])
                self.assertAlmostEqual(elem.molar_mass, ans['molar_mass'], delta=ans['molar_mass']*1.e-4)
                self.assertEqual(elem.lib, ans['lib'])
                self.assertEqual(elem.isomer, ans['isomer'])
                self.assertEqual(elem._comment, ans['comment'])

    def test_str(self):
        for i, (data, ans) in enumerate(zip(el_data.element_data, el_data.str_cases)):
            name, options = data
            with self.subTest(i=i):
                elem = Element(name, **options)
                self.assertEqual(str(elem), ans)
        for i, (data, ans) in enumerate(zip(el_data.element_data, el_data.str_mcnp_cases)):
            name, options = data
            with self.subTest(i=i):
                elem = Element(name, **options)
                self.assertEqual(elem.mcnp_repr(), ans)

    def test_eq(self):
        for i, (input1, ans) in enumerate(zip(el_data.element_data, el_data.equality)):
            name1, options1 = input1
            with self.subTest(i=i):
                elem1 = Element(name1, **options1)
                for j, input2 in enumerate(el_data.element_data):
                    name2, options2 = input2
                    elem2 = Element(name2, **options2)
                    self.assertEqual(elem1 == elem2, ans[j])

    def test_expand(self):
        for i, (data, ans) in enumerate(zip(el_data.element_data, el_data.expand_cases)):
            name, options = data
            with self.subTest(i=i):
                elem = Element(name, **options)
                expand_ans = {Element(name, **opt): v for name, opt, v in ans}
                expand = elem.expand()
                self.assertEqual(len(expand.keys()), len(expand_ans.keys()))
                for k, v in expand_ans.items():
                    if k not in expand.keys():
                        print(k, name)
                    self.assertAlmostEqual(v, expand[k], delta=v * 1.e-4)


class TestComposition(unittest.TestCase):
    def test_composition_creation(self):
        for i, (inp, ans) in enumerate(zip(com_data.comp_data, com_data.create_data)):
            comp = {Element(k): v for k, v in ans.items()}
            with self.subTest(i=i):
                c = Composition(**inp)
                self.assertEqual(len(comp.keys()), len(c._composition.keys()))
                for k, v in comp.items():
                    # TODO: Review accuracy. Check test data!
                    self.assertAlmostEqual(v, c._composition[k], delta=v*1.e-3)
                inp2 = {'atomic': [(Element(k), v) for k, v in inp.get('atomic', [])],
                        'weight': [(Element(k), v) for k, v in inp.get('weight', [])]}
                c = Composition(**inp2)
                self.assertEqual(len(comp.keys()), len(c._composition.keys()))
                for k, v in comp.items():
                    # TODO: Review accuracy. Check test data!
                    self.assertAlmostEqual(v, c._composition[k], delta=v*1.e-3)

    def test_creation_raise(self):
        for i, inp in enumerate(com_data.create_raise):
            with self.subTest(i=i):
                self.assertRaises(ValueError, Composition, **inp)

    def test_molar_mass(self):
        for i, (inp, ans) in enumerate(zip(com_data.comp_data, com_data.molar_mass_data)):
            with self.subTest(i=i):
                c = Composition(**inp)
                self.assertAlmostEqual(ans, c.molar_mass, delta=ans * 1.e-3)

    def test_contains(self):
        for i, inp in enumerate(com_data.comp_data):
            with self.subTest(i=i):
                c = Composition(**inp)
                for j, e in enumerate(com_data.contained_elements):
                    self.assertEqual(Element(e) in c, com_data.contains[j][i])
                    self.assertEqual(e in c, com_data.contains[j][i])

    def test_atomic(self):
        for i, inp in enumerate(com_data.comp_data):
            with self.subTest(i=i):
                c = Composition(**inp)
                for j, e in enumerate(com_data.contained_elements):
                    v = com_data.atomic[j][i]
                    if v == 0:
                        self.assertRaises(KeyError, c.get_atomic, e)
                        self.assertRaises(KeyError, c.get_atomic, Element(e))
                    else:
                        # TODO: verify data!
                        self.assertAlmostEqual(c.get_atomic(e), v, delta=v*1.e-3)
                        self.assertAlmostEqual(c.get_atomic(Element(e)), v, delta=v * 1.e-3)

    def test_weight(self):
        for i, inp in enumerate(com_data.comp_data):
            with self.subTest(i=i):
                c = Composition(**inp)
                for j, e in enumerate(com_data.contained_elements):
                    v = com_data.weight[j][i]
                    if v == 0:
                        self.assertRaises(KeyError, c.get_weight, e)
                        self.assertRaises(KeyError, c.get_weight, Element(e))
                    else:
                        # TODO: verify data!
                        self.assertAlmostEqual(c.get_weight(e), v, delta=v*1.e-3)
                        self.assertAlmostEqual(c.get_weight(Element(e)), v, delta=v * 1.e-3)

    def test_expand(self):
        for i, (inp, ans) in enumerate(zip(com_data.comp_data, com_data.expand)):
            with self.subTest(i=i):
                c = Composition(**inp)
                c.expand()
                self.assertEqual(len(ans.keys()), len(c._composition.keys()))
                for k, v in ans.items():
                    elem = Element(k)
                    self.assertAlmostEqual(v, c.get_atomic(elem), delta=v * 1.e-3)

    def test_natural(self):
        for i, (inp, ans) in enumerate(zip(com_data.comp_data, com_data.natural)):
            with self.subTest(i=i):
                c = Composition(**inp).natural(tolerance=1.e-3)
                if ans is None:
                    self.assertEqual(c, ans)
                    continue
                a = Composition(**ans)
                self.assertEqual(len(a._composition.keys()), len(c._composition.keys()))
                for k, v in a:
                    self.assertAlmostEqual(v, c.get_atomic(k), delta=v * 1.e-3)

    def test_equal(self):
        for i, (inp, ans) in enumerate(zip(com_data.comp_data, com_data.equal_data)):
            c = Composition(**inp)
            for j, (inp2, a) in enumerate(zip(com_data.comp_data, ans)):
                with self.subTest(msg="c1={0}, c2={1}".format(i, j)):
                    c2 = Composition(**inp2)
                    self.assertEqual(c == c2, bool(a))

    def test_get_option(self):
        for i, (inp, ans) in enumerate(zip(com_data.comp_data, com_data.get_option)):
            c = Composition(**inp)
            with self.subTest(i=i):
                for k, v in ans.items():
                    self.assertEqual(c[k], v)

    def test_mixture(self):
        for i, (input, ans_index) in enumerate(com_data.mixture):
            with self.subTest(i=i):
                ans = Composition(**com_data.comp_data[ans_index])
                compositions = []
                for ci, f in input:
                    compositions.append((Composition(**com_data.comp_data[ci]), f))
                m = Composition.mixture(*compositions)
                self.assertEqual(m, ans)


class TestMaterial(unittest.TestCase):
    def test_creation_failure(self):
        for i, case in enumerate(mat_data.failure_cases):
            with self.subTest(i=i):
                if 'composition' in case.keys():
                    case['composition'] = Composition(**case['composition'])
                self.assertRaises(ValueError, Material, **case)

    def test_creation_cases(self):
        for i, case_data in enumerate(mat_data.creation_cases):
            with self.subTest(i=i):
                case = case_data.copy()
                atomic = case.pop('atomic', tuple())
                weight = case.pop('weight', tuple())
                comp = Composition(atomic=atomic, weight=weight)
                mat1 = Material(atomic=atomic, weight=weight, **case)
                mat2 = Material(composition=comp, **case)
                self.assertEqual(mat1.composition, comp)
                self.assertEqual(mat2.composition, comp)
                if 'density' in case.keys():
                    d = case['density']
                    self.assertAlmostEqual(mat1.density, d, delta=d * 1.e-5)
                    self.assertAlmostEqual(mat2.density, d, delta=d * 1.e-5)
                elif 'concentration' in case.keys():
                    c = case['concentration']
                    self.assertAlmostEqual(mat1.concentration, c, delta=c*1.e-5)
                    self.assertAlmostEqual(mat2.concentration, c, delta=c*1.e-5)

    def test_equality(self):
        mats = [Material(**case) for case in mat_data.creation_cases]
        for i, m1 in enumerate(mats):
            for j, m2 in enumerate(mats):
                with self.subTest(msg='i={0}, j={1}'.format(i, j)):
                    self.assertEqual(m1 == m2, bool(mat_data.equal_cases[i][j]))

    def test_factor(self):
        mat = Material(**mat_data.creation_cases[0])
        for i, case in enumerate(mat_data.correct_cases):
            with self.subTest(i=i):
                new_mat = mat.correct(**case)
                self.assertEqual(mat.composition, new_mat.composition)
                if 'factor' in case.keys():
                    self.assertAlmostEqual(new_mat.density, mat.density * case['factor'])
                else:
                    old_mass = case['old_vol'] * mat.density
                    new_mass = case['new_vol'] * new_mat.density
                    self.assertAlmostEqual(old_mass, new_mass, delta= 0.5e-3 * (old_mass + new_mass))

    def test_mixture(self):
        for i, ((materials, ftype), ans) in enumerate(zip(mat_data.mixture_cases, mat_data.mixture_answers)):
            with self.subTest(i=i):
                ans_mat = Material(**ans)
                mats = [(Material(**kws), frac) for kws, frac in materials]
                mix = Material.mixture(*mats, fraction_type=ftype)
                self.assertEqual(ans_mat, mix)


if __name__ == '__main__':
    unittest.main()
