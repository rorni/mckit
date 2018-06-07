import unittest

from mckit.material import Element, Material, merge_materials, mixture


class TestElement(unittest.TestCase):
    def test_split_name(self):
        for i, (i_name, q, a) in enumerate(isotope_name_cases):
            with self.subTest(i=i):
                Q, A = Element._split_name(i_name)
                self.assertEqual(Q, q)
                self.assertEqual(A, a)

    def test_element_creation(self):
        for i, (i_name, q, a) in enumerate(element_creation_cases):
            with self.subTest(i=i):
                elem = Element(i_name)
                Q = elem.charge()
                A = elem.mass_number()
                self.assertEqual(Q, q)
                self.assertEqual(A, a)

    def test_molar_mass(self):
        for i, (i_name, mol) in enumerate(molar_mass_cases):
            with self.subTest(i=i):
                m = Element(i_name).molar_mass()
                self.assertAlmostEqual(m, mol, delta=mol * 1.e-5)

    def test_equality(self):
        N = len(element_eq_cases)
        for i, name1 in enumerate(element_eq_cases):
            el1 = Element(name1)
            for j, name2 in enumerate(element_eq_cases):
                el2 = Element(name2)
                with self.subTest(i=i * N + j):
                    self.assertTrue((el1 == el2) == element_eq_matrix[i][j])

    def test_expand(self):
        for i, (i_name, exp_dict) in enumerate(element_expand_cases):
            with self.subTest(i=i):
                el = Element(i_name)
                exp1 = el.expand()
                self.assertEqual(len(exp1.items()), len(exp_dict.items()))
                for k, v in exp1.items():
                    self.assertAlmostEqual(v, exp_dict[k], delta=1.e-5)


class TestMaterial(unittest.TestCase):
    def test_material_creation(self):
        for i, (kwargs, n, mu, rho) in enumerate(material_creation_cases):
            with self.subTest(i=i):
                mat = Material(**kwargs)
                self.assertAlmostEqual(mat.molar_mass(), mu, delta=mu * 1.e-4)
                self.assertAlmostEqual(mat.concentration(), n, delta=n * 1.e-4)
                self.assertAlmostEqual(mat.density(), rho, delta=rho * 1.e-4)

    def test_material_creation_failure(self):
        for i, kwargs in enumerate(material_creation_failed_cases):
            with self.subTest(i=i):
                self.assertRaises(ValueError, Material, **kwargs)

    def test_material_correct(self):
        fact = [2, 0.5]
        vol1 = 100
        for j, f in enumerate(fact):
            vol2 = vol1 * f
            for i, (kwargs, n, mu, rho) in enumerate(material_creation_cases):
                with self.subTest(i=j*2 + i):
                    mat = Material(**kwargs)
                    mc = mat.correct(old_vol=vol1, new_vol=vol2)
                    self.assertEqual(len(mat._composition._composition.keys()),
                                     len(mc._composition._composition.keys()))
                    self.assertAlmostEqual(mat._n, mc._n * f, delta=n * 1.e-5)
                    for k, v in mat._composition._composition.items():
                        self.assertAlmostEqual(v, mc._composition._composition[k] * f,
                                               delta=v * 1.e-5)

    def test_material_expand(self):
        for i, (arg, ans) in enumerate(material_expand_cases):
            with self.subTest(i=i):
                mat = Material(atomic=arg)
                exp = mat.expand()
                for k, v in ans:
                    elem = Element(k)
                    self.assertAlmostEqual(exp._composition._composition[elem], v,
                                           delta=v * 1.e-5)

    def test_material_merge(self):
        for i, (m1, v1, m2, v2, ans) in enumerate(material_merge_cases):
            with self.subTest(i=i):
                mat = merge_materials(Material(atomic=m1), v1,
                                      Material(atomic=m2), v2)
                mat_ans = Material(atomic=ans)
                self.assertEqual(len(mat._composition._composition.keys()),
                                 len(mat_ans._composition._composition.keys()))
                for k, v in mat._composition.items():
                    self.assertAlmostEqual(v, mat_ans._composition._composition[k],
                                           delta=v * 1.e-5)

    def test_material_eq(self):
        materials = [Material(**eq_cases) for eq_cases in material_eq_cases]
        N = len(material_eq_cases)
        for i in range(N):
            for j in range(N):
                with self.subTest(msg="i={0}, j={1}".format(i, j)):
                    result = (materials[i] == materials[j])
                    self.assertTrue(result == material_eq_matrix[i][j])

    def test_mixture(self):
        for i, case in enumerate(material_mix_cases):
            components = [Material(**c) for c in case['components']]
            for ftype, fvalues in case['fractions'].items():
                with self.subTest(msg='{0} - {1}'.format(i, ftype)):
                    new_mat = mixture(*zip(components, fvalues), fraction_type=ftype)
                    if 'molar_mass' in case.keys():
                        self.assertAlmostEqual(new_mat.molar_mass(), case['molar_mass'], delta=case['molar_mass'] * 1.e-3)
                    if 'density' in case.keys():
                        self.assertAlmostEqual(new_mat.density(), case['density'], delta=case['density'] * 5.e-3)
                    if 'concentration' in case.keys():
                        self.assertAlmostEqual(new_mat.concentration(), case['concentration'], delta=case['concentration'] * 5.e-3)


material_eq_cases = [
    {'atomic': [(1000, 2), (8000, 1)], 'density': 0.9982},
    {'atomic': [(1000, 2), (8000, 1)], 'density': 0.99825},
    {'atomic': [(1000, 2), (8000, 1)], 'density': 0.99815},
    {'atomic': [(1000, 2), (8000, 1)], 'density': 0.9980},
    {'atomic': [(1000, 4), (8000, 2)], 'density': 0.9982},
    {'atomic': [(1001, 2), (8016, 1)], 'density': 0.9982},
    {'atomic': [(1001, 2.001), (8016, 0.999)], 'concentration': 1.0e+22},
    {'wgt': [('N', 75.5), ('O', 23.15), ('Ar', 1.292)], 'density': 1.292e-3},
    {'wgt': [('N', 75.501), ('O', 23.1499), ('Ar', 1.29201)], 'density': 1.292e-3},
    {'wgt': [('N', 75.6), ('O', 23.15), ('Ar', 1.292)], 'density': 1.292e-3},
    {'wgt': [('N', 75.5), ('O', 23.15), ('Ar', 1.292)], 'concentration': 1.0e+22}
]

material_eq_matrix = [
    [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
]


material_mix_cases = [
    {
        'components': ({'atomic': [('N', 1)], 'density': 1.2506e-3},
                       {'atomic': [('O', 1)], 'density': 1.42897e-3},
                       {'atomic': [('Ar', 1)], 'density': 1.784e-3},
                       {'atomic': [('C', 1), ('O', 2)], 'density': 1.9768e-3},
                       {'atomic': [('Ne', 1)], 'density': 0.9002e-3},
                       {'atomic': [('Kr', 1)], 'density': 3.749e-3}),
        'fractions': {
            'weight': [0.755, 0.2315, 0.01292, 0.00046, 0.000014, 0.00003],
            'volume': [0.78084, 0.209476, 0.00934, 0.000314, 0.00001818, 0.00000114]
        },
        'density': 1.2929e-3
    },
    {
        'components': ({'atomic': [('H', 2), ('O', 1)], 'density': 1.0},),
        'fractions': {
            'weight': [0.5],
            'volume': [0.5],
            'atomic': [0.5]
        },
        'density': 0.5
    },
    {
        'components': ({'atomic': [('H', 2), ('O', 1)], 'density': 1.0},),
        'fractions': {
            'weight': [0.7],
            'volume': [0.7],
            'atomic': [0.7]
        },
        'density': 0.7
    },
    {
        'components': ({'atomic': [('H', 2), ('O', 1)], 'density': 1.0},),
        'fractions': {
            'weight': [0.3],
            'volume': [0.3],
            'atomic': [0.3]
        },
        'density': 0.3
    },
    {
        'components': ({'atomic': [('H', 2), ('O', 1)], 'density': 1.0},
                       {'atomic': [('H', 2), ('O', 1)], 'density': 1.0}),
        'fractions': {
            'weight': [0.3, 0.7],
            'volume': [0.7, 0.3],
            'atomic': [0.1, 0.9]
        },
        'density': 1.0
    }
]


material_creation_cases = [
    ({'atomic': [('NI58', 0.680769), ('NI60', 0.262231), ('NI61', 0.011399), ('NI62', 0.036345), ('NI64',  0.009256)],
      'density': 8.902}, 9.133753e+22, 58.6934, 8.902),
    ({'atomic': [('NI58', 68.0769), ('NI60', 26.2231), ('NI61', 1.1399), ('NI62', 3.6345), ('NI64',  0.9256)],
      'density': 8.902}, 9.133753e+22, 58.6934, 8.902),
    ({'atomic': [('NI58', 0.680769), ('NI60', 0.262231), ('NI61', 0.011399), ('NI62', 0.036345), ('NI64',  0.009256)],
      'concentration': 9.133753e+22}, 9.133753e+22, 58.6934, 8.902),
    ({'atomic': [('NI58', 68.0769), ('NI60', 26.2231), ('NI61', 1.1399), ('NI62', 3.6345), ('NI64',  0.9256)],
      'concentration': 9.133753e+22}, 9.133753e+22, 58.6934, 8.902),
    ({'atomic': [('NI58', 0.680769), ('NI60', 0.262231), ('NI61', 0.011399), ('NI62', 0.036345), ('NI64', 0.009256)]},
      1.0, 58.6934, 9.7462675e-23),
    ({'atomic': [('NI58', 68.0769), ('NI60', 26.2231), ('NI61', 1.1399), ('NI62', 3.6345), ('NI64', 0.9256)]},
      100.0, 58.6934, 9.7462675e-21),
    ({'wgt': [('N', 0.755), ('O', 0.232), ('AR', 0.013)], 'density': 1.2929e-3}, 5.351034567e+19, 14.551, 1.2929e-3),
    ({'wgt': [('N', 75.5), ('O', 23.2), ('AR', 1.3)], 'density': 1.2929e-3}, 5.351034567e+19, 14.551, 1.2929e-3),
    ({'wgt': [('N', 0.755), ('O', 0.232), ('AR', 0.013)], 'concentration': 5.351034567e+19}, 5.351034567e+19, 14.551, 1.2929e-3),
    ({'wgt': [('N', 75.5), ('O', 23.2), ('AR', 1.3)], 'concentration': 5.351034567e+19}, 5.351034567e+19, 14.551, 1.2929e-3),
    ({'wgt': [('N', 0.5), ('O', 0.232), ('AR', 0.013), ('N', 0.255)], 'density': 1.2929e-3},
     5.351034567e+19, 14.551, 1.2929e-3),
    ({'wgt': [('N', 0.5), ('O', 0.132), ('AR', 0.013), ('N', 0.255), ('O', 0.1)], 'density': 1.2929e-3},
     5.351034567e+19, 14.551, 1.2929e-3),
    ({'wgt': [('N', 0.2), ('O', 0.132), ('AR', 0.013), ('N', 0.255), ('O', 0.1), ('N', 0.3)],
     'density': 1.2929e-3}, 5.351034567e+19, 14.551, 1.2929e-3),
    ({'atomic': [(28058, 0.680769), (28060, 0.262231), (28061, 0.011399), (28062, 0.036345), (28064, 0.009256)]},
     1.0, 58.6934, 9.7462675e-23),
    ({'atomic': [('H', 2 / 3), ('O', 1 / 3)], 'density': 0.9982}, 1.0010337342849073e+23, 18.01528 / 3, 0.9982)
]

material_creation_failed_cases = [
    {'atomic': [('NI58', 0.680769), ('NI60', 0.262231), ('NI61', 0.011399), ('NI62', 0.036345), ('NI64',  0.009256)],
     'density': 8.902, 'concentration': 1.e+23},
    {'wgt': [('N', 0.755), ('O', 0.232), ('AR', 0.013)], 'density': 1.2929e-3, 'concentration': 1.e+23},
    {'density': 8.902, 'concentration': 1.e+23},
    {'density': 8.902}, {'concentration': 1.e+23},
    {'wgt': [('N', 0.755), ('O', 0.232), ('AR', 0.013)]}
]

material_expand_cases = [
    ([('NI', 1.0)], [('NI58', 0.680769), ('NI60', 0.262231), ('NI61', 0.011399), ('NI62', 0.036345), ('NI64', 0.009256)]),
    ([('NI58', 0.680769), ('NI60', 0.262231), ('NI61', 0.011399), ('NI62', 0.036345), ('NI64', 0.009256)], [('NI58', 0.680769), ('NI60', 0.262231), ('NI61', 0.011399), ('NI62', 0.036345), ('NI64', 0.009256)]),
    ([('NI58', 0.380769), ('NI60', 0.162231), ('NI61', 0.011399), ('NI58', 0.300000), ('NI62', 0.036345), ('NI64', 0.009256), ('NI60', 0.1)], [('NI58', 0.680769), ('NI60', 0.262231), ('NI61', 0.011399), ('NI62', 0.036345), ('NI64', 0.009256)]),
    ([('B', 2), ('C', 3)], [('B10', 0.398), ('B11', 1.602), ('C12', 2.9679), ('C13', 0.0321)])
]

material_merge_cases = [
    ([('NI', 1.0)], 2, [('H', 2)], 2, [('NI', 0.5), ('H', 1.0)]),
    ([('NI', 1.0)], 2, [('H', 2), ('NI', 2)], 2, [('NI', 1.5), ('H', 1.0)]),
]

isotope_name_cases = [
    ('1001', '1', '001'), ('13027', '13', '027'), ('92235', '92', '235'),
    ('H2', 'H', '2'), ('H-3', 'H', '3'), ('I', 'I', '0'), ('HE3', 'HE', '3'),
    ('HE-4', 'HE', '4'), ('CA', 'CA', '0'), ('U235', 'U', '235'),
    ('U-238', 'U', '238'), ('AR-40', 'AR', '40'), ('CL35', 'CL', '35'),
    ('NA0', 'NA', '0'), ('AG-0', 'AG', '0'), ('W-0', 'W', '0'),
    ('F0', 'F', '0')
]

element_creation_cases = [
    ('1001', 1, 1), ('13027', 13, 27), ('92235', 92, 235), ('H2', 1, 2),
    ('H-3', 1, 3), ('I', 53, 0), ('HE3', 2, 3), ('HE-4', 2, 4),
    ('CA', 20, 0), ('U235', 92, 235), ('U-238', 92, 238), ('AR-40', 18, 40),
    ('CL35', 17, 35), ('NA0', 11, 0), ('AG-0', 47, 0), ('W-0', 74, 0),
    ('F0', 9, 0), (1001, 1, 1), (13027, 13, 27), (92235, 92, 235)
]

element_eq_cases = [
    '1001', 'H1', 'CA', '20000', '20040', 'U-238', '92238', '92235'
]

element_eq_matrix = [
    [1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]
]

element_expand_cases = [
    ('Be9', {Element('Be9'): 1.0}),
    ('F', {Element('F19'): 1.0}),
    ('Ne', {Element('Ne20'): 0.9048, Element('Ne21'): 0.0027,
            Element('Ne22'): 0.0925})
]

molar_mass_cases = [
    ('1001', 1.007825), ('H-1', 1.007825), ('20042', 41.958618),
    ('Ca-42', 41.958618), ('Ca', 40.078), ('20000', 40.078),
    ('4000', 9.012182), ('be-0', 9.012182), ('be', 9.012182),
    ('74000', 183.841), ('w', 183.841), ('U-233', 233.00)
]

molar_mass_incorrect_cases = [
    'XR', 'XR-40', 'Q'
]

if __name__ == '__main__':
    unittest.main()
