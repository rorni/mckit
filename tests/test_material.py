import unittest

from mckit.material import Element


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
    ('F0', 9, 0)
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
