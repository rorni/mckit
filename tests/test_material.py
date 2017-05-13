import unittest

from mckit.material import molar_mass


class TestMolarMass(unittest.TestCase):
    def test_molar_mass(self):
        for i, (i_name, mol) in enumerate(molar_mass_correct_cases):
            with self.subTest(i=i):
                m = molar_mass(i_name)
                self.assertAlmostEqual(m, mol, delta=mol * 1.e-5)

    def test_molar_mass_exception(self):
        for i, i_name in enumerate(molar_mass_incorrect_cases):
            with self.subTest(i=i):
                self.assertRaises(ValueError, molar_mass, i_name)


molar_mass_correct_cases = [
    (1001, 1.007825), ('H-1', 1.007825), (20042, 41.958618),
    ('Ca-42', 41.958618), ('Ca', 40.078), (20000, 40.078),
    (4000, 9.012182), ('be-0', 9.012182), ('be', 9.012182),
    (74000, 183.841), ('w', 183.841), ('U-233', 233.00)
]

molar_mass_incorrect_cases = [
    'XR', 'XR-40', 'Q'
]

if __name__ == '__main__':
    unittest.main()
