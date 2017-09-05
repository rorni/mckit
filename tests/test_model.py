import unittest

from tests.model_test_data.geometry_replace import surf_obj, cell_cases
from mckit.model import _replace_geometry_names_by_objects, _get_material, \
    _create_material_objects


class TestAuxiliaryFunctions(unittest.TestCase):
    def test_replace_geometry_names_by_objects(self):
        for i, cell_dict in enumerate(cell_cases):
            with self.subTest(i=i):
                _replace_geometry_names_by_objects(cell_dict, surf_obj)
                for name, cell in cell_dict.items():
                    msg = 'cell #{0}'.format(name)
                    self.assertEqual(cell['geometry'], cell['answer'], msg=msg)

    def test_get_material(self):
        from tests.model_test_data.material_creation import materials, densities
        for i, (den, ans) in enumerate(densities):
            with self.subTest(i=i):
                mat = _get_material(materials, den)
                self.assertIs(mat, ans)

    def test_create_material_objects(self):
        pass


if __name__ == '__main__':
    unittest.main()
