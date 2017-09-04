import unittest

from tests.model_test_data.geometry_replace import surf_obj, cell_cases
from mckit.model import _replace_geometry_names_by_objects


class TestAuxiliaryFunctions(unittest.TestCase):
    def test_replace_geometry_names_by_objects(self):
        for i, cell_dict in enumerate(cell_cases):
            with self.subTest(i=i):
                _replace_geometry_names_by_objects(cell_dict, surf_obj)
                for name, cell in cell_dict.items():
                    msg = 'cell #{0}'.format(name)
                    self.assertEqual(cell['geometry'], cell['answer'], msg=msg)


if __name__ == '__main__':
    unittest.main()
