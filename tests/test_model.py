import unittest

from tests.model_test_data.geometry_replace import surf_obj, cell_cases
from mckit.model import _replace_geometry_names_by_objects, _get_material, \
    _create_material_objects, read_mcnp_model, Model

from tests.model_test_data.model_data import *


class TestModel(unittest.TestCase):
    def test_get_surface_indices(self):
        for case, data in get_surface_indices_ans.items():
            model = read_mcnp_model('tests/model_test_data/{0}.txt'.format(case))
            for cname, cell in model.cells.items():
                with self.subTest(i=cname):
                    surfs = Model.get_surface_indices(cell['geometry'])
                    self.assertEqual(surfs, data[cname])

    def test_get_universe_list(self):
        for case, data in get_universe_list_ans.items():
            model = read_mcnp_model('tests/model_test_data/{0}.txt'.format(case))
            self.assertEqual(data, model.get_universe_list())

    def test_get_contained_universes_ans(self):
        for case, data in contained_universes_ans.items():
            model = read_mcnp_model('tests/model_test_data/{0}.txt'.format(case))
            for uname, ucont in data.items():
                with self.subTest(i=uname):
                    self.assertEqual(model.get_contained_universes(uname), ucont)

    def test_get_universe_model(self):
        for case, data in get_universe_model_ans.items():
            model = read_mcnp_model('tests/model_test_data/{0}.txt'.format(case))
            for uname, part_ans in data.items():
                with self.subTest(i=uname):
                    part = model.get_universe_model(uname)
                    self.assertEqual(part.title, part_ans['title'])
                    self.assertEqual(part.cells.keys(), part_ans['cells'])
                    self.assertEqual(part.surfaces.keys(), part_ans['surfaces'])
                    self.assertEqual(part.data['M'].keys(), part_ans['material'])
                    self.assertEqual(part.data['TR'].keys(), part_ans['transform'])


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
        from tests.model_test_data.material_creation import compositions, \
            cells, mat_cell_ans
        materials = _create_material_objects(cells, compositions)
        with self.subTest(msg='Void cells'):
            for cell_name in mat_cell_ans[0]:
                self.assertNotIn('material', cells[cell_name].keys())
        for comp_name, mats in materials.items():
            with self.subTest(msg="composition = {0}".format(comp_name)):
                for i, mat in enumerate(mats):
                    for cn in mat_cell_ans[comp_name][i]:
                        self.assertIs(cells[cn]['material'], mat)


if __name__ == '__main__':
    unittest.main()
