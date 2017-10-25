import unittest

from tests.model_test_data.geometry_replace import surf_obj, cell_cases
from mckit.model import _replace_geometry_names_by_objects, _get_material, \
    _create_material_objects, _get_universe_dependencies, read_mcnp_model, \
    _get_surface_indices, _get_contained_cells, _get_composition_indices, \
    _get_transformation_indices, Model

from tests.model_test_data.model_data import *

cases = {}


def setUpModule():
    from mckit.parser import lexer, parser
    for case in case_names:
        with open('tests/model_test_data/{0}.txt'.format(case)) as f:
            text = f.read()
            lexer.begin('INITIAL')
            cases[case] = parser.parse(text)


# class TestModel(unittest.TestCase):
#     def test_get_universe_list(self):
#         for case, data in get_universe_list_ans.items():
#             model = read_mcnp_model('tests/model_test_data/{0}.txt'.format(case))
#             self.assertEqual(data, model.get_universe_list())
#
#     def test_get_contained_universes_ans(self):
#         for case, data in contained_universes_ans.items():
#             model = read_mcnp_model('tests/model_test_data/{0}.txt'.format(case))
#             for uname, ucont in data.items():
#                 with self.subTest(i=uname):
#                     self.assertEqual(model.get_contained_universes(uname), ucont)
#
#     def test_get_universe_model(self):
#         for case, data in get_universe_model_ans.items():
#             model = read_mcnp_model('tests/model_test_data/{0}.txt'.format(case))
#             for uname, part_ans in data.items():
#                 with self.subTest(i=uname):
#                     part = model.get_universe_model(uname)
#                     self.assertEqual(part.title, part_ans['title'])
#                     self.assertEqual(part.cells.keys(), part_ans['cells'])
#                     self.assertEqual(part.surfaces.keys(), part_ans['surfaces'])
#                     self.assertEqual(part.data['M'].keys(), part_ans['material'])
#                     self.assertEqual(part.data['TR'].keys(), part_ans['transform'])


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

    def test_universe_dependencies(self):
        for case, data in cases.items():
            with self.subTest(msg=case):
                input = data[1]
                output = get_universe_dependencies_ans[case]
                ud = _get_universe_dependencies(input)
                self.assertDictEqual(ud, output)

    def test_get_contained_cells(self):
        for case, data in cases.items():
            cells = data[1]
            for (uname, flag), ans in get_contained_cells_ans[case].items():
                msg = "model:{0}, u={1}, take={2}".format(case, uname, flag)
                with self.subTest(msg=msg):
                    output = _get_contained_cells(cells, uname, take_inserted=flag)
                    cell_name_dict = {}
                    for cn, cv in output.items():
                        u = cv.get('U', 0)
                        if u not in cell_name_dict.keys():
                            cell_name_dict[u] = set()
                        cell_name_dict[u].add(cn)
                    self.assertDictEqual(cell_name_dict, ans)

    def test_get_surface_indices(self):
        for case, data in cases.items():
            cells = data[1]
            for (uname, flag), ans in get_contained_surfaces_ans[case].items():
                msg = "model:{0}, u={1}, take={2}".format(case, uname, flag)
                with self.subTest(msg=msg):
                    new_cells = _get_contained_cells(cells, uname,
                                                     take_inserted=flag)
                    output = _get_surface_indices(new_cells)
                    self.assertSetEqual(output, ans)

    def test_get_composition_indices(self):
        for case, data in cases.items():
            cells = data[1]
            for (uname, flag), ans in get_contained_compositions_ans[case].items():
                msg = "model:{0}, u={1}, take={2}".format(case, uname, flag)
                with self.subTest(msg=msg):
                    new_cells = _get_contained_cells(cells, uname,
                                                     take_inserted=flag)
                    output = _get_composition_indices(new_cells)
                    self.assertSetEqual(output, ans)

    def test_get_transformation_indices(self):
        for case, data in cases.items():
            cells = data[1]
            for (uname, flag), ans in get_contained_transformations_ans[case].items():
                msg = "model:{0}, u={1}, take={2}".format(case, uname, flag)
                with self.subTest(msg=msg):
                    new_cells = _get_contained_cells(cells, uname,
                                                     take_inserted=flag)
                    sur_ind = _get_surface_indices(new_cells)
                    surfaces = {si: data[2][si] for si in sur_ind}
                    output = _get_transformation_indices(new_cells, surfaces)
                    self.assertSetEqual(output, ans)


if __name__ == '__main__':
    unittest.main()
