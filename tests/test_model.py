import unittest
import os

from tests.model_test_data.geometry_replace import surf_obj, cell_cases
from mckit.model import _get_universe_dependencies, read_mcnp, \
    _get_surface_indices, _get_contained_cells, _get_composition_indices, \
    _get_transformation_indices, Model, MCPrinter

from tests.model_test_data.model_data import *

cases = {}


def setUpModule():
    from mckit.mcnp_input_parser import mcnp_input_lexer, mcnp_input_parser
    for case in case_names:
        with open('tests/model_test_data/{0}.txt'.format(case)) as f:
            text = f.read()
            mcnp_input_lexer.begin('INITIAL')
            cases[case] = mcnp_input_parser.parse(text)


class TestModel(unittest.TestCase):
    def test_list_universes(self):
        for case in case_names:
            with self.subTest(msg='case: {0}'.format(case)):
                model = read_mcnp('tests/model_test_data/{0}.txt'.format(case))
                unames = model.list_universes()
                self.assertListEqual(unames, model_list_universes_ans[case])

    def test_extract_submodel(self):
        for case in case_names:
            model = read_mcnp('tests/model_test_data/{0}.txt'.format(case))
            for (u, flag), ans in extract_submodel_ans[case].items():
                msg = "case: {0}, universe: {1}, inserted: {2}".format(case, u, flag)
                with self.subTest(msg=msg):
                    submodel = model.extract_submodel(u, take_inserted=flag)
                    self.assertEqual(submodel.title, ans['title'])
                    self.assertDictEqual(submodel.cells, ans['cells'])
                    self.assertDictEqual(submodel.surfaces, ans['surfaces'])
                    self.assertDictEqual(submodel.data, ans['data'])

    def test_universe(self):
        pass

    def test_save(self):
        self.maxDiff = None
        for case in case_names:
            with self.subTest(msg='case: {0}'.format(case)):
                model = read_mcnp(
                    'tests/model_test_data/{0}.txt'.format(case))
                model.save('tests/model_test_data/{0}_p.txt'.format(case))
                with open('tests/model_test_data/{0}_ans.txt'.format(
                        case)) as f:
                    text_ans = f.read()
                with open('tests/model_test_data/{0}_p.txt'.format(
                        case)) as f:
                    text = f.read()
                self.assertEqual(text, text_ans)
                os.remove('tests/model_test_data/{0}_p.txt'.format(case))

    
class TestAuxiliaryFunctions(unittest.TestCase):
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

    def test_read_mcnp(self):
        for case in case_names:
            with self.subTest(msg='case: {0}'.format(case)):
                model = read_mcnp('tests/model_test_data/{0}.txt'.format(case))
                self.assertEqual(model.title, read_mcnp_ans[case]['title'])
                self.assertDictEqual(model.cells, read_mcnp_ans[case]['cells'])
                self.assertDictEqual(model.surfaces, read_mcnp_ans[case]['surfaces'])
                self.assertDictEqual(model.data, read_mcnp_ans[case]['data'])


class TestMCPrinter(unittest.TestCase):
    def test_cell_print(self):
        printer = MCPrinter()
        for case in case_names:
            with self.subTest(msg='case: {0}'.format(case)):
                model = read_mcnp('tests/model_test_data/{0}.txt'.format(case))
                ans = {k: printer.cell_print(v) for k, v in model.cells.items()}
                self.assertDictEqual(ans, cell_print_ans[case])

    def test_surface_print(self):
        printer = MCPrinter()
        for case in case_names:
            with self.subTest(msg='case: {0}'.format(case)):
                model = read_mcnp('tests/model_test_data/{0}.txt'.format(case))
                ans = {k: printer.surface_print(v) for k, v in model.surfaces.items()}
                self.assertDictEqual(ans, surface_print_ans[case])

    def test_material_print(self):
        printer = MCPrinter()
        for case in case_names:
            with self.subTest(msg='case: {0}'.format(case)):
                model = read_mcnp('tests/model_test_data/{0}.txt'.format(case))
                ans = {k: printer.material_print(v) for k, v in model.data['M'].items()}
                self.assertDictEqual(ans, material_print_ans[case])

    def test_transformation_print(self):
        printer = MCPrinter()
        self.maxDiff = None
        for case in case_names:
            with self.subTest(msg='case: {0}'.format(case)):
                model = read_mcnp('tests/model_test_data/{0}.txt'.format(case))
                ans = {k: printer.transformation_print(v) for k, v in model.data['TR'].items()}
                self.assertDictEqual(ans, transformation_print_ans[case])


if __name__ == '__main__':
    unittest.main()
