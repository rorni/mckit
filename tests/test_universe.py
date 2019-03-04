import pytest
import numpy as np
import tempfile

from mckit.material import Material, Element
from mckit.parser.mcnp_input_parser import read_mcnp
from mckit.transformation import Transformation
from mckit.geometry import Box
from mckit.universe import *
from mckit.body import Body, Shape
from mckit.surface import Sphere, Surface, create_surface
from mckit.material import Composition


@pytest.fixture(scope='module')
def universe():
    cases = {
        1: 'tests/universe_test_data/universe1.i',
        2: 'tests/universe_test_data/universe2.i',
        3: 'tests/universe_test_data/universe3.i',
        1002: 'tests/universe_test_data/universe1002.i',
        1012: 'tests/universe_test_data/universe1012.i',
        1022: 'tests/universe_test_data/universe1022.i'
    }

    def _universe(case):
        return read_mcnp(cases[case])

    return _universe


@pytest.mark.parametrize('case, points, answer', [
    (1, [[-2, 0, 4], [10, 10, -10], [0, 0, 0], [0, 0, -2.5], [0, 1, 0]],
     [1, 3, 0, 2, 0])
])
def test_points(universe, case, points, answer):
    u = universe(case)
    result = u.test_points(points)
    np.testing.assert_array_equal(result, answer)


_emp = Shape('R')


@pytest.mark.parametrize('kwargs', [
    {'name': 3, 'verbose_name': 'verbose', 'comment': 'comment'},
    {},
    {'name': 4, 'comment': ['1', '2']}
])
@pytest.mark.parametrize('cells', [
    [Body(_emp, name=2), Body(_emp, name=4)],
    [],
    [Body(_emp, name=5)]
])
def test_init(cells, kwargs):
    u = Universe(cells, **kwargs)
    assert len(u._cells) == len(cells)
    for c1, c2 in zip(u._cells, cells):
        assert c1.name() == c2.name()
        assert c1.shape == c2.shape
    assert u.name() == kwargs.get('name', 0)
    assert u.verbose_name() == kwargs.get('verbose_name', kwargs.get('name', 0))
    assert u._comment == kwargs.get('comment', None)


@pytest.mark.parametrize('case, cells, name_rule, new_name, new_surfs', [
    (1, Body(Shape('C', Sphere([0, 3, 0], 0.5, name=8)), name=2), 'keep', None, []),
    (1, Body(Shape('C', Sphere([0, 3, 0], 0.5, name=2)), name=8), 'keep', [8], None),
    (1, Body(Shape('C', Sphere([0, 3, 0], 0.5, name=8)), name=7), 'keep', [7], [Sphere([0, 3, 0], 0.5, name=8)]),
    (1, Body(Shape('C', Sphere([0, 3, 0], 0.5, name=8)), name=2), 'clash', [5], [Sphere([0, 3, 0], 0.5, name=8)]),
    (1, Body(Shape('C', Sphere([0, 3, 0], 0.5, name=2)), name=8), 'clash', [8], [Sphere([0, 3, 0], 0.5, name=6)]),
    (1, Body(Shape('C', Sphere([0, 3, 0], 0.5, name=8)), name=7), 'clash', [7], [Sphere([0, 3, 0], 0.5, name=8)]),
    (1, Body(Shape('C', Sphere([0, 3, 0], 0.5, name=8)), name=2), 'new', [5], [Sphere([0, 3, 0], 0.5, name=6)]),
    (1, Body(Shape('C', Sphere([0, 3, 0], 0.5, name=2)), name=8), 'new', [5], [Sphere([0, 3, 0], 0.5, name=6)]),
    (1, Body(Shape('C', Sphere([0, 3, 0], 0.5, name=8)), name=7), 'new', [5], [Sphere([0, 3, 0], 0.5, name=6)]),
    (1, Body(Shape('C', Sphere([0, 0, 0], 2, name=8)), name=2), 'new', [5], []),
    (1, Body(Shape('C', Sphere([0, 0, 0], 2, name=2)), name=8), 'keep', [8], []),
    (1, Body(Shape('C', Sphere([0, 0, 0], 2, name=8)), name=7), 'new', [5], []),
    (1, [Body(Shape('C', Sphere([0, 3, 0], 0.5, name=8)), name=8),
         Body(Shape('C', Sphere([0, 4, 0], 0.5, name=8)), name=9)], 'keep', [8, 9], None),
    (1, [Body(Shape('C', Sphere([0, 3, 0], 0.5, name=8)), name=9),
         Body(Shape('C', Sphere([0, 4, 0], 0.5, name=9)), name=9)], 'keep', None, [8, 9]),
    (1, [Body(Shape('C', Sphere([0, 3, 0], 0.5, name=2)), name=2),
         Body(Shape('C', Sphere([0, 4, 0], 0.5, name=2)), name=2)], 'clash',
     [5, 6], [Sphere([0, 3, 0], 0.5, name=6), Sphere([0, 4, 0], 0.5, name=7)]),
    (1, [Body(Shape('C', Sphere([0, 3, 0], 0.5, name=8)), name=9),
         Body(Shape('C', Sphere([0, 4, 0], 0.5, name=9)), name=9)], 'new',
     [5, 6], [Sphere([0, 3, 0], 0.5, name=6), Sphere([0, 4, 0], 0.5, name=7)]),

])
def test_add_cells(universe, case, cells, name_rule, new_name, new_surfs):
    u = universe(case)
    s_before = u.get_surfaces()
    if new_surfs is None or new_name is None:
        with pytest.raises(NameClashError):
            u.add_cells(cells, name_rule=name_rule)
    else:
        u.add_cells(cells, name_rule=name_rule)
        s_after = u.get_surfaces()
        acells = u._cells[-len(new_name):]
        if isinstance(cells, Body):
            cells = [cells]
        for acell, cell, name in zip(acells, cells, new_name):
            assert acell.shape == cell.shape
            assert acell is not cell
            assert acell.name() == name
            assert acell.options['U'] is u
        for k, v in s_before.items():
            assert v is s_after[k]
        assert len(s_after.keys()) - len(s_before.keys()) == len(new_surfs)
        for s in new_surfs:
            assert s_after[s.name()] == s


@pytest.mark.parametrize('case, recursive, answer_data', [
    (1, False, [(1, 'SO', [2]), (2, 'PZ', [3]), (3, 'PZ', [5]),
                (4, 'C/Z', [-2, 0, 1]), (5, 'SZ', [3.5, 10])]),
    (1, True, [(1, 'SO', [2]), (2, 'PZ', [3]), (3, 'PZ', [5]),
                (4, 'C/Z', [-2, 0, 1]), (5, 'SZ', [3.5, 10])]),
])
def test_get_surfaces(universe, case, recursive, answer_data):
    answer = {n: create_surface(k, *p, name=n) for n, k, p in answer_data}
    u = universe(case)
    surfaces = u.get_surfaces(inner=recursive)
    assert surfaces == answer


@pytest.mark.parametrize('case, answer', [
    (1, {1: Composition(atomic=[(Element('C-12', lib='31c'), 1)]), 2: Composition(atomic=[(Element('H1', lib='31c'), 2 / 3), (Element('O16', lib='31c'), 1 / 3)])}),
    (2, {}),
    (3, {4: Composition(atomic=[(Element('C-12', lib='31c'), 1)]), 3: Composition(atomic=[(Element('H1', lib='31c'), 2 / 3), (Element('O16', lib='31c'), 1 / 3)])}),
])
def test_get_compositions(universe, case, answer):
    u = universe(case)
    comps = u.get_compositions()
    assert comps == answer


@pytest.mark.parametrize('tr', [
    Transformation(translation=[-3, 2, 0.5]),
    Transformation(translation=[1, 2, 3]),
    Transformation(translation=[-4, 2, -3]),
    Transformation(translation=[3, 0, 9],
                   rotation=[30, 60, 90, 120, 30, 90, 90, 90, 0],
                   indegrees=True),
    Transformation(translation=[1, 4, -2],
                   rotation=[0, 90, 90, 90, 30, 60, 90, 120, 30],
                   indegrees=True),
    Transformation(translation=[-2, 5, 3],
                   rotation=[30, 90, 60, 90, 0, 90, 120, 90, 30],
                   indegrees=True)
])
@pytest.mark.parametrize('case, extent', [
    (1, [20, 10, 10])
])
def test_transform(universe, case, tr, extent):
    u = universe(case)
    points = np.random.random((50000, 3))
    points -= np.array([0.5, 0.5, 0.5])
    points *= np.array(extent)
    test_results = [c.shape.test_points(points) for c in u]

    tr_u = u.transform(tr)
    tr_points = tr.apply2point(points)
    tr_results = [c.shape.test_points(tr_points) for c in tr_u]
    assert len(u._cells) == len(tr_u._cells)
    for r, r_tr in zip(test_results, tr_results):
        np.testing.assert_array_equal(r, r_tr)


@pytest.mark.parametrize('case', [1])
def test_copy(universe, case):
    u = universe(case)
    uc = u.copy()
    assert uc is not u
    assert u.name() == uc.name()
    assert u.verbose_name() == uc.verbose_name()
    assert u._comment == uc._comment
    assert len(u._cells) == len(uc._cells)
    for c, cc in zip(u, uc):
        assert cc is not c
        assert c.name() == cc.name()
        assert c.shape == cc.shape
    surfs = u.get_surfaces()
    csurfs = uc.get_surfaces()
    assert surfs == csurfs
    for k, v in surfs.items():
        assert csurfs[k] is not v
        assert csurfs[k] == v


@pytest.mark.slow
@pytest.mark.parametrize('tol', [0.2, None])
@pytest.mark.parametrize('case, expected', [
    (1, [[-10, 10], [-10, 10], [-6.5, 13.5]]),
])
def test_bounding_box(universe, tol, case, expected):
    u = universe(case)
    base = [0, 0, 0]
    dims = [30, 30, 30]
    gb = Box(base, dims[0], dims[1], dims[2])
    if tol is not None:
        bb = u.bounding_box(box=gb, tol=tol)
    else:
        tol = 100.0
        bb = u.bounding_box()
    for j, (low, high) in enumerate(expected):
        bbdim = 0.5 * bb.dimensions[j]
        assert bb.center[j] - bbdim <= low
        assert bb.center[j] - bbdim >= low - tol
        assert bb.center[j] + bbdim >= high
        assert bb.center[j] + bbdim <= high + tol


@pytest.mark.parametrize('case, condition, inner, answer', [
    (1, get_cell_selector(1), False, [(Body, 1)]),
    (1, get_cell_selector([1, 3]), False, [(Body, 1), (Body, 3)]),
    (1, get_surface_selector(1), False, [(Surface, 1)]),
    (1, get_surface_selector([1, 4]), False, [(Surface, 1), (Surface, 4)]),
    (1, lambda c: [c] if c.material() else [], False, [(Body, 1), (Body, 2)]),
    (2, get_cell_selector(3), False, [(Body, 3)]),
    (2, get_cell_selector(11), False, []),
    (2, get_cell_selector(11), True, [(Body, 11)])
])
def test_select(universe, case, condition, inner, answer):
    u = universe(case)
    result = u.select(condition, inner=inner)
    assert len(result) == len(answer)
    for r, (cls, name) in zip(result, answer):
        assert isinstance(r, cls)
        assert r.name() == name


@pytest.mark.parametrize('case, answer', [
    (1, {0: {1, 2, 3, 4}}),
    (2, {0: {1, 2, 3}, 1: {10, 11, 12}, 2: {21, 22, 23, 24}})
])
def test_get_universes(universe, case, answer):
    u = universe(case)
    unvs = u.get_universes()
    assert len(unvs.keys()) == len(answer.keys())
    for k, v in unvs.items():
        names = {c.name() for c in v}
        assert names == answer[k]


@pytest.mark.parametrize('case, rename, stat', [
    (1, {}, {}),
    (2, {}, {}),
    (2, {2: {'start_cell': 2}}, {'cell': {2: [0, 2], 3: [0, 2]}}),
    (2, {2: {'start_surf': 2}}, {'surf': {2: [0, 2], 3: [0, 2], 4: [0, 2]}}),
    (2, {2: {'start_cell': 2, 'start_surf': 3}}, {'cell': {2: [0, 2], 3: [0, 2]}, 'surf': {3: [0, 2], 4: [0, 2]}}),
    (2, {2: {'start_cell': 2}, 1: {'start_cell': 1}}, {'cell': {1: [0, 1], 2: [0, 1, 2], 3: [0, 1, 2]}}),
    (3, {}, {})
])
def test_name_clashes(universe, case, rename, stat):
    u = universe(case)
    unvs = u.get_universes()
    for uname, ren_dict in rename.items():
        unvs[uname].rename(**ren_dict)
    s = u.name_clashes()
    assert s == stat


@pytest.mark.parametrize('case, condition, answer_case, box', [
    (2, {}, 1002, Box([0, 0, 0], 20, 20, 20)),
    (2, {'cell': 1}, 1012, Box([0, 0, 0], 20, 20, 20)),
    (2, {'cell': 2}, 1022, Box([0, 0, 0], 20, 20, 20)),
    (2, {'universe': 1}, 1022, Box([0, 0, 0], 20, 20, 20)),
    (2, {'universe': 2}, 1012, Box([0, 0, 0], 20, 20, 20)),
    (2, {'predicate': lambda c: 'transform' in c.options['FILL'].keys()}, 1022, Box([0, 0, 0], 20, 20, 20))
])
def test_apply_fill(universe, case, condition, answer_case, box):
    u = universe(case)
    ua = universe(answer_case)
    u.apply_fill(**condition)
    points = box.generate_random_points(1000000)
    test_f = u.test_points(points)
    test_a = ua.test_points(points)
    np.testing.assert_array_equal(test_f, test_a)


@pytest.mark.parametrize('case, start, answer', [
    (1, {'name': 4}, {'name': 4, 'cell': [1, 2, 3, 4], 'surface': [1, 2, 3, 4, 5]}),
    (1, {'start_cell': 6}, {'name': 0, 'cell': [6, 7, 8, 9], 'surface': [1, 2, 3, 4, 5]}),
    (1, {'start_surf': 7}, {'name': 0, 'cell': [1, 2, 3, 4], 'surface': [7, 8, 9, 10, 11]}),
    (1, {'name': 4, 'start_cell': 6, 'start_surf': 7}, {'name': 4, 'cell': [6, 7, 8, 9], 'surface': [7, 8, 9, 10, 11]})
])
def test_rename(universe, case, start, answer):
    u = universe(case)
    u.rename(**start)
    assert u.name() == answer['name']
    cnames = sorted(c.name() for c in u)
    snames = sorted(u.get_surfaces().keys())
    assert cnames == answer['cell']
    assert snames == answer['surface']


@pytest.mark.parametrize('case, complexities', [
    (1, {1: 1, 2: 3, 3: 5, 4: 1}),
    (3, {1: 1, 2: 3, 4: 5, 5: 1})
])
def test_simplify(universe, case, complexities):
    u = universe(case)
    u.simplify(min_volume=0.1)
    assert len(u._cells) == len(complexities.keys())
    for c in u:
        assert c.shape.complexity() == complexities[c.name()]


@pytest.mark.parametrize('case, box', [
    (1, Box([0, 0, 0], 20, 20, 20)),
    (2, Box([0, 0, 0], 20, 20, 20)),
    (3, Box([0, 0, 0], 20, 20, 20))
])
def test_save(universe, case, box):
    u = universe(case)
    out = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
    u.save(out.name)
    out.close()
    ur = read_mcnp(out.name)

    points = box.generate_random_points(100000)
    universes_orig = u.get_universes()
    universes_answ = ur.get_universes()
    assert universes_orig.keys() == universes_answ.keys()
    for k, univ in universes_orig.items():
        test_a = univ.test_points(points)
        test_f = universes_answ[k].test_points(points)
        np.testing.assert_array_equal(test_f, test_a)

#
# @pytest.mark.parametrize('case_no, u_name, materials', [
#     (0, 0, set()),
#     (0, 1, {Material(atomic=[(Element('C12', lib='21C'), 1)], density=2.7),
#             Material(weight=[(Element('H1', lib='21C'), 0.11191),
#                              (Element('O16', lib='50c'), 0.88809)], density=1.0)}),
#     (0, 2, {Material(atomic=[(Element('H1', lib='21C'), 2),
#                              (Element('C12', lib='21C'), 1)], density=0.9),
#             Material(atomic=[(Element('Fe56', lib='21c'), 1)], density=7.8),
#             Material(weight=[(Element('B10', lib='21c'), 1)], density=1.8)})
# ])
# def test_get_materials(universe, case_no, u_name, materials):
#     base_u = universe[case_no]
#     if u_name == 0:
#         u = base_u
#     else:
#         u = base_u.select_universe(u_name)
#     mats = u.get_materials()
#     assert mats == materials
#
#
#
# @pytest.mark.parametrize('case_no, cells, status, output', [
#     (0, None, False, {}),
#     (0, 1, False, {}),
#     (0, 2, False, {}),
#     (1, None, True, {
#         'cells': {21: {None}, 22: {None}, 23: [None], 24: [None]},
#         'surfaces': {20: [None], 21: [None], 22: [None], 23: [None], 24: [None]}
#     }),
#     (1, 1, True, {'cells': {21: [2], 22: [2], 23: [2], 24: [2]}}),
#     (1, 2, True, {
#         'cells': {21: [2], 22: [2], 23: [2], 24: [2]},
#         'surfaces': {20: [2], 21: [2], 22: [2], 23: [2], 24: [2]}
#     })
# ])
# def test_check_names(universe, case_no, cells, status, output):
#     u = universe[case_no]
#     u = u.fill(cells=cells)
#     s, r = u.name_clashes()
#     assert s == status
#     # assert r == output
#
