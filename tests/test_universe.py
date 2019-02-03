import pytest
import numpy as np

from mckit.material import Material, Element
from mckit.parser.mcnp_input_parser import read_mcnp
from mckit.transformation import Transformation
from mckit.geometry import Box
from mckit.universe import *
from mckit.body import Body, Shape
from mckit.surface import Sphere, Surface, create_surface


@pytest.fixture(scope='module')
def universe():
    cases = {
        1: 'tests/universe_test_data/universe1.i',
        2: 'tests/universe_test_data/universe2.i'
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


_emp = Shape('E')


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


@pytest.mark.parametrize('case, cell, name_rule, new_name, new_surfs', [
    (1, Body(Shape('C', Sphere([0, 3, 0], 0.5, name=8)), name=2), 'keep', None, []),
    (1, Body(Shape('C', Sphere([0, 3, 0], 0.5, name=2)), name=8), 'keep', 8, None),
    (1, Body(Shape('C', Sphere([0, 3, 0], 0.5, name=8)), name=7), 'keep', 7, [Sphere([0, 3, 0], 0.5, name=8)]),
    (1, Body(Shape('C', Sphere([0, 3, 0], 0.5, name=8)), name=2), 'clash', 5, [Sphere([0, 3, 0], 0.5, name=8)]),
    (1, Body(Shape('C', Sphere([0, 3, 0], 0.5, name=2)), name=8), 'clash', 8, [Sphere([0, 3, 0], 0.5, name=6)]),
    (1, Body(Shape('C', Sphere([0, 3, 0], 0.5, name=8)), name=7), 'clash', 7, [Sphere([0, 3, 0], 0.5, name=8)]),
    (1, Body(Shape('C', Sphere([0, 3, 0], 0.5, name=8)), name=2), 'new', 5, [Sphere([0, 3, 0], 0.5, name=6)]),
    (1, Body(Shape('C', Sphere([0, 3, 0], 0.5, name=2)), name=8), 'new', 5, [Sphere([0, 3, 0], 0.5, name=6)]),
    (1, Body(Shape('C', Sphere([0, 3, 0], 0.5, name=8)), name=7), 'new', 5, [Sphere([0, 3, 0], 0.5, name=6)]),
    (1, Body(Shape('C', Sphere([0, 0, 0], 2, name=8)), name=2), 'new', 5, []),
    (1, Body(Shape('C', Sphere([0, 0, 0], 2, name=2)), name=8), 'keep', 8, []),
    (1, Body(Shape('C', Sphere([0, 0, 0], 2, name=8)), name=7), 'new', 5, []),

])
def test_add(universe, case, cell, name_rule, new_name, new_surfs):
    u = universe(case)
    s_before = u.get_surfaces()
    if new_surfs is None or new_name is None:
        with pytest.raises(NameClashError):
            u.add_cell(cell, name_rule=name_rule)
    else:
        u.add_cell(cell, name_rule=name_rule)
        s_after = u.get_surfaces()
        acell = u._cells[-1]
        assert acell.shape == cell.shape
        assert acell is not cell
        assert acell.name() == new_name
        for k, v in s_before.items():
            assert v is s_after[k]
        assert len(s_after.keys()) - len(s_before.keys()) == len(new_surfs)
        for s in new_surfs:
            assert s_after[s.name()] == s
        assert acell.options['U'] is u


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


@pytest.mark.parametrize('case', [

])
def test_apply_fill(universe, case):
    pass

# @pytest.mark.parametrize('case_no, cells, recur, simp, complexity', [
#     (0, 1, False, False, {2: 3, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}),
#     (0, 2, False, False, {10: 6, 11: 6, 12: 9, 3: 5, 1: 2}),
#     (0, None, False, False, {10: 6, 11: 6, 12: 9, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}),
#     (0, 1, True, False, {2: 3, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}),
#     (0, 2, True, False, {10: 6, 11: 6, 12: 9, 3: 5, 1: 2}),
#     (0, None, True, False, {10: 6, 11: 6, 12: 9, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}),
#     pytest.param(0, 1, False, True, {2: 3, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}, marks=pytest.mark.xfail(reason='need full simplification approach')),
#     pytest.param(0, 2, False, True, {10: 6, 11: 6, 12: 9, 3: 5, 1: 2}, marks=pytest.mark.xfail(reason='need full simplification approach')),
#     pytest.param(0, None, False, True, {10: 6, 11: 6, 12: 9, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}, marks=pytest.mark.xfail(reason='need full simplification approach')),
#     pytest.param(0, 1, True, True, {2: 3, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}, marks=pytest.mark.xfail(reason='need full simplification approach')),
#     pytest.param(0, 2, True, True, {10: 6, 11: 6, 12: 9, 3: 5, 1: 2}, marks=pytest.mark.xfail(reason='need full simplification approach')),
#     pytest.param(0, None, True, True, {10: 6, 11: 6, 12: 9, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}, marks=pytest.mark.xfail(reason='need full simplification approach')),
# ])
# def test_fill(universe, case_no, cells, recur, simp, complexity):
#     u = universe[case_no]
#     u = u.fill(cells=cells, recurrent=recur, simplify=simp)
#     assert len(u._cells) == len(complexity.keys())
#     for c in u:
#         print(c['name'], c.shape.complexity())
#         assert c.shape.complexity() == complexity[c['name']]
#
#
# @pytest.mark.slow
# @pytest.mark.parametrize('case_no, u_name, complexity', [
#     pytest.param(0, 0, {1: 2, 2: 3, 3: 4}, marks=pytest.mark.xfail(reason='need full simplification approach')),
#     pytest.param(0, 1, {10: 3, 11: 3, 12: 4}, marks=pytest.mark.xfail(reason='need full simplification approach')),
#     pytest.param(0, 2, {21: 3, 22: 4, 23: 3, 24: 4}, marks=pytest.mark.xfail(reason='need full simplification approach'))
# ])
# def test_simplify(universe, case_no, u_name, complexity):
#     base_u = universe[case_no]
#     if u_name == 0:
#         u = base_u
#     else:
#         u = base_u.select_universe(u_name)
#     u.simplify(trim_size=5)
#     assert len(u._cells) == len(complexity.keys())
#     for c in u:
#         assert c.shape.complexity() == complexity[c['name']]
#
#
# @pytest.mark.slow
# @pytest.mark.parametrize('tol', [0.2, None])
# @pytest.mark.parametrize('case_no, u_name, expected', [
#     (0, 0, [[-5, 6], [-3, 3], [-3, 3]]),
#     (0, 1, [[3, 9], [-1, 1], [-1, 1]]),
#     (0, 2, [[-6, -1], [-4, 4], [-4, 4]])
# ])
# def test_bounding_box(universe, tol, case_no, u_name, expected):
#     base_u = universe[case_no]
#     if u_name == 0:
#         u = base_u
#     else:
#         u = base_u.select_universe(u_name)
#     base = [0, 0, 0]
#     dims = [30, 30, 30]
#     gb = Box(base, dims[0], dims[1], dims[2])
#     if tol is not None:
#         bb = u.bounding_box(box=gb, tol=tol)
#     else:
#         tol = 100.0
#         bb = u.bounding_box()
#     for j, (low, high) in enumerate(expected):
#         bbdim = 0.5 * bb.dimensions[j]
#         assert bb.center[j] - bbdim <= low
#         assert bb.center[j] - bbdim >= low - tol
#         assert bb.center[j] + bbdim >= high
#         assert bb.center[j] + bbdim <= high + tol
#
#
# @pytest.mark.parametrize('case_no, u_name, sur_names', [
#     (0, 0, {1, 2, 3, 4}),
#     (0, 1, {10, 11, 12, 13}),
#     (0, 2, {20, 21, 22, 23, 24})
# ])
# def test_get_surfaces(universe, case_no, u_name, sur_names):
#     base_u = universe[case_no]
#     if u_name == 0:
#         u = base_u
#     else:
#         u = base_u.select_universe(u_name)
#     surfs = u.get_surfaces()
#     sn = set(s.options['name'] for s in surfs)
#     assert sn == sur_names
#
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
# @pytest.mark.parametrize('case_no, rec, expected', [
#     (0, False, {1, 2}),
#     (0, True, {1, 2}),
#     (1, False, {2}),
#     (1, True, {2})
# ])
# def test_get_universes(universe, case_no, rec, expected):
#     u = universe[case_no].get_universes(recurrent=rec)
#     assert u == expected
#
#
# @pytest.mark.parametrize('case_no, u_name, expected', [
#     (0, 0, {'cells': {1, 2, 3}, 'surfaces': {1, 2, 3, 4},
#             'compositions': set(), 'transformations': set()}),
#     (0, 1, {'cells': {1, 2, 3}, 'surfaces': {1, 2, 3, 4},
#             'compositions': {1, 2}, 'transformations': set()}),
#     (0, 2, {'cells': {1, 2, 3, 4}, 'surfaces': {1, 2, 3, 4, 5},
#             'compositions': {1, 2, 3}, 'transoformations': set()})
# ])
# def test_rename(universe, case_no, u_name, expected):
#     base_u = universe[case_no]
#     if u_name == 0:
#         u = base_u.copy()
#     else:
#         u = base_u.select_universe(u_name).copy()
#     u.rename()
#     surf_names = {s.options['name'] for s in u.get_surfaces()}
#     comp_names = {m.composition['name'] for m in u.get_materials()}
#     cell_names = {c['name'] for c in u if 'name' in c.keys()}
#     assert surf_names == expected['surfaces']
#     assert cell_names == expected['cells']
#     assert comp_names == expected['compositions']
#
#
# @pytest.mark.skip
# def test_save():
#     raise NotImplementedError
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
#     s, r = u.check_names()
#     assert s == status
#     # assert r == output
#
