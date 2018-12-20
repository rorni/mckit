import pytest
import numpy as np

from mckit.material import Material, Element
from mckit.parser.mcnp_input_parser import read_mcnp
from mckit.transformation import Transformation
from mckit.geometry import Box


cases = [
    'tests/universe1.i',
    'tests/universe2.i'
]


@pytest.fixture(scope='module')
def universe():
    univs = []
    for filename in cases:
        univs.append(read_mcnp(filename))
    return univs


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
@pytest.mark.parametrize('case_no, u_name, extent', [
    (0, 0, [20, 10, 10]),
    (0, 1, [20, 10, 10]),
    (0, 2, [20, 10, 10]),
    (1, 0, [20, 10, 10]),
    (1, 2, [20, 10, 10])
])
def test_transform(universe, case_no, u_name, tr, extent):
    base_u = universe[case_no]
    if u_name == 0:
        u = base_u
    else:
        u = base_u.select_universe(u_name)
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


@pytest.mark.parametrize('case_no, cells, recur, simp, complexity', [
    (0, 1, False, False, {2: 3, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}),
    (0, 2, False, False, {10: 6, 11: 6, 12: 9, 3: 5, 1: 2}),
    (0, None, False, False, {10: 6, 11: 6, 12: 9, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}),
    (0, 1, True, False, {2: 3, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}),
    (0, 2, True, False, {10: 6, 11: 6, 12: 9, 3: 5, 1: 2}),
    (0, None, True, False, {10: 6, 11: 6, 12: 9, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}),
    pytest.param(0, 1, False, True, {2: 3, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}, marks=pytest.mark.xfail(reason='need full simplification approach')),
    pytest.param(0, 2, False, True, {10: 6, 11: 6, 12: 9, 3: 5, 1: 2}, marks=pytest.mark.xfail(reason='need full simplification approach')),
    pytest.param(0, None, False, True, {10: 6, 11: 6, 12: 9, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}, marks=pytest.mark.xfail(reason='need full simplification approach')),
    pytest.param(0, 1, True, True, {2: 3, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}, marks=pytest.mark.xfail(reason='need full simplification approach')),
    pytest.param(0, 2, True, True, {10: 6, 11: 6, 12: 9, 3: 5, 1: 2}, marks=pytest.mark.xfail(reason='need full simplification approach')),
    pytest.param(0, None, True, True, {10: 6, 11: 6, 12: 9, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}, marks=pytest.mark.xfail(reason='need full simplification approach')),
])
def test_fill(universe, case_no, cells, recur, simp, complexity):
    u = universe[case_no]
    u = u.fill(cells=cells, recurrent=recur, simplify=simp)
    assert len(u._cells) == len(complexity.keys())
    for c in u:
        print(c['name'], c.shape.complexity())
        assert c.shape.complexity() == complexity[c['name']]


@pytest.mark.slow
@pytest.mark.parametrize('case_no, u_name, complexity', [
    pytest.param(0, 0, {1: 2, 2: 3, 3: 4}, marks=pytest.mark.xfail(reason='need full simplification approach')),
    pytest.param(0, 1, {10: 3, 11: 3, 12: 4}, marks=pytest.mark.xfail(reason='need full simplification approach')),
    pytest.param(0, 2, {21: 3, 22: 4, 23: 3, 24: 4}, marks=pytest.mark.xfail(reason='need full simplification approach'))
])
def test_simplify(universe, case_no, u_name, complexity):
    base_u = universe[case_no]
    if u_name == 0:
        u = base_u
    else:
        u = base_u.select_universe(u_name)
    u.simplify(trim_size=5)
    assert len(u._cells) == len(complexity.keys())
    for c in u:
        assert c.shape.complexity() == complexity[c['name']]


@pytest.mark.slow
@pytest.mark.parametrize('tol', [0.2, None])
@pytest.mark.parametrize('case_no, u_name, expected', [
    (0, 0, [[-5, 6], [-3, 3], [-3, 3]]),
    (0, 1, [[3, 9], [-1, 1], [-1, 1]]),
    (0, 2, [[-6, -1], [-4, 4], [-4, 4]])
])
def test_bounding_box(universe, tol, case_no, u_name, expected):
    base_u = universe[case_no]
    if u_name == 0:
        u = base_u
    else:
        u = base_u.select_universe(u_name)
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


@pytest.mark.parametrize('case_no, u_name, sur_names', [
    (0, 0, {1, 2, 3, 4}),
    (0, 1, {10, 11, 12, 13}),
    (0, 2, {20, 21, 22, 23, 24})
])
def test_get_surfaces(universe, case_no, u_name, sur_names):
    base_u = universe[case_no]
    if u_name == 0:
        u = base_u
    else:
        u = base_u.select_universe(u_name)
    surfs = u.get_surfaces()
    sn = set(s.options['name'] for s in surfs)
    assert sn == sur_names


@pytest.mark.parametrize('case_no, u_name, materials', [
    (0, 0, set()),
    (0, 1, {Material(atomic=[(Element('C12', lib='21C'), 1)], density=2.7),
            Material(weight=[(Element('H1', lib='21C'), 0.11191),
                             (Element('O16', lib='50c'), 0.88809)], density=1.0)}),
    (0, 2, {Material(atomic=[(Element('H1', lib='21C'), 2),
                             (Element('C12', lib='21C'), 1)], density=0.9),
            Material(atomic=[(Element('Fe56', lib='21c'), 1)], density=7.8),
            Material(weight=[(Element('B10', lib='21c'), 1)], density=1.8)})
])
def test_get_materials(universe, case_no, u_name, materials):
    base_u = universe[case_no]
    if u_name == 0:
        u = base_u
    else:
        u = base_u.select_universe(u_name)
    mats = u.get_materials()
    assert mats == materials


@pytest.mark.parametrize('case_no, rec, expected', [
    (0, False, {1, 2}),
    (0, True, {1, 2}),
    (1, False, {2}),
    (1, True, {2})
])
def test_get_universes(universe, case_no, rec, expected):
    u = universe[case_no].get_universes(recurrent=rec)
    assert u == expected


@pytest.mark.parametrize('case_no, u_name, expected', [
    (0, 0, {'cells': {1, 2, 3}, 'surfaces': {1, 2, 3, 4},
            'compositions': set(), 'transformations': set()}),
    (0, 1, {'cells': {1, 2, 3}, 'surfaces': {1, 2, 3, 4},
            'compositions': {1, 2}, 'transformations': set()}),
    (0, 2, {'cells': {1, 2, 3, 4}, 'surfaces': {1, 2, 3, 4, 5},
            'compositions': {1, 2, 3}, 'transoformations': set()})
])
def test_rename(universe, case_no, u_name, expected):
    base_u = universe[case_no]
    if u_name == 0:
        u = base_u.copy()
    else:
        u = base_u.select_universe(u_name).copy()
    u.rename()
    surf_names = {s.options['name'] for s in u.get_surfaces()}
    comp_names = {m.composition['name'] for m in u.get_materials()}
    cell_names = {c['name'] for c in u if 'name' in c.keys()}
    assert surf_names == expected['surfaces']
    assert cell_names == expected['cells']
    assert comp_names == expected['compositions']


@pytest.mark.skip
def test_save():
    raise NotImplementedError


@pytest.mark.parametrize('case_no, cells, status, output', [
    (0, None, False, {}),
    (0, 1, False, {}),
    (0, 2, False, {}),
    (1, None, True, {
        'cells': {21: {None}, 22: {None}, 23: [None], 24: [None]},
        'surfaces': {20: [None], 21: [None], 22: [None], 23: [None], 24: [None]}
    }),
    (1, 1, True, {'cells': {21: [2], 22: [2], 23: [2], 24: [2]}}),
    (1, 2, True, {
        'cells': {21: [2], 22: [2], 23: [2], 24: [2]},
        'surfaces': {20: [2], 21: [2], 22: [2], 23: [2], 24: [2]}
    })
])
def test_check_names(universe, case_no, cells, status, output):
    u = universe[case_no]
    u = u.fill(cells=cells)
    s, r = u.check_names()
    assert s == status
    # assert r == output

