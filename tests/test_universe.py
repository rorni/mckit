import pytest
import numpy as np

from mckit.universe import Universe
from mckit.material import Material, Element
from mckit.parser.mcnp_input_parser import read_mcnp


cases = [
    'tests/universe1.i'
]


@pytest.fixture(scope='module')
def universe():
    univs = []
    for filename in cases:
        univs.append(read_mcnp(filename))
    return univs


def test_transform():
    raise NotImplementedError


@pytest.mark.parametrize('case_no, cells, recur, simp, complexity', [
    (0, 1, False, False, {2: 3, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}),
    (0, 2, False, False, {10: 6, 11: 6, 12: 9, 3: 5, 1: 2}),
    (0, None, False, False, {10: 6, 11: 6, 12: 9, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}),
    (0, 1, True, False, {2: 3, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}),
    (0, 2, True, False, {10: 6, 11: 6, 12: 9, 3: 5, 1: 2}),
    (0, None, True, False, {10: 6, 11: 6, 12: 9, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}),
    (0, 1, False, True, {2: 3, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}),
    (0, 2, False, True, {10: 6, 11: 6, 12: 9, 3: 5, 1: 2}),
    (0, None, False, True, {10: 6, 11: 6, 12: 9, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}),
    (0, 1, True, True, {2: 3, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}),
    (0, 2, True, True, {10: 6, 11: 6, 12: 9, 3: 5, 1: 2}),
    (0, None, True, True, {10: 6, 11: 6, 12: 9, 3: 5, 21: 5, 22: 6, 23: 5, 24: 12}),
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
    (0, 0, {1: 2, 2: 3, 3: 4}),
    (0, 1, {10: 3, 11: 3, 12: 4}),
    (0, 2, {21: 3, 22: 4, 23: 3, 24: 4})
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


get_u_ans = [
    [{1, 2}, {1, 2}]
]


@pytest.mark.parametrize('case_no', range(len(cases)))
@pytest.mark.parametrize('rec', [False, True])
def test_get_universes(universe, case_no, rec):
    u = universe[case_no].get_universes(recurrent=rec)
    assert u == get_u_ans[case_no][int(rec)]


def test_rename():
    raise NotImplementedError


