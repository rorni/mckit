import pytest
import numpy as np

from mckit.body import Shape, Body
from mckit.surface import create_surface
from mckit.geometry import Box
from mckit.material import Material
from mckit.transformation import Transformation


@pytest.fixture(scope='module')
def surfaces():
    surf_data = {
        1: ('sx', [4, 2]),
        2: ('cx', [2]),
        3: ('px', [-3]),
        4: ('sx', [-3, 1]),
        5: ('px', [4]),
        6: ('sx', [4, 1]),
        7: ('cx', [3]),
        8: ('cx', [1]),
        9: ('px', [-5]),
        10: ('px', [8])
    }
    surfs = {}
    for name, (kind, params) in surf_data.items():
        surfs[name] = create_surface(kind, *params, name=name)
    return surfs


def create_node(kind, args, surfs):
    new_args = []
    for g in args:
        if isinstance(g, tuple):
            g = create_node(g[0], g[1], surfs)
        else:
            g = surfs[g]
        new_args.append(g)
    return Shape(kind, *new_args)


basic_geoms = [
    ('U', [('I', [('C', [2]), ('S', [3]), ('S', [1]), ('C', [5])]), ('C', [4])]),
    ('I', [('C', [6]), ('C', [1])]),
    ('U', [('C', [6]), ('C', [1])]),
    ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('S', [9])]), ('I', [('S', [6]), ('S', [1]), ('C', [2]), ('C', [5]), ('S', [3])])])])]),
    ('U', [('C', [1]), ('I', [('C', [5]), ('S', [3]), ('C', [2])])]),
    ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('S', [3]), ('C', [8]), ('C', [5])])])]),
    ('U', [('I', [('C', [1]), ('C', [5])]), ('I', [('C', [1]), ('S', [5])])]),
    ('I', [('U', [('S', [1]), ('S', [5])]), ('U', [('S', [1]), ('C', [5])])])
]


@pytest.fixture(scope='class')
def geometry(surfaces):
    geoms = [create_node(g[0], g[1], surfaces) for g in basic_geoms]
    return geoms


class TestShape:
    @staticmethod
    def filter_arg(arg, surfs):
        if isinstance(arg, int):
            return surfs[arg]
        elif isinstance(arg, tuple):
            return create_node(arg[0], arg[1], surfs)
        else:
            return arg

    @pytest.mark.parametrize('opc, args, ans_opc, ans_args', [
        ('S', [1], 'S', [1]),
        ('C', [1], 'C', [1]),
        ('E', [], 'E', []),
        ('R', [], 'R', []),
        ('C', [('E', [])], 'R', []),
        ('C', [('R', [])], 'E', []),
        ('S', [('S', [1])], 'S', [1]),
        ('S', [('I', [1, 2])], 'I', [('S', [1]), ('S', [2])]),
        ('S', [('U', [1, 2])], 'U', [('S', [1]), ('S', [2])]),
        ('I', [('S', [1])], 'S', [1]),
        ('I', [('S', [1]), ('C', [2])], 'I', [('S', [1]), ('C', [2])]),
        ('I', [1, ('C', [2])], 'I', [('S', [1]), ('C', [2])]),
        ('U', [1, 2], 'U', [('S', [1]), ('S', [2])]),
        ('C', [('S', [1])], 'C', [1]),
        ('C', [('C', [1])], 'S', [1]),
        ('I', [('I', [('S', [1]), ('C', [2])]), ('S', [3])],
         'I', [('S', [1]), ('C', [2]), ('S', [3])]),
        ('U', [('U', [('I', [('S', [1]), ('S', [2])]), ('C', [3])]),
               ('U', [('S', [4]), ('S', [5])]), ('I', [('S', [6]), ('C', [1])])],
         'U', [('I', [('S', [1]), ('S', [2])]), ('C', [3]), ('S', [4]),
               ('S', [5]), ('I', [('S', [6]), ('C', [1])])]),
        ('I', [('S', [1]), ('C', [1])], 'E', []),
        ('U', [('S', [1]), ('C', [1])], 'R', []),
        ('I', [('S', [1]), ('E', [])], 'E', []),
        ('I', [('S', [1]), ('R', [])], 'S', [1]),
        ('U', [('S', [1]), ('E', [])], 'S', [1]),
        ('U', [('S', [1]), ('R', [])], 'R', []),
    ])
    def test_create(self, surfaces, opc, args, ans_opc, ans_args):
        args = [self.filter_arg(a, surfaces) for a in args]
        ans_args = sorted([self.filter_arg(a, surfaces) for a in ans_args], key=hash)
        shape = Shape(opc, *args)
        print(shape)
        assert shape.opc == ans_opc
        assert shape.args == tuple(ans_args)

    @pytest.mark.parametrize('opc, args', [
        ('E', [1]),
        ('R', [1]),
        ('S', [1, 2]),
        ('S', [('S', [1]), ('C', [2])]),
        ('C', [1, 2]),
        ('C', [('C', [1]), ('S', [2])]),
        ('I', []),
        ('U', [])
    ])
    def test_create_failure(self, surfaces, opc, args):
        args = [self.filter_arg(a, surfaces) for a in args]
        with pytest.raises(ValueError):
            Shape(opc, *args)

    polish_cases = [
        [2, 'C', 3, 'I', 1, 'I', ('C', [5]), 'I', 4, 'C', 'U'],
        [6, 'C', 1, 'C', 'I'],
        [6, 'C', 1, 'C', 'U'],
        [6, 1, 'I', 2, 'C', 'I', 5, 'C', 'I', 3, 'I', 4, 'C', 9, 'I', 'U', 3, 'C',
         4, 'C', 'I', 'U', 7, 'C', 'I', 10, 'C', 'I', 4, 'C', 10, 'C', 'I', 'U'],
        [5, 'C', 3, 'I', 2, 'C', 'I', 1, 'C', 'U'],
        [3, 8, 'C', 'I', 5, 'C', 'I', 4, 'C', 'U', 6, 'C', 'U', 2, 'C', 'I'],
        [1, 'C', 5, 'C', 'I', 1, 'C', 5, 'I', 'U'],
        [1, 5, 'U', 1, 5, 'C', 'U', 'I']
    ]

    @pytest.mark.parametrize('case_no, polish', enumerate(polish_cases))
    def test_from_polish(self, geometry, surfaces, case_no, polish):
        polish = [self.filter_arg(a, surfaces) for a in polish]
        shape = Shape.from_polish_notation(polish)
        assert shape == geometry[case_no]

    @pytest.mark.parametrize('case_no, expected', enumerate([
        ('I', [('U', [('S', [2]), ('C', [3]), ('C', [1]), ('S', [5])]), ('S', [4])]),
        ('U', [('S', [6]), ('S', [1])]),
        ('I', [('S', [6]), ('S', [1])]),
        ('I', [('U', [('S', [4]), ('S', [10])]), ('U', [('S', [7]), ('S', [10]), ('I', [('U', [('S', [3]), ('S', [4])]), ('U', [('S', [4]), ('C', [9])]), ('U', [('C', [6]), ('C', [1]), ('S', [2]), ('S', [5]), ('C', [3])])])])]),
        ('I', [('S', [1]), ('U', [('S', [5]), ('C', [3]), ('S', [2])])]),
        ('U', [('S', [2]), ('I', [('S', [6]), ('S', [4]), ('U', [('C', [3]), ('S', [8]), ('S', [5])])])]),
        ('I', [('U', [('S', [1]), ('S', [5])]), ('U', [('S', [1]), ('C', [5])])]),
        ('U', [('I', [('C', [1]), ('C', [5])]), ('I', [('C', [1]), ('S', [5])])])
    ]))
    def test_complement(self, geometry, surfaces, case_no, expected):
        expected = create_node(expected[0], expected[1], surfaces)
        shape = geometry[case_no].complement()
        assert shape == expected

    @pytest.mark.parametrize('no1, no2, opc, args', [
        (0, 1, 'I', [('U', [('I', [('C', [2]), ('S', [3]), ('S', [1]), ('C', [5])]), ('C', [4])]), ('I', [('C', [6]), ('C', [1])])]),
        (0, 2, 'I', [('U', [('I', [('C', [2]), ('S', [3]), ('S', [1]), ('C', [5])]), ('C', [4])]), ('U', [('C', [6]), ('C', [1])])]),
        (0, 3, 'I', [('U', [('I', [('C', [2]), ('S', [3]), ('S', [1]), ('C', [5])]), ('C', [4])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('S', [9])]), ('I', [('S', [6]), ('S', [1]), ('C', [2]), ('C', [5]), ('S', [3])])])])])]),
        (0, 4, 'I', [('U', [('I', [('C', [2]), ('S', [3]), ('S', [1]), ('C', [5])]), ('C', [4])]), ('U', [('C', [1]), ('I', [('C', [5]), ('S', [3]), ('C', [2])])])]),
        (0, 5, 'I', [('U', [('I', [('C', [2]), ('S', [3]), ('S', [1]), ('C', [5])]), ('C', [4])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('S', [3]), ('C', [8]), ('C', [5])])])])]),
        (1, 0, 'I', [('U', [('I', [('C', [2]), ('S', [3]), ('S', [1]), ('C', [5])]), ('C', [4])]), ('I', [('C', [6]), ('C', [1])])]),
        (1, 2, 'I', [('I', [('C', [6]), ('C', [1])]), ('U', [('C', [6]), ('C', [1])])]),
        (1, 3, 'I', [('I', [('C', [6]), ('C', [1])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('S', [9])]), ('I', [('S', [6]), ('S', [1]), ('C', [2]), ('C', [5]), ('S', [3])])])])])]),
        (1, 4, 'I', [('I', [('C', [6]), ('C', [1])]), ('U', [('C', [1]), ('I', [('C', [5]), ('S', [3]), ('C', [2])])])]),
        (1, 5, 'I', [('I', [('C', [6]), ('C', [1])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('S', [3]), ('C', [8]), ('C', [5])])])])]),        (2, 0, 'I', [('U', [('I', [('C', [2]), ('S', [3]), ('S', [1]), ('C', [5])]), ('C', [4])]), ('U', [('C', [6]), ('C', [1])])]),
        (2, 1, 'I', [('I', [('C', [6]), ('C', [1])]), ('U', [('C', [6]), ('C', [1])])]),
        (2, 3, 'I', [('U', [('C', [6]), ('C', [1])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('S', [9])]), ('I', [('S', [6]), ('S', [1]), ('C', [2]), ('C', [5]), ('S', [3])])])])])]),
        (2, 4, 'I', [('U', [('C', [6]), ('C', [1])]), ('U', [('C', [1]), ('I', [('C', [5]), ('S', [3]), ('C', [2])])])]),
        (2, 5, 'I', [('U', [('C', [6]), ('C', [1])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('S', [3]), ('C', [8]), ('C', [5])])])])]),
        (3, 0, 'I', [('U', [('I', [('C', [2]), ('S', [3]), ('S', [1]), ('C', [5])]), ('C', [4])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('S', [9])]), ('I', [('S', [6]), ('S', [1]), ('C', [2]), ('C', [5]), ('S', [3])])])])])]),
        (3, 1, 'I', [('I', [('C', [6]), ('C', [1])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('S', [9])]), ('I', [('S', [6]), ('S', [1]), ('C', [2]), ('C', [5]), ('S', [3])])])])])]),
        (3, 2, 'I', [('U', [('C', [6]), ('C', [1])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('S', [9])]), ('I', [('S', [6]), ('S', [1]), ('C', [2]), ('C', [5]), ('S', [3])])])])])]),
        (3, 4, 'I', [('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('S', [9])]), ('I', [('S', [6]), ('S', [1]), ('C', [2]), ('C', [5]), ('S', [3])])])])]), ('U', [('C', [1]), ('I', [('C', [5]), ('S', [3]), ('C', [2])])])]),
        (3, 5, 'I', [('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('S', [9])]), ('I', [('S', [6]), ('S', [1]), ('C', [2]), ('C', [5]), ('S', [3])])])])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('S', [3]), ('C', [8]), ('C', [5])])])])]),
        (4, 0, 'I', [('U', [('I', [('C', [2]), ('S', [3]), ('S', [1]), ('C', [5])]), ('C', [4])]), ('U', [('C', [1]), ('I', [('C', [5]), ('S', [3]), ('C', [2])])])]),
        (4, 1, 'I', [('I', [('C', [6]), ('C', [1])]), ('U', [('C', [1]), ('I', [('C', [5]), ('S', [3]), ('C', [2])])])]),
        (4, 2, 'I', [('U', [('C', [6]), ('C', [1])]), ('U', [('C', [1]), ('I', [('C', [5]), ('S', [3]), ('C', [2])])])]),
        (4, 3, 'I', [('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('S', [9])]), ('I', [('S', [6]), ('S', [1]), ('C', [2]), ('C', [5]), ('S', [3])])])])]), ('U', [('C', [1]), ('I', [('C', [5]), ('S', [3]), ('C', [2])])])]),
        (4, 5, 'I', [('U', [('C', [1]), ('I', [('C', [5]), ('S', [3]), ('C', [2])])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('S', [3]), ('C', [8]), ('C', [5])])])])]),
        (5, 0, 'I', [('U', [('I', [('C', [2]), ('S', [3]), ('S', [1]), ('C', [5])]), ('C', [4])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('S', [3]), ('C', [8]), ('C', [5])])])])]),
        (5, 1, 'I', [('I', [('C', [6]), ('C', [1])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('S', [3]), ('C', [8]), ('C', [5])])])])]),
        (5, 2, 'I', [('U', [('C', [6]), ('C', [1])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('S', [3]), ('C', [8]), ('C', [5])])])])]),
        (5, 3, 'I', [('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('S', [9])]), ('I', [('S', [6]), ('S', [1]), ('C', [2]), ('C', [5]), ('S', [3])])])])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('S', [3]), ('C', [8]), ('C', [5])])])])]),
        (5, 4, 'I', [('U', [('C', [1]), ('I', [('C', [5]), ('S', [3]), ('C', [2])])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('S', [3]), ('C', [8]), ('C', [5])])])])])
    ])
    def test_intersection(self, geometry, surfaces, no1, no2, opc, args):
        expected = create_node(opc, args, surfaces)
        result = Shape.intersection(geometry[no1], geometry[no2])
        assert result == expected

    @pytest.mark.parametrize('no1, no2, opc, args', [
        (0, 1, 'U', [('U', [('I', [('C', [2]), ('S', [3]), ('S', [1]), ('C', [5])]), ('C', [4])]), ('I', [('C', [6]), ('C', [1])])]),
        (0, 2, 'U', [('U', [('I', [('C', [2]), ('S', [3]), ('S', [1]), ('C', [5])]), ('C', [4])]), ('U', [('C', [6]), ('C', [1])])]),
        (0, 3, 'U', [('U', [('I', [('C', [2]), ('S', [3]), ('S', [1]), ('C', [5])]), ('C', [4])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('S', [9])]), ('I', [('S', [6]), ('S', [1]), ('C', [2]), ('C', [5]), ('S', [3])])])])])]),
        (0, 4, 'U', [('U', [('I', [('C', [2]), ('S', [3]), ('S', [1]), ('C', [5])]), ('C', [4])]), ('U', [('C', [1]), ('I', [('C', [5]), ('S', [3]), ('C', [2])])])]),
        (0, 5, 'U', [('U', [('I', [('C', [2]), ('S', [3]), ('S', [1]), ('C', [5])]), ('C', [4])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('S', [3]), ('C', [8]), ('C', [5])])])])]),
        (1, 0, 'U', [('U', [('I', [('C', [2]), ('S', [3]), ('S', [1]), ('C', [5])]), ('C', [4])]), ('I', [('C', [6]), ('C', [1])])]),
        (1, 2, 'U', [('I', [('C', [6]), ('C', [1])]), ('U', [('C', [6]), ('C', [1])])]),
        (1, 3, 'U', [('I', [('C', [6]), ('C', [1])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('S', [9])]), ('I', [('S', [6]), ('S', [1]), ('C', [2]), ('C', [5]), ('S', [3])])])])])]),
        (1, 4, 'U', [('I', [('C', [6]), ('C', [1])]), ('U', [('C', [1]), ('I', [('C', [5]), ('S', [3]), ('C', [2])])])]),
        (1, 5, 'U', [('I', [('C', [6]), ('C', [1])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('S', [3]), ('C', [8]), ('C', [5])])])])]),
        (2, 0, 'U', [('U', [('I', [('C', [2]), ('S', [3]), ('S', [1]), ('C', [5])]), ('C', [4])]), ('U', [('C', [6]), ('C', [1])])]),
        (2, 1, 'U', [('I', [('C', [6]), ('C', [1])]), ('U', [('C', [6]), ('C', [1])])]),
        (2, 3, 'U', [('U', [('C', [6]), ('C', [1])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('S', [9])]), ('I', [('S', [6]), ('S', [1]), ('C', [2]), ('C', [5]), ('S', [3])])])])])]),
        (2, 4, 'U', [('U', [('C', [6]), ('C', [1])]), ('U', [('C', [1]), ('I', [('C', [5]), ('S', [3]), ('C', [2])])])]),
        (2, 5, 'U', [('U', [('C', [6]), ('C', [1])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('S', [3]), ('C', [8]), ('C', [5])])])])]),
        (3, 0, 'U', [('U', [('I', [('C', [2]), ('S', [3]), ('S', [1]), ('C', [5])]), ('C', [4])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('S', [9])]), ('I', [('S', [6]), ('S', [1]), ('C', [2]), ('C', [5]), ('S', [3])])])])])]),
        (3, 1, 'U', [('I', [('C', [6]), ('C', [1])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('S', [9])]), ('I', [('S', [6]), ('S', [1]), ('C', [2]), ('C', [5]), ('S', [3])])])])])]),
        (3, 2, 'U', [('U', [('C', [6]), ('C', [1])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('S', [9])]), ('I', [('S', [6]), ('S', [1]), ('C', [2]), ('C', [5]), ('S', [3])])])])])]),
        (3, 4, 'U', [('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('S', [9])]), ('I', [('S', [6]), ('S', [1]), ('C', [2]), ('C', [5]), ('S', [3])])])])]), ('U', [('C', [1]), ('I', [('C', [5]), ('S', [3]), ('C', [2])])])]),
        (3, 5, 'U', [('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('S', [9])]), ('I', [('S', [6]), ('S', [1]), ('C', [2]), ('C', [5]), ('S', [3])])])])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('S', [3]), ('C', [8]), ('C', [5])])])])]),        (4, 0, 'U', [('U', [('I', [('C', [2]), ('S', [3]), ('S', [1]), ('C', [5])]), ('C', [4])]), ('U', [('C', [1]), ('I', [('C', [5]), ('S', [3]), ('C', [2])])])]),
        (4, 1, 'U', [('I', [('C', [6]), ('C', [1])]), ('U', [('C', [1]), ('I', [('C', [5]), ('S', [3]), ('C', [2])])])]),
        (4, 2, 'U', [('U', [('C', [6]), ('C', [1])]), ('U', [('C', [1]), ('I', [('C', [5]), ('S', [3]), ('C', [2])])])]),
        (4, 3, 'U', [('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('S', [9])]), ('I', [('S', [6]), ('S', [1]), ('C', [2]), ('C', [5]), ('S', [3])])])])]), ('U', [('C', [1]), ('I', [('C', [5]), ('S', [3]), ('C', [2])])])]),
        (4, 5, 'U', [('U', [('C', [1]), ('I', [('C', [5]), ('S', [3]), ('C', [2])])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('S', [3]), ('C', [8]), ('C', [5])])])])]),
        (5, 0, 'U', [('U', [('I', [('C', [2]), ('S', [3]), ('S', [1]), ('C', [5])]), ('C', [4])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('S', [3]), ('C', [8]), ('C', [5])])])])]),
        (5, 1, 'U', [('I', [('C', [6]), ('C', [1])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('S', [3]), ('C', [8]), ('C', [5])])])])]),
        (5, 2, 'U', [('U', [('C', [6]), ('C', [1])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('S', [3]), ('C', [8]), ('C', [5])])])])]),
        (5, 3, 'U', [('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('S', [9])]), ('I', [('S', [6]), ('S', [1]), ('C', [2]), ('C', [5]), ('S', [3])])])])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('S', [3]), ('C', [8]), ('C', [5])])])])]),
        (5, 4, 'U', [('U', [('C', [1]), ('I', [('C', [5]), ('S', [3]), ('C', [2])])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('S', [3]), ('C', [8]), ('C', [5])])])])])
    ])
    def test_union(self, geometry, surfaces, no1, no2, opc, args):
        expected = create_node(opc, args, surfaces)
        result = Shape.union(geometry[no1], geometry[no2])
        assert result == expected

    @pytest.mark.parametrize('geom_no', range(len(basic_geoms)))
    @pytest.mark.parametrize('point, ans', [
        ([-6, 0, 0],     [-1, -1, -1, -1, -1, -1, -1, +1]),
        ([-3.5, 0, 0],   [+1, -1, -1, +1, -1, +1, -1, +1]),
        ([-3.5, 1.5, 0], [-1, -1, -1, -1, -1, -1, -1, +1]),
        ([-2.5, 1.5, 0], [+1, -1, -1, +1, +1, -1, -1, +1]),
        ([-1, 2.5, 0],   [-1, -1, -1, -1, -1, -1, -1, +1]),
        ([1, -1.5, 0],   [+1, -1, -1, +1, +1, -1, -1, +1]),
        ([1, -0.5, 0],   [+1, -1, -1, +1, +1, +1, -1, +1]),
        ([2.5, 0.5, 0],  [-1, -1, +1, -1, +1, +1, +1, -1]),
        ([4, -0.5, 0],   [-1, +1, +1, -1, +1, +1, +1, -1]),
        ([5.5, 0.5, 0],  [-1, -1, +1, -1, +1, -1, +1, -1]),
        ([7, -0.5, 0],   [-1, -1, -1, -1, -1, -1, -1, +1]),
        ([[-6, 0, 0], [-3.5, 0, 0], [-3.5, 1.5, 0], [-2.5, 1.5, 0],
          [-1, 2.5, 0], [1, -1.5, 0], [1, -0.5, 0], [2.5, 0.5, 0],
          [4, -0.5, 0], [5.5, 0.5, 0], [7, -0.5, 0]],
         [[-1, +1, -1, +1, -1, +1, +1, -1, -1, -1, -1],
          [-1, -1, -1, -1, -1, -1, -1, -1, +1, -1, -1],
          [-1, -1, -1, -1, -1, -1, -1, +1, +1, +1, -1],
          [-1, +1, -1, +1, -1, +1, +1, -1, -1, -1, -1],
          [-1, -1, -1, +1, -1, +1, +1, +1, +1, +1, -1],
          [-1, +1, -1, -1, -1, -1, +1, +1, +1, -1, -1],
          [-1, -1, -1, -1, -1, -1, -1, +1, +1, +1, -1],
          [+1, +1, +1, +1, +1, +1, +1, -1, -1, -1, +1]]
         )
    ])
    def test_points(self, geometry, geom_no, point, ans):
        ans = ans[geom_no]
        result = geometry[geom_no].test_points(point)
        np.testing.assert_array_equal(result, ans)

    @pytest.mark.parametrize('case_no, expected', enumerate([
        5, 2, 2, 13, 4, 6, 4, 4
    ]))
    def test_complexity(self, geometry, case_no, expected):
        assert geometry[case_no].complexity() == expected

    box_data = [
        [[1.25, 1.75, 1.5], 2.5, 3.5, 3],
        [[-3.25, -1.5, 1.5], 2.5, 3, 3],
        [[5.5, -0.75, 0.75], 2, 1.5, 1.5]
    ]

    @pytest.fixture(scope='class')
    def box(self):
        boxes = [Box(*b) for b in self.box_data]
        return boxes

    @pytest.mark.parametrize('box_no', range(len(box_data)))
    @pytest.mark.parametrize('case_no, expected', enumerate([
        (0, 0, -1),
        (-1, -1, 0),
        (0, -1, 0),
        (0, 0, -1),
        (0, 0, 0),
        (0, 0, 0)
    ]))
    def test_box(self, geometry, box, box_no, case_no, expected):
        result = geometry[case_no].test_box(box[box_no])
        assert result == expected[box_no]

    @pytest.mark.slow
    @pytest.mark.parametrize('case_no, expected', enumerate([
        [[-4, 4], [-2, 2], [-2, 2]],
        [[3, 5], [-1, 1], [-1, 1]],
        [[2, 6], [-2, 2], [-2, 2]],
        [[-4, 4], [-2, 2], [-2, 2]],
        [[-3, 6], [-2, 2], [-2, 2]],
        [[-4, 5], [-1, 1], [-1, 1]],
        [[2, 6], [-2, 2], [-2, 2]]
    ]))
    def test_bounding_box(self, geometry, case_no, expected):
        tol = 0.2
        base = [0, 0, 0]
        dims = [30, 30, 30]
        gb = Box(base, dims[0], dims[1], dims[2])
        bb = geometry[case_no].bounding_box(gb, tol)
        for j, (low, high) in enumerate(expected):
            bbdim = 0.5 * bb.dimensions[j]
            assert bb.center[j] - bbdim <= low
            assert bb.center[j] - bbdim >= low - tol
            assert bb.center[j] + bbdim >= high
            assert bb.center[j] + bbdim <= high + tol

    @pytest.mark.slow
    @pytest.mark.parametrize('box_no', range(len(box_data)))
    @pytest.mark.parametrize('case_no, expected', enumerate([
        [7.4940,  3.6652,  0],
        [0,       0,   0.1636],
        [0.35997, 0,   2.3544],
        [7.4940,  3.6652,  0],
        [7.8540,  3.1416, 2.3544],
        [1.9635,  1.30900, 0.1636]
    ]))
    def test_volume(self, geometry, box, box_no, case_no, expected):
        v = geometry[case_no].volume(box[box_no], min_volume=1.e-4)
        assert v == pytest.approx(expected[box_no], rel=1.e-2)

    @pytest.mark.parametrize('case_no, expected', enumerate([
        [2, 3, 1, 5, 4],
        [6, 1],
        [6, 1],
        [6, 1, 2, 5, 3, 4, 9, 7, 10],
        [5, 3, 2, 1],
        [3, 8, 5, 4, 6, 2],
        [1, 5],
        [1, 5]
    ]))
    def test_get_surface(self, geometry, surfaces, case_no, expected):
        expected = set(surfaces[s] for s in expected)
        surfs = geometry[case_no].get_surfaces()
        assert surfs == expected


class TestBody:
    kwarg_data = [
        {'name': 1},
        {'name': 2, 'MAT': Material(atomic=[('C-12', 1)], density=3.5)},
        {'name': 3, 'U': 4},
        {'name': 4, 'U': 5, 'MAT': Material(atomic=[('C-12', 1)], density=2.7)}
    ]

    @pytest.mark.parametrize('case_no', range(len(basic_geoms)))
    @pytest.mark.parametrize('kwargs', kwarg_data)
    def test_create(self, geometry, case_no, kwargs):
        shape = geometry[case_no]
        body = Body(shape, **kwargs)
        assert body.shape == shape
        for k, v in kwargs.items():
            assert body[k] == v
        assert body.material() == kwargs.get('MAT', None)

    @pytest.mark.parametrize('case_no, polish', enumerate(TestShape.polish_cases))
    def test_create_polish(self, geometry, surfaces, case_no, polish):
        polish = [TestShape.filter_arg(a, surfaces) for a in polish]
        body = Body(polish)
        assert body.shape == geometry[case_no]

    @pytest.mark.parametrize('kwargs', kwarg_data)
    @pytest.mark.parametrize('no1, no2',
        [(i, j) for i in range(len(basic_geoms)) \
                for j in range(len(basic_geoms)) if i != j]
    )
    def test_intersection(self, geometry, no1, no2, kwargs):
        body1 = Body(geometry[no1], **kwargs)
        body2 = Body(geometry[no2], name='1001')
        body = body1.intersection(body2)
        assert body.shape == body1.shape.intersection(body2.shape)
        for k, v in kwargs.items():
            assert body[k] == v

    @pytest.mark.parametrize('kwargs', kwarg_data)
    @pytest.mark.parametrize('no1, no2',
        [(i, j) for i in range(len(basic_geoms)) \
                for j in range(len(basic_geoms)) if i != j]
    )
    def test_union(self, geometry, no1, no2, kwargs):
        body1 = Body(geometry[no1], **kwargs)
        body2 = Body(geometry[no2], name='1001')
        body = body1.union(body2)
        assert body.shape == body1.shape.union(body2.shape)
        for k, v in kwargs.items():
            assert body[k] == v

    @pytest.mark.slow
    @pytest.mark.parametrize('kwarg', kwarg_data)
    @pytest.mark.parametrize('case_no, expected', enumerate([
        [2, 'C', 3, 'I', 1, 'I', 5, 'C', 'I', 4, 'C', 'U'],
        [6, 'C'],
        [1, 'C'],
        [2, 'C', 3, 'I', 1, 'I', 5, 'C', 'I', 4, 'C', 'U'],
        [5, 'C', 3, 'I', 2, 'C', 'I', 1, 'C', 'U'],
        [3, 8, 'C', 'I', 5, 'C', 'I', 4, 'C', 'U', 6, 'C', 'U'],
        [4, 'C'],
        [4]
    ]))
    def test_simplify(self, geometry, surfaces, case_no, expected, kwarg):
        expected = [TestShape.filter_arg(a, surfaces) for a in expected]
        expected_shape = Shape.from_polish_notation(expected)
        body = Body(geometry[case_no], **kwarg)
        gb = Box([3, 0, 0], 26, 20, 20)
        simple_body = body.simplify(min_volume=0.001, box=gb)
        assert simple_body.shape == expected_shape
        for k, v in kwarg.items():
            assert simple_body[k] == v
        assert simple_body.material() == kwarg.get('MAT', None)

    @pytest.mark.parametrize('fill_tr', [
        None,
        Transformation(translation=[2, -1, -0.5]),
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
    @pytest.mark.parametrize('case_no', range(len(basic_geoms)))
    def test_transform(self, geometry, tr, case_no, fill_tr):
        # The idea is to generate many random points. This points have some
        # definite test results with respect to the body being tested.
        # After transformation they must have absolutely the same results.
        points = np.random.random((10000, 3))
        points -= np.array([0.5, 0.5, 0.5])
        points *= np.array([20, 10, 10])

        if fill_tr is not None:
            fill = {'transform': fill_tr}
            points1 = fill_tr.apply2point(points)
        else:
            fill = None
            points1 = points
        body = Body(geometry[case_no], FILL=fill)
        results = body.shape.test_points(points1)

        new_body = body.transform(tr)
        if fill_tr:
            points2 = new_body['FILL']['transform'].apply2point(points)
        else:
            points2 = tr.apply2point(points)
        new_results = new_body.shape.test_points(points2)

        np.testing.assert_array_equal(results, new_results)

    @pytest.mark.skip
    def test_print(self):
        raise NotImplementedError

    def test_fill(self):
        raise NotImplementedError
