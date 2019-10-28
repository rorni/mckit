# -*- coding: utf-8 -*-

import pytest
import mckit.parser.material_parser as mp
from mckit.material import Composition, Element


@pytest.mark.parametrize("text, expected_types, expected_values", [
    (
        "1001.21c -1.0",
        ['INTEGER', 'LIB', 'FLOAT'],
        [1001, '21c', -1.0],
    ),
    (
        """
        1001.21c -1.0  $ comment
        """.strip(),
        ['INTEGER', 'LIB', 'FLOAT', 'EOL_COMMENT'],
        [1001, '21c', -1.0, '$ comment'],
    ),
    (
        "  M100 1000",
        ['NAME', 'INTEGER'],
        [100, 1000],
    ),
])
def test_test_composition_lexer(text, expected_types, expected_values):
    lexer = mp.Lexer()
    tokens = list(t for t in lexer.tokenize(text))
    result = list(t.type for t in tokens)
    assert result == expected_types
    result = list(t.value for t in tokens)
    assert result == expected_values



@pytest.mark.parametrize("text, expected", [
    (
        """m1
        1001.21c -1.0
        """,
        Composition(weight=[(1001, 1.0)], name=1)
    ),
    (
        """m1000
        1001.21c -1.0 $ eol comment
        """,
        Composition(weight=[(Element(1001, lib='21c', comment='$ eol comment'), 1.0)], name=1000)
    ),
    (
        """M1000
        1001.21c -1.0 
            $ trailing comment1
            $ trailing comment2
        """,
        Composition(
            weight=[(Element(1001, lib='21c'), 1.0)],
            name=1000,
            comment=['$ trailing comment1', '$ trailing comment2']
        )
    ),
    (
        """M1000
    1001.21c -1.0
c bzzzzzz 
        $ trailing comment1
        $ trailing comment2
        """,
        Composition(
            weight=[(Element(1001, lib='21c'), 1.0)],
            name=1000,
            comment=['$ trailing comment1', '$ trailing comment2']
        )
    ),
    (
        """M1000
        1001.21c -1.0
            gas 1 
            $ trailing comment1
            $ trailing comment2
        """,
        Composition(
            weight=[(Element(1001, lib='21c'), 1.0)],
            name=1000,
            comment=['$ trailing comment1', '$ trailing comment2'],
            gas=1,
        )
    ),
])
def test_test_composition_parser(text, expected):
    result = mp.parse(text)
    assert isinstance(result, Composition), "Parser should create instance of Composition"
    assert result == expected
    assert result.options == expected.options
