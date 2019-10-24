# -*- coding: utf-8 -*-

import pytest
import mckit.parser.material_parser as mp
from mckit.material import Composition


@pytest.mark.parametrize("text, expected_types, expected_values", [
    (
        """
        1001.21c -1.0
        """,
        ['INTEGER', 'LIB', 'FLOAT'],
        [1001, '21C', -1.0],
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
        """
        1001.21c -1.0
        """,
        Composition(weight=[(1001, -1.0)], lib="21c")
    ),
])
def test_test_composition_parser(text, expected):
    lexer = mp.Lexer()
    parser = mp.Parser()
    result = parser.parse(lexer.tokenize(text))
    assert result == expected
