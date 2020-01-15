import pytest
import mckit.parser.cell_parser as crp
from mckit.body import (
    Shape, Body
)
from  mckit.material import Composition, Material

def create_cell(
    cell_no,
    geometry,
    material=None,
    transformation=None,
    **options
):
    pass

@pytest.mark.parametrize("text,expected_types,expected_values", [
    (
        "1 0",
        ["INTEGER", "ZERO"],
        [1, 0]
    ),
    (
        "1 1 -0.083",
        ["INTEGER", "INTEGER", "FLOAT"],
        [1, 1, -0.083]
    ),
])
def test_cell_lexer(text, expected_types, expected_values):
    lexer = crp.Lexer()
    tokens = list(lexer.tokenize(text))
    actual_types = list(f.type for f in tokens)
    actual_values = list(f.value for f in tokens)
    assert actual_types == expected_types
    assert actual_values == expected_values


@pytest.mark.parametrize("text,expected", [
    ("1 0 1", create_cell(1, 0,  0.0)),
])
def test_surface_parser(text, expected):
    actual = srp.parse(text, transformations={})
    assert actual == expected


if __name__ == '__main__':
    pytest.main()
