import pytest
import mckit.parser.cell_parser as crp
from mckit.body import (
    Shape
)

@pytest.mark.parametrize("text,expected", [
    (
        "1 0",
        [
            ("INTEGER", 1),
            ("INTEGER", 0),
        ]
    ),
])
def test_surface_lexer(text, expected):
    lexer = crp.Lexer()
    actual = [(f.type, f.value) for f in lexer.tokenize(text)]
    assert actual == expected


@pytest.mark.parametrize("text,expected", [
    ("1 0", create_cell(1, 0,  0.0)),
])
def test_surface_parser(text, expected):
    actual = srp.parse(text, transformations={})
    assert actual == expected


if __name__ == '__main__':
    pytest.main()
