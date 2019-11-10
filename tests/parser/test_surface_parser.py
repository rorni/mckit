import pytest
import mckit.parser.surface_parser as srp
from mckit.surface import (
    Cone, Cylinder, Plane, Sphere, GQuadratic, Torus, create_surface
)

@pytest.mark.parametrize("text,expected", [
    (
        "PX 0",
        [
            ("SURFACE_TYPE", "PX"),
            ("INTEGER", 0),
        ]
    ),
])
def test_surface_lexer(text, expected):
    lexer = srp.Lexer()
    actual = [(f.type, f.value) for f in lexer.tokenize(text)]
    assert actual == expected


@pytest.mark.parametrize("text,expected", [
    ("PX 0", create_surface("PX", 0.0)),
])
def test_surface_parser(text, expected):
    actual = srp.parse(text, transformations={})
    assert actual == expected


if __name__ == '__main__':
    pytest.main()
