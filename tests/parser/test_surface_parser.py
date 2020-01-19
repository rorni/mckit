import pytest
from mckit.parser.common import IgnoringIndex
import mckit.parser.common.transformation_index as ti
import mckit.parser.surface_parser as srp
from mckit.surface import (
    create_surface
)


@pytest.mark.parametrize("text,expected", [
    (
            "1 PX 0",
            [
                ("INTEGER", 1),
                ("SURFACE_TYPE", "PX"),
                ("INTEGER", 0),
            ]
    ),
    (
            "+2 py 2",
            [
                ('MODIFIER', '+'),
                ("INTEGER", 2),
                ("SURFACE_TYPE", "PY"),
                ("INTEGER", 2),
            ]
    ),
])
def test_surface_lexer(text, expected):
    lexer = srp.Lexer()
    actual = [(f.type, f.value) for f in lexer.tokenize(text)]
    assert actual == expected


# noinspection PyTypeChecker
@pytest.mark.parametrize("text,expected", [
    ("1 PX 0", create_surface("PX", 0.0, name=1)),
    ("*1 P 1.5 1.4 1.3 1.2", create_surface("P", 1.5, 1.4, 1.3, 1.2, name=1, modifier='*')),
])
def test_good_path(text, expected):
    actual = srp.parse(text)
    assert actual.mcnp_words() == expected.mcnp_words()


# noinspection PyTypeChecker
@pytest.mark.parametrize("text,expected", [
    ("1 2 PX 0", create_surface("PX", 0.0)),
])
def test_absent_surface_with_ignore_strategy(text, expected):
    actual = srp.parse(text, transformations=IgnoringIndex())
    assert actual == expected
    assert 'transform' not in actual.options


@pytest.mark.parametrize("text,msg_contains", [
    ("1 2 PX 0", 2),
])
def test_absent_surface_with_raise_strategy(text, msg_contains):
    with pytest.raises(KeyError, match=f"Transformation #{msg_contains} is not found"):
        srp.parse(
            text,
            transformations=ti.TransformationStrictIndex()
        )


# noinspection PyTypeChecker
@pytest.mark.parametrize("text,expected", [
    ("1 2 PX 0.1", create_surface("PX", 0.1, transform=ti.DummyTransformation(2))),
    ("1 1 SX 4 +5.0", create_surface("SX", 4, 5.0, name=1, transform=ti.DummyTransformation(1))),
])
def test_absent_surface_with_dummy_strategy(text, expected):
    actual = srp.parse(
        text,
        transformations=ti.TransformationDummyIndex(),
    )
    assert actual == expected


if __name__ == '__main__':
    pytest.main()
