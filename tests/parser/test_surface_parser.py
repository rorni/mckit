import pytest

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
    actual = srp.parse(text, transformations={})
    assert actual.mcnp_words() == expected.mcnp_words()


# noinspection PyTypeChecker
@pytest.mark.parametrize("text,expected", [
    ("1 2 PX 0", create_surface("PX", 0.0)),
])
def test_absent_surface_with_ignore_strategy(text, expected):
    actual = srp.parse(text)
    assert actual == expected
    assert 'transform' not in actual.options


@pytest.mark.parametrize("text,msg_contains", [
    ("1 2 PX 0", 2),
])
def test_absent_surface_with_raise_strategy(text, msg_contains):
    with pytest.raises(KeyError, match=f"Transformation {msg_contains} is not found"):
        srp.parse(
            text,
            transformations={1: srp.DummyTransformation(1)}
        )


# noinspection PyTypeChecker
@pytest.mark.parametrize("text,expected", [
    ("1 2 PX 0.1", create_surface("PX", 0.1, transform=srp.DummyTransformation(2))),
])
def test_absent_surface_with_dummy_strategy(text, expected):
    actual = srp.parse(
        text,
        transformations={1: srp.DummyTransformation(1)},
        on_absent_transformation=srp.dummy_on_absent_transformation_strategy
    )
    assert actual == expected


if __name__ == '__main__':
    pytest.main()
