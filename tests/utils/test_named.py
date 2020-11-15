from typing import NamedTuple

import pytest

import mckit.utils.named as nm


class Something(NamedTuple):
    name: int


@pytest.mark.parametrize(
    "entities, expected",
    [
        ([], {}),
        (
            list(map(Something, range(1, 3))),
            dict((k, Something(k)) for k in range(1, 3)),
        ),
    ],
)
def test_build_index(entities, expected):
    actual = nm.build_index_of_named_entities(entities, key=lambda x: x.name)
    assert actual == expected


@pytest.mark.parametrize(
    "entities",
    [
        (list(map(Something, [1, 1]))),
    ],
)
def test_clashes(entities):
    # try:
    #     actual = nm.build_index_of_named_entities(entities, key=lambda x: x.name)
    # except nm.DuplicateError as ex:
    #     pass
    with pytest.raises(nm.DuplicateError, match="Something"):
        actual = nm.build_index_of_named_entities(entities, key=lambda x: x.name)
        assert not actual


@pytest.mark.parametrize(
    "entities, expected",
    [
        (list(map(Something, [1, 1])), {1: Something(1)}),
    ],
)
def test_equal_duplicates(entities, expected):
    actual = nm.build_index_of_named_entities(
        entities,
        key=lambda x: x.name,
        on_clash_strategy=nm.raise_when_not_equal_strategy,
    )
    assert actual == expected


class Something2(NamedTuple):
    name: nm.Name
    value: int


@pytest.mark.parametrize(
    "entities",
    [
        [Something2(1, 1), Something2(1, 2)],
    ],
)
def test_clashes_on_non_equal_items(entities):
    with pytest.raises(nm.DuplicateError, match="Something2"):
        actual = nm.build_index_of_named_entities(
            entities,
            key=lambda x: x.name,
            on_clash_strategy=nm.raise_when_not_equal_strategy,
        )
        assert not actual
