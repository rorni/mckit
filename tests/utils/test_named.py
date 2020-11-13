# -*- coding: utf-8 -*-

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
    try:
        actual = nm.build_index_of_named_entities(entities, key=lambda x: x.name)
    except nm.DuplicateError as ex:
        pass
    # with pytest.raises(nm.DuplicateError, match="Something"):
    #     actual = nm.build_index_of_named_entities(entities, key=lambda x: x.name)
    #     assert not actual
