from typing import NamedTuple

import mckit.utils.named as nm
import pytest

from mckit.utils.Index import (
    IndexOfNamed,
    NumberedItemDuplicateError,
    StatisticsCollector,
    ignore_equal_objects_strategy,
    raise_on_duplicate_strategy,
)


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
def test_index_of_named_happy_path(entities, expected):
    actual = IndexOfNamed.from_iterable(entities, key=lambda x: x.name)
    assert actual == expected


@pytest.mark.parametrize(
    "entities",
    [
        (list(map(Something, [1, 1]))),
    ],
)
def test_clashes(entities):
    # try:
    #     actual = IndexOfNamed.from_iterable(entities, key=lambda x: x.name, on_duplicate=raise_on_duplicate_strategy)
    # except NumberedItemDuplicateError as ex:
    #     pass
    with pytest.raises(NumberedItemDuplicateError, match="Something"):
        actual = IndexOfNamed.from_iterable(
            entities, key=lambda x: x.name, on_duplicate=raise_on_duplicate_strategy
        )
        assert not actual


@pytest.mark.parametrize(
    "entities, expected",
    [
        (list(map(Something, [1, 1])), {1: Something(1)}),
    ],
)
def test_equal_duplicates(entities, expected):
    actual = IndexOfNamed.from_iterable(
        entities, key=lambda x: x.name, on_duplicate=ignore_equal_objects_strategy
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
    with pytest.raises(NumberedItemDuplicateError, match="Something2"):
        actual = IndexOfNamed.from_iterable(
            entities, key=lambda x: x.name, on_duplicate=raise_on_duplicate_strategy
        )
        assert not actual


@pytest.mark.parametrize(
    "entities, expected, expected_collected",
    [
        ([Something2(1, 1), Something2(1, 2)], {1: Something2(1, 2)}, {1: 2}),
        (
            [Something2(1, 1), Something2(1, 2), Something2(2, 3)],
            {1: Something2(1, 2), 2: Something2(2, 3)},
            {1: 2},
        ),
        (
            [Something2(1, 1), Something2(1, 2), Something2(2, 3), Something2(2, 4)],
            {1: Something2(1, 2), 2: Something2(2, 4)},
            {1: 2, 2: 2},
        ),
    ],
)
def test_collect_statistics_on_clashes(entities, expected, expected_collected):
    collector = StatisticsCollector()
    actual = IndexOfNamed.from_iterable(
        entities, key=lambda x: x.name, on_duplicate=collector
    )
    assert actual == expected
    assert collector == expected_collected


# @pytest.mark.parametrize(
#     "entities, expected, expected_collected",
#     [
#         ([Something2(1, 1), Something2(1, 2), Something2(2,3)], {1: Something2(1, 2)}, {1: 2, 2: 1}),
#     ],
# )
# def test_collect_statistics_on_clashes(entities, expected, expected_collected):
#     collector = StatisticsCollector()
#     actual = IndexOfNamed.from_iterable(
#         entities, key=lambda x: x.name, on_duplicate=collector
#     )
#     assert actual == expected
#     assert collector == expected_collected
