# -*- coding: utf-8 -*-
from typing import Dict, List, NoReturn

import pytest

from mckit.utils.indexes import Index


def dummy_strategy(c: int) -> int:
    return c * c


@pytest.mark.parametrize("inp", [{1: 1, 2: 4}, {}])
def test_index_with_dummy_strategy(inp: Dict[int, int]) -> None:
    dictionary = Index(dummy_strategy)
    dictionary.update(inp)
    for k in range(5):
        v = dictionary[k]
        expected = k * k
        assert expected == v


class MyKeyError(KeyError):
    def __init__(self, key: int, *args, **kwargs):
        super().__init__(args, kwargs)
        self._key = key

    @property
    def key(self):
        return self._key


def strict_strategy(c: int) -> NoReturn:
    raise MyKeyError(c)


@pytest.mark.parametrize(
    "inp,keys,success",
    [({1: 1, 2: 4}, [1, 2, 3], [True, True, False]), ({}, [1], [False])],
)
def test_index_with_strict_strategy(
    inp: Dict[int, int], keys: List[int], success: List[bool]
) -> None:
    dictionary = Index(strict_strategy)
    dictionary.update(inp)
    for k, s in zip(keys, success):
        try:
            _ = dictionary[k]
            assert s is True
        except MyKeyError as ex:
            assert ex.key == k
            assert s is False
