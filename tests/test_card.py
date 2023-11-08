from __future__ import annotations

import pytest

from mckit.card import Card


class DummyCard(Card):
    def mcnp_words(self, pretty=False):
        return [f"{k}: {v}" for k, v in self.options]


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ({"a": 1}, {}, False),
        ({"a": 1}, {"a": 1}, True),
        ({"a": 1}, {"b": 1}, False),
        ({"a": 1}, {"a": 1, "b": 2}, False),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}, True),
        ({"a": 1, "b": 2}, {"a": 1}, False),
        ({"a": 1}, {"a": 2}, False),
    ],
)
def test_eq(a, b, expected):
    assert a == a
    assert b == b
    da, db = DummyCard(**a), DummyCard(**b)
    actual = da == db
    assert actual == expected
    assert (hash(da) == hash(db)) == expected
