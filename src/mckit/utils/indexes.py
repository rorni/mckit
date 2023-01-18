"""Classes to index MCNP objects on model file parsing."""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, NoReturn, Optional, Type, TypeVar, cast

from functools import reduce

from mckit.utils.named import Name, default_name_key

Key = TypeVar("Key")
Item = TypeVar("Item")
FactoryMethodWithKey = Callable[[Key], Optional[Item]]


class Index(Dict[Key, Item]):
    """Like collections.defaultdict but the factory takes key as argument.

    The class redefines __missing__ method to use `key` on calling factory method.

    Attrs:
        _default_factory (FactoryMethodWithKey):
            Same as in Args.
    """

    def __init__(
        self,
        default_factory: FactoryMethodWithKey = None,
        **kwargs: Dict[Key, Item],
    ) -> None:
        """Create `Index`.

        Args:
            default_factory: factory method accepting `key` as the only positional argument.
            kwargs: keyword arguments to pass to base class, if any
        """
        super().__init__(self, **kwargs)
        self._default_factory = default_factory

    @property
    def default_factory(self) -> FactoryMethodWithKey:
        """Public accessor to `self._default_factory`."""
        return self._default_factory

    def __missing__(self, key: Key) -> Optional[Item]:
        """Calls default factory with the key as argument."""
        if self._default_factory:
            return self._default_factory(key)

        raise NumberedItemNotFoundError(key)


# noinspection PyUnusedLocal
def ignore(_: Key) -> Optional[Item]:
    """Default factory for `IgnoringIndex`.

    Returns:
        None - always.
    """
    return None


class IgnoringIndex(Index[Key, Item]):
    """Index ignoring absence of a key in the dictionary."""

    def __init__(self, **kwargs: Dict[Key, Item]) -> None:
        Index.__init__(self, ignore, **kwargs)


class NumberedItemNotFoundError(KeyError):
    """Error to raise, when an item is not found in an `Index`."""

    kind: str = ""

    def __init__(self, key: Key, *args, **kwargs: Dict[Any, Any]) -> None:
        super().__init__(args, **kwargs)
        self._key = key

    def __str__(self) -> str:
        return f"{self.kind} #{self._key} is not found"


class NumberedItemDuplicateError(ValueError):
    """Error to raise, when an item has duplicate in an `Index`."""

    kind: str = ""

    def __init__(self, item: Key, prev: Item, curr: Item, *args, **kwargs) -> None:
        super().__init__(args, kwargs)
        self._item = item
        self._prev = prev
        self._curr = curr

    def __str__(self):
        return f"{self.kind} #{self._item} is duplicated, see {self._prev} and {self._curr}"


def raise_on_duplicate_strategy(key: Key, prev: Item, curr: Item) -> NoReturn:
    raise NumberedItemDuplicateError(key, prev, curr)


def ignore_equal_objects_strategy(key: Key, prev: Item, curr: Item) -> None:
    if prev is not curr and prev != curr:
        raise NumberedItemDuplicateError(key, prev, curr)


class StatisticsCollector(Dict[Key, int]):
    def __init__(self, ignore_equal=False):
        super().__init__(self)
        self.ignore_equal = ignore_equal

    def __missing__(self, key):
        return 1

    def __call__(self, key: Key, prev: Item, curr: Item) -> None:
        if not self.ignore_equal or prev != curr:
            self[key] += 1


class IndexOfNamed(Index[Key, Item]):
    @classmethod
    def from_iterable(
        cls: Type["IndexOfNamed[Key, Item]"],
        items: Iterable[Item],
        *,
        key: Callable[[Item], Name] = default_name_key,
        default_factory: Callable[[Key], Item] = None,
        on_duplicate: Callable[[Key, Item, Item], None] = None,
    ) -> "IndexOfNamed[Key, Item]":
        index = cls(default_factory)

        def reducer(a, b):
            name = key(b)
            if on_duplicate is not None and name in a:
                on_duplicate(name, a[name], b)
            a[name] = b
            return a

        return cast("IndexOfNamed[Key, Item]", reduce(reducer, items, index))
