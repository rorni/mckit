from typing import Callable, Dict, Iterable, NoReturn, Optional, Type, TypeVar, cast

from functools import reduce

from mckit.utils.named import Name, default_name_key

Key = TypeVar("Key")
Item = TypeVar("Item")


class Index(Dict[Key, Item]):
    """Like collections.defaultdict but the factory takes key as argument."""

    def __init__(
        self, default_factory: Callable[[Key], Optional[Item]] = None, **kwargs
    ):
        super().__init__(self, **kwargs)
        self._default_factory = default_factory

    def __missing__(self, key: Key) -> Optional[Item]:
        """
        Calls default factory with the key as argument.
        """
        if self._default_factory:
            df = self._default_factory
            return df(key)

        raise NumberedItemNotFoundError(key)

    @property
    def default_factory(self):
        return self._default_factory


# noinspection PyUnusedLocal
def ignore(item: Key) -> Optional[Item]:
    return None


class IgnoringIndex(Index[Key, Item]):
    def __init__(self) -> None:
        Index.__init__(self, ignore)


class NumberedItemNotFoundError(KeyError):
    kind: str = ""

    def __init__(self, item: Key, *args, **kwargs) -> None:
        super().__init__(args, kwargs)
        self._item = item

    def __str__(self) -> str:
        return f"{self.kind} #{self._item} is not found"


class NumberedItemDuplicateError(ValueError):
    kind: str = ""

    def __init__(self, item: Key, prev: Item, curr: Item, *args, **kwargs) -> None:
        super().__init__(args, kwargs)
        self._item = item
        self._prev = prev
        self._curr = curr

    def __str__(self):
        return f"{self.kind} #{self._item} is duplicated, see {self._prev} and {self._curr}"  # NOQA


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
        **kwargs,
    ) -> "IndexOfNamed[Key, Item]":
        index = cls(default_factory, **kwargs)

        def reducer(a, b):
            name = key(b)
            if on_duplicate is not None:
                if name in a:
                    on_duplicate(name, a[name], b)
            a[name] = b
            return a

        return cast("IndexOfNamed[Key, Item]", reduce(reducer, items, index))
