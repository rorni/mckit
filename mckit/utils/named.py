from __future__ import annotations

from pprint import pformat
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import NewType
from typing import Optional
from typing import Text
from typing import TypeVar
from typing import cast

Name = NewType("Name", int)
"""Our names are integer"""

T = TypeVar("T")
"""Abstract item type"""

TContainer = Iterable
"""Items container"""

TStrategy = Callable[[Name, str, TContainer[T], T, T], T]
"""A strategy should either select a proper object to store in the index or raise an exception."""


def default_name_key(x) -> Name:
    return cast(Name, x.name())


class DuplicateError(ValueError):
    """Attempt to insert duplicated key to index of named entities"""


def default_get_container_tag_strategy(a: Any):
    """By default just print the container"""
    return pformat(a)


def raise_exception_on_clash_strategy(
    name: Name,
    container_tag: str,
    container: TContainer[T],
    old_entity: T,
    new_entity: T,
):
    msg = f"\nname {name} is duplicated in {container_tag}"
    raise DuplicateError(msg)


def raise_when_not_equal_strategy(
    name: Name,
    container_tag: Text,
    container,
    old_entity,
    new_entity,
):
    if old_entity == new_entity:
        return old_entity
    msg = f"\nname {old_entity} is duplicated with {new_entity}in {container_tag}"
    raise DuplicateError(msg)


def build_index_of_named_entities(
    container: TContainer[T],
    *,
    key: Callable[[T], Name] = default_name_key,
    container_tag_strategy: Callable[
        [TContainer[T]], str
    ] = default_get_container_tag_strategy,
    on_clash_strategy: TStrategy = raise_exception_on_clash_strategy,
    _filter: Optional[Callable[[T], bool]] = None,
) -> Dict[Name, T]:
    result: Dict[Name, T] = dict()
    for c in container:
        if not _filter or _filter(c):
            i = key(c)
            if i in result:
                selected = on_clash_strategy(
                    i,
                    container_tag_strategy(container),
                    container,
                    result[i],
                    c,
                )
                if selected is not c:
                    result[i] = selected
            else:
                result[i] = c
    return result
