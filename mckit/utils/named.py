from __future__ import annotations

from io import StringIO

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    NamedTuple,
    NewType,
    NoReturn,
    Optional,
    Text,
)

Name = NewType("Name", int)


def default_name_key(x: Any) -> Name:
    res: Name = x.name()
    return res


class DuplicateError(ValueError):
    """Attempt to insert duplicated key to index of named entities"""


def default_get_container_tag_strategy(a: Any):
    return str(a)


def raise_exception_on_clash_strategy(
    name: Name,
    container_tag,
    *,
    container=None,
    old_entity=None,
    new_entity=None,
):
    msg = f""""
        name {name} with is duplicated in {container_tag} 
    """
    raise DuplicateError(msg)


def build_index_of_named_entities(
    container: Iterable[Any],
    *,
    key=default_name_key,
    container_tag_strategy=default_get_container_tag_strategy,
    on_clash_strategy=raise_exception_on_clash_strategy,
    _filter: Optional[Callable[[Any], bool]] = None,
) -> Dict[Name, Any]:
    result = dict()
    for c in container:
        if not _filter or _filter(c):
            i = key(c)
            if i in result:
                c = on_clash_strategy(
                    i,
                    container_tag_strategy(c),
                    container=container,
                    old_entity=result[i],
                    new_entity=c,
                )
            result[i] = c
    return result
