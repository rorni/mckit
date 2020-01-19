# -*- coding: utf-8 -*-
"""
Проверяет корректность модели и выдает статистику.
"""
from typing import Any, Callable, Iterable, Optional
import logging
from pathlib import Path
from mckit import Universe, read_mcnp
from mckit.card import Card


def check_duplicates(
    iterable: Optional[Iterable[Any]],
    label: str,
    key: Callable[[Any], Any]
) -> None:
    # logger = logging.getLogger(__name__)
    if iterable is None:
        # logger.info("No %ss are found", label)
        print("No %ss are found"%label)
    else:
        visited = set()
        for c in iterable:
            k = key(c)
            if k in visited:
                # logger.error("Duplicate of %s %d is found", label, k)
                print("Duplicate of %s %d is found"%(label, k))
            else:
                visited.add(k)
        # logger.info("Total of %ss: %d", label, len(visited))
        print("Total of %ss: %d"%(label, len(visited)))


def check(source):
    result = 0
    logger = logging.getLogger(__name__)
    logger.debug("Check model %s", source)
    source = Path(source)
    model: Universe = read_mcnp(source)
    logger.debug("Read the model okay")
    universes = model.get_universes()
    check_duplicates(universes, "universe", Universe.name)
    cells = []
    surfaces = []
    transformations = []
    compositions = []
    for u in universes:
        cells.extend(u)
        surfaces.extend(u.get_surfaces(inner=False))
        transformations_to_add = u.get_transformations()
        if transformations_to_add:
            transformations.extend(transformations_to_add)
        compositions.extend(u.get_compositions())
    if not transformations:
        transformations = None
    if not compositions:
        compositions = None
    check_duplicates(cells, "cell", Card.name)
    check_duplicates(surfaces, "surface", Card.name)
    check_duplicates(transformations, "transformation", Card.name)
    check_duplicates(compositions, "composition", Card.name)
    return result






