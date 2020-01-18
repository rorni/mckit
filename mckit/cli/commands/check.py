# -*- coding: utf-8 -*-

"""
Разложение модели на составляющие юниверсы.

Читает модель, извлекает юниверсы первого уровня и сохраняет их в каталог simple_cubes.universes под именем uN.i, где N - номер юниверса.
Также сохраняет общую модель (без юниверсов) под именем envelopes.i
Создает спецификацию для сборки модели в виде TOML файла.

"""
from datetime import datetime
import logging
from pathlib import Path
import click
import mckit as mk
from .common import save_mcnp, get_default_output_directory
from ...constants import MCNP_ENCODING


def check_duplicates(iterable, label, key):
    logger = logging.getLogger(__name__)
    visited = set()
    for c in iterable:
        k = key(c)
        if k in visited:
            logger.error("Duplicate of %s %d is found", label, k)
        else:
            visited.add(k)
    logger.info("Total of %ss: %d", label, len(visited))


def check(source, output, override):
    result = 0
    logger = logging.getLogger(__name__)
    logger.debug("Check model %s", source)
    source = Path(source)
    # if output is None:
    #     output = get_default_output_directory(source, ".check")
    # else:
    #     output = Path(output)
    # output.parent.mkdir(parents=True, exist_ok=True)
    model: mk.Universe = mk.read_mcnp(source)
    logger.debug("Read the model okay")
    check_duplicates(model, "cell", mk.Body.name)
    check_duplicates(model.get_surfaces(inner=True), "surface", lambda s: s.name())
    return result






