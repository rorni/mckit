# -*- coding: utf-8 -*-

"""
Сборка модели из конвертов и входяших в них юниверсов по заданной спецификации.

"""
from functools import reduce
from typing import Dict, Optional
from pathlib import Path

import numpy as np
import tomlkit as tk
from loguru import logger
from tomlkit import items as tk_items

import mckit as mk
from mckit.parser.mcnp_input_sly_parser import from_file, ParseResult
from .common import save_mcnp
from mckit import Transformation
from mckit.utils import filter_dict


def compose(output, fill_descriptor_path, source, override):
    logger.debug("Loading model from %s", source)
    parse_result: ParseResult = from_file(source)
    envelopes = parse_result.universe
    source = Path(source)
    universes_dir = source.absolute().parent
    assert universes_dir.is_dir()
    logger.debug("Loading fill-descriptor from %s", fill_descriptor_path)
    with fill_descriptor_path.open() as fid:
        fill_descriptor = tk.parse(fid.read())

    universes = load_universes(fill_descriptor, universes_dir)
    named_transformations = load_named_transformations(fill_descriptor)

    comps = {}

    for k, v in universes.items():
        u, _ = v
        cps = u.get_compositions()
        comps[k] = {c for c in cps}

    common = reduce(set.union, comps.values())
    envelopes.set_common_materials(common)
    cells_index = dict((cell.name(), cell) for cell in envelopes)

    for i, spec in universes.items():
        universe, transformation = spec
        universe.set_common_materials(common)
        cell = cells_index[i]
        cell.options = filter_dict(cell.options, "original")
        cell.options["FILL"] = {"universe": universe}
        if transformation is not None:
            if isinstance(transformation, tk_items.Array):
                transformation1 = np.fromiter(
                    map(float, iter(transformation)), dtype=np.double
                )
                try:
                    translation = transformation1[:3]
                    if len(transformation1) > 3:
                        rotation = transformation1[3:]
                    else:
                        rotation = None
                    transformation2 = mk.Transformation(
                        translation=translation,
                        rotation=rotation,
                        indegrees=True,  # Assuming that on decompose we store a transformation in degrees as well
                    )
                except ValueError as ex:
                    raise ValueError(
                        f"Failed to process FILL transformation in cell #{cell.name()} of universe #{universe.name()}"
                    ) from ex
                cell.options["FILL"]["transform"] = transformation2
            elif isinstance(transformation, tk_items.Integer):
                assert (
                    named_transformations is not None
                ), "There are no named transformations in the fill descriptor file"
                transformation1 = named_transformations[int(transformation)]
                cell.options["FILL"]["transform"] = transformation1
            else:
                raise NotImplementedError(
                    f"Unexpected type of transformation parameter {type(transformation)}"
                )
    save_mcnp(envelopes, output, override)


def load_universes(fill_descriptor, universes_dir):
    universes = {}
    for k, v in fill_descriptor.items():
        if isinstance(v, dict) and "universe" in v:
            cell_name = int(k)
            universe_name = int(v["universe"])
            transformation = v.get("transform", None)
            universe_path = Path(v["file"])
            if not universe_path.exists():
                universe_path = universes_dir / universe_path
                if not universe_path.exists():
                    raise FileNotFoundError(universe_path)
            logger.debug("Loading universe from file '%s'", universe_path)
            parse_result: ParseResult = from_file(universe_path)
            universe: mk.Universe = parse_result.universe
            universe.rename(name=universe_name)
            universes[cell_name] = (universe, transformation)
    return universes


def load_named_transformations(fill_descriptor) -> Optional[Dict[int, Transformation]]:
    transformations = fill_descriptor.get("named_transformations", None)
    if transformations:
        named_transformations = {}
        for k, v in transformations.items():
            name = int(k[2:])
            transform_params = np.fromiter(map(float, v), dtype=np.float)
            translation = transform_params[:3]
            if transform_params.size == 9:
                rotation = transform_params[3:]
            else:
                rotation = None
            # noinspection PyTypeChecker
            transform = Transformation(
                translation=translation, rotation=rotation, indegrees=True, name=name
            )
            named_transformations[name] = transform
        return named_transformations
    else:
        return None
