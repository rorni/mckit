# -*- coding: utf-8 -*-

"""
Разложение модели на составляющие юниверсы.

Читает модель, извлекает юниверсы первого уровня и сохраняет их
в каталог universes под именем uN.i, где N - номер юниверса.
Также сохраняет общую модель (без юниверсов) под именем envelopes.i
Создает спецификацию для последующей сборки модели в виде TOML файла.

"""
from datetime import datetime
from pathlib import Path

import tomlkit as tk

from mckit import Universe
from mckit.parser.mcnp_input_sly_parser import ParseResult, from_file
from mckit.universe import collect_transformations
from mckit.utils.logging import logger
from tomlkit.items import item

from .common import save_mcnp


def get_default_output_directory(source):
    return Path(Path(source).with_suffix(".universes").name)


def move_universe_attribute_to_comments(universe):
    no = universe.name()
    for cell in universe:
        cell.options.pop("U", None)
        comm = cell.options.get("comment", [])
        comm.append(f"U={no}")
        cell.options["comment"] = comm


def decompose(output, fill_descriptor_path, source, override):
    logger.debug("Loading model from {}", source)
    source = Path(source)
    if output is None:
        output = get_default_output_directory(source)
    else:
        output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fill_descriptor = tk.document()
    fill_descriptor.add(tk.comment(f'This is a decomposition of "{source.name}" model'))
    parse_result: ParseResult = from_file(source)
    if parse_result.title:
        fill_descriptor.append("title", parse_result.title)
    model: Universe = parse_result.universe
    if model.comment:
        fill_descriptor.append("comment", model.comment)
    named_transformations = list(collect_transformations(model))
    fill_descriptor.append("created", item(datetime.now()))
    fill_descriptor.add(tk.nl())
    already_processed_universes = set()
    for c in model:
        fill = c.options.pop("FILL", None)
        if fill:
            universe = fill["universe"]
            words = [f"FILL={universe.name()}"]
            transform = fill.get("transform", None)
            if transform:
                words[0] = "*" + words[0]
                words.append("(")
                words.extend(transform.get_words())
                words.append(")")
            comm = c.options.get("comment", [])
            comm.append("".join(words))
            c.options["comment"] = comm
            descriptor = tk.table()
            universe_name = universe.name()
            fn = f"u{universe_name}.i"
            descriptor["universe"] = universe_name
            if transform:
                name = transform.name()
                if name is None:
                    # The transformation is anonymous, so, store it's specification
                    # omitting redundant '*', TR0 words, and interleaving space tokens
                    descriptor["transform"] = tk.array(transform.mcnp_words()[2:][1::2])
                else:
                    descriptor["transform"] = name
            descriptor["file"] = fn
            fill_descriptor.append(str(c.name()), descriptor)
            fill_descriptor.add(tk.nl())
            if universe_name not in already_processed_universes:
                move_universe_attribute_to_comments(universe)
                save_mcnp(universe, output / fn, override)
                logger.debug("The universe {} has been saved to {}", universe_name, fn)
                already_processed_universes.add(universe_name)

    named_transformations_descriptor = tk.table()
    named_transformations = sorted(named_transformations, key=lambda x: x.name())
    for t in named_transformations:
        named_transformations_descriptor[f"tr{t.name()}"] = tk.array(
            t.mcnp_words()[2:][1::2]
        )
    fill_descriptor.append("named_transformations", named_transformations_descriptor)
    fill_descriptor.add(tk.nl())

    fdp = output / fill_descriptor_path
    with open(fdp, "w") as fid:
        res = tk.dumps(fill_descriptor)
        fid.write(res)
    logger.debug("Fill descriptor is saved in {}", fdp)
    envelopes_path = output / "envelopes.i"
    save_mcnp(model, envelopes_path, override)
    logger.debug("The envelopes are saved to {}", envelopes_path)
