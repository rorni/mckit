# -*- coding: utf-8 -*-

"""
Разложение модели на составляющие юниверсы.

Читает модель, извлекает юниверсы первого уровня и сохраняет их в каталог universes под именем uN.i, где N - номер юниверса.
Также сохраняет общую модель (без юниверсов) под именем envelopes.i
Создает спецификацию для сборки модели в виде TOML файла.

"""
from datetime import datetime
import logging
from pathlib import Path
import click
import tomlkit as tk
import mckit as mk
from .common import save_mcnp, MCNP_ENCODING


def get_default_output_directory(source):
    return Path(Path(source).with_suffix(".universes").name)


def move_universe_attribute_to_comments(universe):
    no = universe.name()
    for cell in universe:
        cell.options.pop('U', None)
        comm = cell.options.get('comment', [])
        comm.append(f'U={no}')
        cell.options['comment'] = comm


def decompose(output, fill_descriptor_path, source, override):
    logger = logging.getLogger(__name__)
    logger.debug("Loading model from %s", source)
    source = Path(source)
    if output is None:
        output = get_default_output_directory(source)
    else:
        output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model: mk.Universe = mk.read_mcnp(source, encoding=MCNP_ENCODING)
    fill_descriptor = tk.document()
    fill_descriptor.add(tk.comment(f"This is a decomposition of \"{source.name}\" model"))
    if model.comment:
        fill_descriptor.append("comment", "model.comment")
    fill_descriptor.append("created", datetime.now())
    fill_descriptor.add(tk.nl())
    already_processed_universes = set()
    for c in model:
        fill = c.options.pop('FILL', None)
        if fill:
            universe = fill['universe']
            transform = fill.get('transform', None)
            words = [f'FILL={universe.name()}']
            if transform:
                words[0] = '*' + words[0]
                words.append('(')
                words.extend(transform.get_words())
                words.append(')')
            comm = c.options.get('comment', [])
            comm.append(''.join(words))
            c.options['comment'] = comm
            descriptor = tk.table()
            universe_name = universe.name()
            fn = f'u{universe_name}.i'
            descriptor['universe'] = universe_name
            if transform:
                name = transform.name()
                if name is None:
                    # The transformation is anonymous, so, store it's specification
                    # omitting redundant '*', TR0 words, and interleaving space tokens
                    descriptor['transform'] = tk.array(transform.mcnp_words()[2:][1::2])
                else:
                    descriptor['transform'] = name
            descriptor['file'] = fn
            fill_descriptor.append(str(c.name()), descriptor)
            fill_descriptor.add(tk.nl())
            if universe_name not in already_processed_universes:
                move_universe_attribute_to_comments(universe)
                save_mcnp(universe, output / fn, override)
                logger.debug("The universe %s are saved to %s", universe_name, fn)
                already_processed_universes.add(universe_name)

    with open(output / fill_descriptor_path, "w") as fid:
        res = tk.dumps(fill_descriptor)
        fid.write(res)
    envelopes_path = output / "envelopes.i"
    save_mcnp(model, envelopes_path, override)
    logger.debug("The envelopes are saved to %s", envelopes_path)





