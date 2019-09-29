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
from .common import save, MCNP_ENCODING


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
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model: mk.Universe = mk.read_mcnp(source, encoding=MCNP_ENCODING)
    fill_descriptor = tk.document()
    fill_descriptor.add(tk.comment(f"This is a decomposition of \"{Path(source).name}\" model"))
    if model.comment:
        fill_descriptor.append("comment", "model.comment")
    fill_descriptor.append("created", datetime.now())
    fill_descriptor.add(tk.nl())
    for c in model:
        fill = c.options.pop('FILL', None)
        if fill:
            un = fill['universe']
            tr = fill.get('transform', None)
            words = [f'FILL={un.name()}']
            if tr:
                words[0] = '*' + words[0]
                words.append('(')
                words.extend(tr.get_words())
                words.append(')')
            comm = c.options.get('comment', [])
            comm.append(''.join(words))
            c.options['comment'] = comm
            descriptor = tk.table()
            universe_name = un.name()
            fn = f'u{universe_name}.i'
            descriptor['universe'] = universe_name
            if tr:
                descriptor['transform'] = tr.name()
            descriptor['file'] = fn
            fill_descriptor.append(str(c.name()), descriptor)
            fill_descriptor.add(tk.nl())
            move_universe_attribute_to_comments(un)
            save(un, output_dir / fn, override)
            logger.debug("The universe %s are saved to %s", universe_name, fn)

    with open(output_dir / fill_descriptor_path, "w") as fid:
        res = tk.dumps(fill_descriptor)
        fid.write(res)
    envelopes_path = output_dir / "envelopes.i"
    save(model, envelopes_path, override)
    logger.debug("The envelopes are saved to %s", envelopes_path)





