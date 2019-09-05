# -*- coding: utf-8 -*-

"""
Разложение модели на составляющие юниверсы.

Читает C-Model, извлекает все юниверсы и сохраняет их в каталог universes под именем uN.i, где N - номер юниверса.
Также сохраняет общую модель (без юниверсов) под именем envelopes.i# -*- coding: utf-8 -*-

"""
import logging
from pathlib import Path

import mckit as mk


# This is the encoding swallowing non ascii (neither unicode) symbols happening in MCNP models code
MCNP_ENCODING = "Cp1251"


def delete_fill(universe):
    for c in universe:
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


def delete_universe(universe):
    for c in universe:
        c.options.pop('U', None)
        comm = c.options.get('comment', [])
        comm.append('U={0}'.format(universe.name()))
        c.options['comment'] = comm
            

def get_universes(universe):
    unames = universe.get_universes()
    result = {}
    for un in unames:
        inner = mk.universe.select(un)
        in_res = get_universes(inner)
        if in_res:
            print('>1')
        result.update(in_res)
        result[un] = inner
    return result


def decompose(ctx, source):
    logger = logging.getLogger(__name__)
    logger.debug("Loading model from %s", source)
    model = mk.read_mcnp(source, encoding=MCNP_ENCODING)
    universes = model.get_universes()
    universes = {u.name(): u for u in universes}
    delete_fill(model)
    output_dir = Path('universes')
    output_dir.mkdir(parents=True, exist_ok=True)
    envelops_path = output_dir / "envelops.i"
    model.save(envelops_path)
    logger.debug("The envelops are saved to %s", envelops_path)
    for name, univ in universes.items():
        delete_fill(univ)
        delete_universe(univ)
        univ.save(f'universes/u{name}.i')




