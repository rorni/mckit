# -*- coding: utf-8 -*-

"""
Сборка модели из конвертов и входяших в них юниверсов по заданной спецификации.

"""
from pathlib import Path
from functools import reduce
import tomlkit as tk
import mckit as mk
from .common import save, MCNP_ENCODING


def compose(output, fill_descriptor_path, source, override):
    envelopes = mk.read_mcnp(source, encoding=MCNP_ENCODING)
    source = Path(source)
    universes_dir = source.absolute().parent
    assert universes_dir.is_dir()

    with fill_descriptor_path.open() as fid:
        fill_descriptor = tk.parse(fid.read())

    universes = {}

    for k, v in fill_descriptor.items():
        if isinstance(v, dict) and 'universe' in v:
            cell_name = int(k)
            # transformation = v['transform']
            universe_path = Path(v['file'])
            if not universe_path.exists():
                universe_path = universes_dir / universe_path
                if not universe_path.exists():
                    raise FileNotFoundError(universe_path)
            universe = mk.read_mcnp(universe_path, encoding=MCNP_ENCODING)
            universes[cell_name] = universe

    comps = {}

    for k, u in universes.items():
        cps = u.get_compositions()
        comps[k] = {c for c in cps}

    common = reduce(set.union, comps.values())
    envelopes.set_common_materials(common)
    cells_index = dict((cell.name(), cell) for cell in envelopes)

    for i, universe in universes.items():
        cell = cells_index[i]
        assert cell.name() == i, "Check indexes"
        universe.rename(name=i)
        universe.set_common_materials(common)
        cell.options["FILL"] = {"universe": universe}

    save(envelopes, output, override)
