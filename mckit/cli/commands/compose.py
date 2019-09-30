# -*- coding: utf-8 -*-

"""
Сборка модели из конвертов и входяших в них юниверсов по заданной спецификации.

"""
import numpy as np
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
            transformation = v.get('transform', None)
            universe_path = Path(v['file'])
            if not universe_path.exists():
                universe_path = universes_dir / universe_path
                if not universe_path.exists():
                    raise FileNotFoundError(universe_path)
            universe = mk.read_mcnp(universe_path, encoding=MCNP_ENCODING)
            universes[cell_name] = (universe, transformation)

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
        cell = cells_index[i]
        assert cell.name() == i, "Check indexes"
        universe.rename(name=i)
        universe.set_common_materials(common)
        cell.options["FILL"] = {"universe": universe}
        if transformation is not None:
            if isinstance(transformation, tk.array):
                transformation = np.fromiter(map(float, iter(transformation)), dtype=np.double)
                transformation = mk.Transformation(
                    translation=transformation[:3],
                    rotaion=transformation[3:],
                    indegrees=True,
                )
            else:
                raise NotImplementedError(
                    """\
                    Specification of fill with a universe with a named transformation "fill=<...> ( number )" occurs. \
                    Only anonymous transformations are implemented.\
                    """
                )

    save(envelopes, output, override)
