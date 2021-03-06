# -*- coding: utf-8 -*-

"""
Сборка модели из конвертов и входяших в них юниверсов по заданной спецификации.

"""
import numpy as np
from pathlib import Path
from functools import reduce
import tomlkit as tk
import mckit as mk
from mckit.parser.mcnp_input_sly_parser import from_file, ParseResult
from .common import save_mcnp
from mckit.utils import filter_dict


def compose(output, fill_descriptor_path, source, override):
    parse_result: ParseResult = from_file(source)
    envelopes = parse_result.universe
    source = Path(source)
    universes_dir = source.absolute().parent
    assert universes_dir.is_dir()

    with fill_descriptor_path.open() as fid:
        fill_descriptor = tk.parse(fid.read())

    universes = {}

    for k, v in fill_descriptor.items():
        if isinstance(v, dict) and 'universe' in v:
            cell_name = int(k)
            universe_name = int(v['universe'])
            transformation = v.get('transform', None)
            universe_path = Path(v['file'])
            if not universe_path.exists():
                universe_path = universes_dir / universe_path
                if not universe_path.exists():
                    raise FileNotFoundError(universe_path)
            parse_result: ParseResult = from_file(universe_path)
            universe: mk.Universe = parse_result.universe
            universe.rename(name=universe_name)
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
        universe.set_common_materials(common)
        cell = cells_index[i]
        cell.options = filter_dict(cell.options, "original")
        cell.options["FILL"] = {"universe": universe}
        if transformation is not None:
            if isinstance(transformation, tk.array):
                transformation = np.fromiter(map(float, iter(transformation)), dtype=np.double)
                transformation = mk.Transformation(
                    translation=transformation[:3],
                    rotation=transformation[3:],
                    indegrees=True,
                )
                cell.options["FILL"]["transform"] = transformation
            else:
                # TODO dvp: use parse results to implement this: there's an index of transformations
                raise NotImplementedError(
                    """\
                    Specification of fill with a universe with a named transformation "fill=<...> ( number )" occurs. \
                    Only anonymous transformations are implemented.\
                    """
                )
    save_mcnp(envelopes, output, override)
