# -*- coding: utf-8 -*-

from collections import deque

from .parser import lexer, parser
from .surface import create_surface
from .transformation import Transformation


def read_mcnp_model(filename):
    """Reads MCNP model from file and creates corresponding objects.

    Parameters
    ----------
    filename : str
        File that contains MCNP model.

    Returns
    -------
    title : str
        Model's title.
    cells, surfaces, data : dict
        Dictionaries of created cells, surfaces and data cards respectively.
    """
    with open(filename) as f:
        text = f.read()
    lexer.begin('INITIAL')
    title, cells, surfaces, data = parser.parse(text)
    # Transformation instances creation
    if 'TR' in data.keys():
        for tr_name, tr_data in data['TR'].items():
            data['TR'][tr_name] = Transformation(**tr_data)
    # Creation of Surface instances
    for sur_name, (kind, params, options) in surfaces.items():
        if 'transform' in options.keys():
            tr_name = options['transform']
            options['transform'] = data['TR'][tr_name]
        surfaces[sur_name] = create_surface(kind, *params, **options)
    # Creation of Cell instances
    # First, we must replace all cell complement operations (NUM followed by#)
    # and surface numbers by corresponding Surface instances.
    _replace_geometry_names_by_objects(cells, surfaces)


def _replace_geometry_names_by_objects(cells, surfaces):
    """Replaces surface and cell names in geometry description by objects.

    Parameters
    ----------
    cells : dict
        Dictionary of cell parameters.
    surfaces : dict
        Dictionary of surface objects
    """
    unhandled_cells = deque(cells.keys())
    while unhandled_cells:
        name = unhandled_cells.popleft()
        if 'reference' in cells[name].keys():
            ref_cell = cells[name]['reference']
            if ref_cell in unhandled_cells:
                unhandled_cells.append(name)
            else:
                cells[name]['geometry'] = cells[ref_cell]['geometry'].copy()
        else:
            geom = cells[name]['geometry']
            for i in reversed(range(len(geom))):
                if geom[i] == '#':
                    comp_cell = geom[i-1]
                    if comp_cell in unhandled_cells:
                        unhandled_cells.append(name)
                        break
                    geom[i-1:i+1] = cells[comp_cell]['geometry']
                elif isinstance(geom[i], int):
                    geom[i] = surfaces[geom[i]]


class Model:
    def __init__(self):
        pass
