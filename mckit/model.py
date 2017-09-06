# -*- coding: utf-8 -*-

from collections import deque
from functools import reduce

from .parser import lexer, parser
from .surface import create_surface
from .cell import Cell
from .transformation import Transformation
from .material import Material
from .constants import RELATIVE_DENSITY_TOLERANCE


def read_mcnp_model(filename):
    """Reads MCNP model from file and creates corresponding objects.

    Parameters
    ----------
    filename : str
        File that contains MCNP model.

    Returns
    -------
    model : Model
        Calculation model.
    """
    with open(filename) as f:
        text = f.read()
    lexer.begin('INITIAL')
    title, cells, surfaces, data = parser.parse(text)
    return Model(title, cells, surfaces, data)


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
                    geom[i-1:i+1] = cells[comp_cell]['geometry'] + ['C']
                elif isinstance(geom[i], int):
                    geom[i] = surfaces[geom[i]]


def _create_material_objects(cells, compositions):
    """Creates material objects and assign them to cells.

    Materials objects are assigned to 'material' keyword of cell.

    Parameters
    ----------
    cells : dict
        Dictionary of cell parameters.
    compositions : dict
        Dictionary of material parameters, that define material composition.
        Its values are also dictionaries with 'wgt' or 'atomic' keywords.
        
    Returns
    -------
    materials : dict
        A dictionary of created material objects. Keys - composition numbers, 
        values - lists of corresponding Material instances (they differ only in
        density). 
    """
    created_materials = {}  # Stores materials created in dictionary
                            # composition_name -> material_instance
    for cell in cells.values():
        if 'MAT' in cell.keys():
            comp_name = cell['MAT']
            density = cell['RHO']
            # Check if material already exists.
            if comp_name in created_materials.keys():
                mat = _get_material(created_materials[comp_name], density)
            else:
                mat = None
                created_materials[comp_name] = []

            # Create new material
            if not mat:
                mat_params = {}
                if density > 0:
                    mat_params['concentration'] = density
                elif density < 0:
                    mat_params['density'] = abs(density)
                if 'atomic' in compositions[comp_name].keys():
                    mat_params['atomic'] = compositions[comp_name]['atomic']
                if 'wgt' in compositions[comp_name].keys():
                    mat_params['wgt'] = compositions[comp_name]['wgt']
                mat = Material(**mat_params)
                created_materials[comp_name].append(mat)
            cell['material'] = mat
    return created_materials


def _get_material(materials, density):
    """Checks if material with specified density already exists and returns it.

    Parameters
    ----------
    materials : list[Material]
        List of materials with the same composition.
    density : float
        Density of material. If positive, then it is concentration; if negative,
        it is weight density.

    Returns
    -------
    material : Material
        If material with specified density already exists, it is returned.
        Otherwise None is returned.
    """
    if density < 0:
        method = Material.density
        density = abs(density)
    else:
        method = Material.concentration
    for mat in materials:
        mat_den = method(mat)
        rel = 2 * abs(mat_den - density) / (mat_den + density)
        if rel < RELATIVE_DENSITY_TOLERANCE:
            return mat
    return None


class Model:
    """Represents calculation model.
    
    Parameters
    ----------
    title : str
        Title that briefly describes the model.
    cells : dict
        A dictionary of cell data. It is pairs cell_name -> cell_params. 
        Cell_params is also a dictionary of raw cell parameters. It contains
        indices (references) of other model objects like transformations, 
        surfaces, materials, etc.
    surfaces : dict
        A dictionary of raw surface data. It is pairs surf_name -> surf_params.
    data : dict
        Dictionary of raw data, that describes datacards.
        
    Methods
    -------
    get_universe_model(uname)
        Gets the Model instance that corresponds to the specified universe.
    get_universe_list()
        Gets the list of universe names.
    universe()
        Gets the model in object representation (as Universe instance). 
    """
    def __init__(self, title, cells, surfaces, data):
        self.title = title
        self.cells = cells
        self.surfaces = surfaces
        self.data = data
        self.universes = {0: {}}
        for cell_name, cell_params in cells.items():
            uname = cell_params.get('U', 0)
            if uname not in self.universes.keys():
                self.universes[uname] = {}
            self.universes[uname][cell_name] = cell_params

    def get_universe_list(self):
        """Gets the list of universe names.
        
        Returns
        -------
        unames : list[int]
            List of included universe names.
        """
        return list(self.universes.keys())

    def get_universe_model(self, uname):
        """Gets the Model instance that corresponds to the specified universe.
        
        Parameters
        ----------
        uname : int
            The name of universe under consideration.
            
        Returns
        -------
        model : Model
            Model, that corresponds to the specified universe.
        """
        cells = self.universes[uname].copy()
        surf_ind = reduce(set.union, [self.get_surface_indices(c['geometry'])
                                      for c in cells.values()])
        surfaces = {i: self.surfaces[i] for i in surf_ind}

    def get_universes_contained(self, uname):
        """Gets a list of universes, that are contained in this one."""
        contained = set()
        for c in self.universes[uname].values():
            if 'FILL' in c.keys():
                contained.add(c['FILL']['universe'])
        inner = []
        for u in contained:
            inner.append(self.get_universes_contained(u))
        return contained.union(*inner)

    @classmethod
    def get_surface_indices(self, geometry):
        """Gets a set of surface indices included in the geometry."""
        surfs = set()
        for i, c in enumerate(geometry):
            if isinstance(c, int) and geometry[i+1] != '#':
                surfs.add(c)
        return surfs

        # # --------------------------------------------------------------------------
        # # Transformation instances creation
        # if 'TR' in data.keys():
        #     for tr_name, tr_data in data['TR'].items():
        #         data['TR'][tr_name] = Transformation(**tr_data)
        # # Creation of Surface instances
        # for sur_name, (kind, params, options) in surfaces.items():
        #     if 'transform' in options.keys():
        #         tr_name = options['transform']
        #         options['transform'] = data['TR'][tr_name]
        #     surfaces[sur_name] = create_surface(kind, *params, **options)
        # # Creation of Cell instances
        # # First, we must replace all cell complement operations (NUM followed by#)
        # # and surface numbers by corresponding Surface instances.
        # _replace_geometry_names_by_objects(cells, surfaces)
        #
        # # Create Material instances and assign them to cells
        # _create_material_objects(cells, data['M'])
        #
        # # Create Cell objects itself.
        # for cell_name, cell_params in cells.items():
        #     geometry = cell_params.pop('geometry')
        #     cells[cell_name] = Cell(geometry, **cell_params)
