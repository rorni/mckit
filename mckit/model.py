# -*- coding: utf-8 -*-

from collections import deque
from functools import reduce

import numpy as np

from .parser import lexer, parser
from .surface import create_surface, Surface
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
    get_universe_model(uname, title)
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

    def universe(self):
        """Gets the model in object representation.
        
        Returns
        -------
        universe : Universe
            The model represented as universe object.
        """
        # 1. Create transformation objects
        self._transformations = {}
        if 'TR' in self.data.keys():
            for tr_name, tr_data in self.data['TR'].items():
                self._transformations[tr_name] = Transformation(**tr_data)
        # 2. Create surface objects
        self._surfaces = {}
        for sur_name, (kind, params, options) in self._surfaces.items():
            if 'transform' in options.keys():
                tr_name = options['transform']
                options['transform'] = self._transformations[tr_name]
            self._surfaces[sur_name] = create_surface(kind, *params, **options)
        # 3. Create cell geometries
        geometries = {}
        self._materials = {}
        self._densities = {}
        # 3. Create material objects
        materials = []
        # 4. Adjust cell params and create corresponding objects.
        # 5. Create cell objects
        # 6. Create universe

    def _create_universe(self, uname):
        """Creates new universe from data of this model.
        
        Returns
        -------
        universe : Universe
        """
        cells = []
        for cell_name in self.universes[uname].keys():
            geometry = self._produce_cell_geometry(cell_name)
            options = self.cells[cell_name].copy()
            options.pop('geometry', None)
            options.pop('reference', None)

    def _produce_cell(self, cell_name):
        """Creates Cell instance."""
        geometry = self._produce_cell_geometry(cell_name)
        options = self.cells[cell_name].copy()
        options.pop('geometry', None)
        options.pop('reference', None)
        fill = options.pop('FILL', None)
        if fill:
            universe = self._create_universe(fill.pop('universe'))
            tr = self._get_transform_object(fill.pop('transform', None))
            if tr:
                universe = universe.transform(tr)
            options['FILL'] = universe

        tr = options.pop('TRCL', None)
        if tr:
            for i, s in enumerate(geometry):
                if isinstance(s, Surface):
                    geometry[i] = s.transform(tr)
            if 'FILL' in options.keys():
                options['FILL'] = universe.transform(tr)
        self._cells[cell_name] = Cell(geometry, **options)
        return self._cells[cell_name]

    def _produce_cell_geometry(self, cell_name):
        """Creates a list that describes cell geometry.
        
        This function can use reference geometry. It replaces cell complement
        operations, replaces surface numbers by surface objects.
        
        Parameters
        ----------
        cell_name : int
            A name of the cell under consideration.
            
        Returns
        -------
        geometry : list
            List that describes cell geometry in object representation.
        """
        geometry = self._get_reference_geometry(cell_name).copy()
        i = 0
        while i < len(geometry):
            elem = geometry[i]
            if isinstance(elem, int) and geometry[i+1] == '#':
                geometry[i:i+2] = self._get_reference_geometry(elem) + ['C']
                continue
            elif isinstance(elem, int):
                geometry[i] = self._get_surface_object(elem)
            i += 1
        return geometry

    def _get_reference_geometry(self, cell_no):
        """Gets geometry of the cell with name cell_no."""
        ref_cell = cell_no
        while 'reference' in self.cells[ref_cell].keys():
            ref_cell = self.cells[ref_cell]['reference']
        return self.cells[ref_cell]['geometry']

    def _get_material_object(self, comp_no, density):
        """Gets material object that corresponds to comp_no and density.
        
        Parameters
        ----------
        comp_no : int
            Composition name.
        density : float
            Density of material.
            
        Returns 
        -------
        material : Material
            Material object.
        """
        if comp_no not in self._compositions.keys():
            self._densities[comp_no] = [0]
            self._materials[comp_no] = [None]
        i = np.searchsorted(self._densities[comp_no], density)
        test_indices = []
        if i-1 > 0:
            test_indices.append(i-1)
        if i < len(self._densities):
            test_indices.append(i)
        rels = []
        for ti in test_indices:
            t_den = self._densities[ti]
            rels.append(abs(t_den - density) / abs(t_den + density))
        min_ind = np.argmin(rels)
        if rels[min_ind] >= RELATIVE_DENSITY_TOLERANCE:
            mat_params = {}
            if density > 0:
                mat_params['concentration'] = density
            elif density < 0:
                mat_params['density'] = abs(density)
            if 'atomic' in self.data['M'][comp_no].keys():
                mat_params['atomic'] = self.data['M'][comp_no]['atomic']
            if 'wgt' in self.data['M'][comp_no].keys():
                mat_params['wgt'] = self.data['M'][comp_no]['wgt']
            mat = Material(**mat_params)
            self._densities[comp_no].insert(i, density)
            if density > 0:
                self._densities[comp_no].insert(-i, mat.density())
            else:
                self._densities[comp_no].insert(-i, mat.concentration())
            self._materials[comp_no].insert(i, mat)
            self._materials[comp_no].insert(-i, mat)
            min_ind = i
        return self._materials[min_ind]

    def _get_surface_object(self, surf_no):
        """Gets surface object that corresponds to surf_no."""
        if surf_no not in self._surfaces.keys():
            kind, params, options = self.surfaces[surf_no]
            if 'transform' in options.keys():
                tr_name = options['transform']
                options = options.copy()
                options['transform'] = self._get_transform_object(tr_name)
            self._surfaces[surf_no] = create_surface(kind, *params, **options)
        return self._surfaces[surf_no]

    def _get_transform_object(self, tr):
        """Gets transformation object that corresponds to tr."""
        if isinstance(tr, dict):
            return Transformation(**tr)
        elif isinstance(tr, int):
            if tr not in self._transformations.keys():
                tr_data = self.data['TR'][tr]
                self._transformations[tr] = Transformation(**tr_data)
            return self._transformations[tr]

    def get_universe_list(self):
        """Gets the list of universe names.
        
        Returns
        -------
        unames : list[int]
            List of included universe names.
        """
        return list(self.universes.keys())

    def get_universe_model(self, uname, title=None):
        """Gets the Model instance that corresponds to the specified universe.
        
        Parameters
        ----------
        uname : int
            The name of universe under consideration.
        title : str
            A brief description of the universe.
            
        Returns
        -------
        model : Model
            Model, that corresponds to the specified universe.
        """
        if title is None:
            title = "Universe {0}".format(uname)

        contained_univ = self.get_contained_universes(uname) | {uname}
        cells = {}
        for u in contained_univ:
            cells.update(self.universes[u])
        surf_ind = reduce(set.union, [self.get_surface_indices(c['geometry'])
                                      for c in cells.values()], set())
        surfaces = {i: self.surfaces[i] for i in surf_ind}
        tr_ind = set()  # indices of transformations
        for s in surfaces.values():
            if 'transform' in s[2].keys():
                tr_ind.add(s[2]['transform'])
        mat_ind = set()  # indices of material compositions
        for c in cells.values():
            if 'TRCL' in c.keys() and isinstance(c['TRCL'], int):
                tr_ind.add(c['TRCL'])
            if 'FILL' in c.keys():
                if 'transform' in c['FILL'].keys() and \
                        isinstance(c['FILL']['transform'], int):
                    tr_ind.add(c['FILL']['transform'])
            if 'MAT' in c.keys():
                mat_ind.add(c['MAT'])
        data = self.data.copy()     # All data cards remain the same except
                                    # material and transformation cards.
        data['TR'] = {i: data['TR'][i] for i in tr_ind}
        data['M'] = {i: data['M'][i] for i in mat_ind}
        return Model(title, cells, surfaces, data)

    def get_contained_universes(self, uname):
        """Gets a list of universes, that are contained in this one."""
        contained = set()
        for c in self.universes[uname].values():
            if 'FILL' in c.keys():
                contained.add(c['FILL']['universe'])
        inner = []
        for u in contained:
            inner.append(self.get_contained_universes(u))
        return contained.union(*inner)

    @classmethod
    def get_surface_indices(cls, geometry):
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
