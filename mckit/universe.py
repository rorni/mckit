# -*- coding: utf-8 -*-

from collections import OrderedDict

from .body import Body
from .surface import create_surface
from .constants import GLOBAL_BOX, MIN_BOX_VOLUME
from .material import Composition, Material
from .parser.mcnp_input_parser import read_mcnp
from .transformation import Transformation


class Universe:
    """Describes universe - a set of cells.
    
    Universe is a set of cells from which it consist of. Each cell can be filled
    with other universe. In this case, cells of other universe are bounded by
    cell being filled.
    
    Parameters
    ----------
    cells : iterable
        A list of cells this universe consist of. Among these cells can be
        cells of other universes. The most outer universe will be created.
        Inner universes will be created either, but they will be accessed
        via the outer.
    name : int
        Name of the universe, usually used in MCNP. 0 - means global universe.
    verbose_name : str
        Optional verbose name. This name also can be used to retrieve inner universe.
    comment : list[str] or str
        String or list of strings that describes the universe.
    universes : dict
        A dictionary of already created universes: universe_name -> universe.
        Default: None.
        
    Methods
    -------
    fill(cell)
        Fills every cell that has fill option by cells of filling universe.
    transform(tr)
        Applies transformation tr to this universe. Returns a new universe.
    simplify(box, split_disjoint, min_volume, trim_size)
        Simplifies all cells of the universe.
    get_surfaces()
        Gets all surfaces of the universe.
    get_materials()
        Gets all materials of the universe.
    get_transformations()
        Gets all transformations of the universe.
    get_universes()
        Gets all inner universes.
    select_universe(key)
        Gets specified inner universe.
    select_cell(name)
        Gets specified cell.
    give_names(start_cell, start_surf, start_mat, start_tr)
        Renames all entities contained in the universe.
    """
    def __init__(self, cells, name=0, verbose_name=None, comment=None, universes=None):
        self._name = name
        self._comment = comment
        self._verbose_name = verbose_name
        if universes is None:
            universes = {}
        self._cells = []

        for c in cells:
            u = c.get('U', 0)
            if u != name:
                continue
            fill = c.get('FILL')
            if fill is not None:
                uname = fill['universe']
                if not isinstance(uname, Universe):
                    if uname not in universes.keys():
                        universes[uname] = Universe(cells, name=uname, universes=universes)
                    fill['universe'] = universes[uname]
            c.set('U', self)
            self._cells.append(c)

    @staticmethod
    def from_file(filename):
        """Reads MCNP universe from file.

        Parameters
        ----------
        filename : str
            Input file name.

        Returns
        -------
        universe : Universe
            New universe corresponding to the file.
        """
        title, cells, surfaces, data = read_mcnp(filename)

        # Create transformations objects
        transforms = data.get('TR', {})
        for tr_name, tr_data in transforms.items():
            transforms[tr_name] = Transformation(**tr_data)

        # Create composition objects (MCNP materials)
        compositions = data.get('M', {})
        for mat_name, mat_data in compositions.items():
            compositions[mat_name] = Composition(**mat_data)

        # create surface objects
        surfaces = {name: _create_surface_object(data, transforms) for name, data in surfaces.items()}

        # create cell objects
        cells = OrderedDict(cells.items())
        created_cells = {}
        while cells:
            name, data = cells.popitem(last=False)
            cell = _create_cell_object(data, created_cells, surfaces, transforms, compositions)
            if cell:
                created_cells[name] = cell
            else:
                cells[name] = data

        return Universe(created_cells, verbose_name=title)

    def __iter__(self):
        return iter(self._cells)

    @property
    def name(self):
        return self._name

    @property
    def verbose_name(self):
        return self._verbose_name

    def fill(self):
        """Fills cells that have fill option by filling universe cells.

        This method modifies current universe. The cells that initially
        were in this universe and had fill option are replaced by filling
        universe. Geometry of the cell being filled is used to bound cells
        of filler universe. Simplification won't be done by default.
        The user should call simplification when needed.
        """
        new_cells = []
        i = 0
        while i < len(self._cells):
            fill = self._cells[i].get('FILL')
            if fill:
                c = self._cells.pop(i)
                u = fill['universe']
                tr = fill.get('transform', None)
                if tr:
                    u = u.transform(tr)
                for uc in u:
                    new_cells.append(uc.intersection(c))
            else:
                i += 1
        for c in new_cells:
            c.set('U', self)
        self._cells.extend(new_cells)

    def simplify(self, box=GLOBAL_BOX, split_disjoint=False,
                 min_volume=MIN_BOX_VOLUME, trim_size=1):
        """Simplifies this universe by simplifying all cells.

        The simplification procedure goes in the following way.
        # TODO: insert brief description!

        Parameters
        ----------
        box : Box
            Box where geometry should be simplified.
        split_disjoint : bool
            Whether to split disjoint geometries into separate geometries.
        min_volume : float
            The smallest value of box's volume when the process of box splitting
            must be stopped.
        trim_size : int
            Max size of set to return. It is used to prevent unlimited growth
            of the variant set.

        Returns
        -------
        universe : Universe
            Simplified version of this universe.
        """
        cells = []
        for c in self._cells:
            new_cell = c.simplify(box=box, split_disjoint=split_disjoint, min_volume=min_volume, trim_size=trim_size)
            if not new_cell.is_empty():
                cells.append(new_cell)
        return Universe(cells)

    def transform(self, tr):
        """Applies transformation tr to this universe. 
        
        Parameters
        ----------
        tr : Transformation
            Transformation to be applied.
            
        Returns
        -------
        universe : Universe
            New transformed universe.
        """
        tr_cells = []
        for cell in self._cells:
            tr_cells.append(cell.transform(tr))
        return Universe(tr_cells)

    def get_surfaces(self):
        """Gets all surfaces that discribe this universe.

        Returns
        -------
        surfaces : set[Surface]
            A set of all contained surfaces.
        """
        surfaces = set()
        for c in self._cells:
            cs = c.get_surfaces()
            surfaces.update(cs)
        return surfaces

    def get_materials(self):
        """Gets all materials that belong to the universe.

        Returns
        -------
        materials : set[Material]
            A set of all materials that are contained in the universe.
        """
        materials = set()
        for c in self._cells:
            m = c.material
            if m is not None:
                materials.add(m)
        return materials

    def get_universes(self, recurrent=True):
        """Gets names of all universes that are included in the universe.

        Parameters
        ----------
        recurrent : bool
            Whether to list universes that are not directly included in this one.
            Default: True.

        Returns
        -------
        universes : set[int or str]
            A set of universe names. It can be either MCNP name or verbose name.
        """
        universes = set()
        for c in self._cells:
            fill = c.get('FILL')
            if fill:
                universes.add(fill['universe'].verbose_name)
                if recurrent:
                    universes.update(fill['universe'].get_universes(recurrent=True))
        return universes

    def select_universe(self, key):
        """Selects given universe.

        If no such universe present, KeyError is raised.

        Parameters
        ----------
        key : int or str
            Numeric or verbose name of universe to be selected.

        Returns
        -------
        universe : Universe
            Requested universe.
        """
        if key == self.name or key == self.verbose_name:
            return self
        for c in self._cells:
            fill = c.get('FILL')
            if fill:
                u = fill['universe']
                try:
                    return u.select_universe(key)
                except KeyError:
                    pass
        raise KeyError("No such universe: {0}".format(key))

    def select_cell(self, key):
        """Selects given cell.

        Cell must belong this universe.
        If no such cell present, KeyError is raised.

        Parameters
        ----------
        key : int
            Name of cell.

        Returns
        -------
        cell : Body
            Requested cell.
        """
        for c in self._cells:
            if c.get['name'] == key:
                return c
        raise KeyError("No such cell: {0}".format(key))

    def give_names(self, start_cell=1, start_surf=1, start_mat=1, start_tr=1):
        raise NotImplementedError


def replace(data, keyword, replace_items, factory):
    """Replaces values of dict according to replacement dictionary.

    It modifies current object.

    Parameters
    ----------
    data : dict
        dict instance that has key which value has to be replaced. It must
        contain 'name' keyword.
    keyword : str
        Keyword to be replaced.
    replace_items : dict
        A dictionary of replacements.
    factory : func
        A function or class constructor, that takes keyword arguments to
        create an object, if value is a dict instance.
    """
    value = data.get(keyword, None)
    if isinstance(value, dict):
        data[keyword] = factory(**value)
    else:
        data[keyword] = replace_items[value]


def _create_surface_object(surf_data, transforms):
    """Creates surface object from surface data dictionary.

    Parameters
    ----------
    surf_data : dict
        A dictionary of surface data. It must contain 'kind' and 'params' keys.
        Other keys are optional.
    transforms : dict
        A dictionary of transformation objects. tr_name (int) -> transform (Transformation).

    Returns
    -------
    surf : Surface
        New Surface instance.
    """
    surf_data = surf_data.copy()
    kind = surf_data.pop('kind')
    params = surf_data.pop('params')
    tr_name = surf_data.get('transform', None)
    if tr_name:
        surf_data['transform'] = transforms[tr_name]
    return create_surface(kind, *params, **surf_data)


def _create_cell_object(cell_data, created_cells, surfaces, transforms, compositions):
    """Creates cell object from cell data dictionary.

    Parameters
    ----------
    cell_data : dict
        A dictionary of cell data.
    created_cells : dict
        A dictionary of already created cells. cell_name (int) -> cell (Body).
    surfaces : dict
        A dictionary of Surface objects.
    transforms : dict
        A dictionary of Transformation objects.
    compositions : dict
        A dictionary of Composition objects.

    Returns
    -------
    cell : Body
        New Body instance. None if this cell cannot be created for now.
    """
    geometry = cell_data.pop('geometry', None)
    if geometry is None:
        # reference geometry
        ref_name = cell_data.pop('reference')
        ref_cell = created_cells.get(ref_name, None)
        if ref_cell:
            for k, v in ref_cell.items():
                if k not in cell_data.keys():
                    cell_data[k] = v
            geometry = ref_cell.shape
        else:  # Reference cell has not been created yet. Terminating.
            cell_data['reference'] = ref_name
            return None
    else:
        # replace geometry numbers by corresponding surface objects.
        n = len(geometry)
        for i in range(n):
            if isinstance(geometry[i], int):
                if i + 1 < n and geometry[i+1] == '#':
                    if geometry[i] in created_cells.keys():
                        geometry[i] = created_cells[geometry[i]]
                        geometry[i+1] = 'C'
                    else:
                        cell_data['geometry'] = geometry
                        return None
                else:
                    geometry[i] = surfaces[geometry[i]]
    # replace options
    _transform_replace(cell_data, 'TRCL', transforms)

    fill = cell_data.get('FILL', None)
    if fill:
        _transform_replace(fill, 'transform', transforms)

    comp_name = cell_data.get('MAT', None)
    if comp_name:
        data = {'composition': compositions[comp_name]}
        rho = cell_data.pop('RHO')
        if rho > 0:
            data['concentration'] = rho * 1.e+24
        else:
            data['density'] = - rho
        cell_data['MAT'] = Material(**data)

    return Body(geometry, **cell_data)


def _transform_replace(data, key, transforms):
    tr = data.get(key, None)
    if isinstance(tr, int):
        data[key] = transforms[tr]
    elif isinstance(tr, dict):
        data[key] = Transformation(**tr)

