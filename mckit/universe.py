# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np

from .body import Body
from .surface import create_surface
from .constants import MIN_BOX_VOLUME
from .geometry import GLOBAL_BOX, Box
from .material import Composition, Material
from .transformation import Transformation


__all__ = ['Universe']


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
    bounding_box(tol, box)
        Gets bounding box of the universe.
    fill(cell)
        Fills every cell that has fill option by cells of filling universe.
    copy()
        Makes a copy of the universe.
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
    select_universe(name)
        Gets specified inner universe.
    select_cell(name)
        Gets specified cell.
    rename(start_cell, start_surf, start_mat, start_tr)
        Renames all entities contained in the universe.
    check_names()
        Checks if there is name clashes.
    save(filename)
        Saves the universe to file.
    """
    def __init__(self, cells, name=0, verbose_name=None, comment=None, universes=None):
        self._name = name
        self._comment = comment
        self._verbose_name = name if verbose_name is None else verbose_name
        if universes is None:
            universes = {}
        self._cells = []

        for c in cells:
            u = c.get('U', 0)
            if isinstance(u, Universe):
                u = u.name
            if u != name:
                continue
            fill = c.get('FILL')
            if fill is not None:
                uname = fill['universe']
                if not isinstance(uname, Universe):
                    if uname not in universes.keys():
                        universes[uname] = Universe(cells, name=uname, universes=universes)
                    fill['universe'] = universes[uname]
            c['U'] = self
            self._cells.append(c)

    def __iter__(self):
        return iter(self._cells)

    def copy(self):
        """Makes a shallow copy of this universe."""
        u = Universe([], name=self.name, verbose_name=self.verbose_name,
                     comment=self._comment)
        u._cells = self._cells.copy()
        return u

    @property
    def name(self):
        return self._name

    @property
    def verbose_name(self):
        return self._verbose_name

    def bounding_box(self, tol=100, box=GLOBAL_BOX):
        """Gets bounding box for the universe.

        It finds all bounding boxes for universe cells and then constructs
        box, that covers all cells' bounding boxes. The main purpose of this
        method is to find boundaries of the model geometry to compare
        transformation and surface objects for equality. It is recommended to
        choose tol value not too small to reduce computation time.

        Parameters
        ----------
        tol : float
            Linear tolerance for the bounding box. The distance [in cm] between
            every box's surface and universe in every direction won't exceed
            this value. Default: 100 cm.
        box : Box
            Starting box for the search. The user must be sure that box covers
            all geometry, because cells, that already contains corners of the
            box will be considered as infinite and will be excluded from
            analysis.

        Returns
        -------
        bbox : Box
            Universe bounding box.
        """
        boxes = []
        for c in self._cells:
            test = c.shape.test_points(box.corners)
            if np.any(test == +1):
                continue
            boxes.append(c.shape.bounding_box(tol=tol, box=box))
        all_corners = np.empty((8 * len(boxes), 3))
        for i, b in enumerate(boxes):
            all_corners[i * 8: (i + 1) * 8, :] = b.corners
        min_pt = np.min(all_corners, axis=0)
        max_pt = np.max(all_corners, axis=0)
        center = 0.5 * (min_pt + max_pt)
        dims = max_pt - min_pt
        return Box(center, *dims)

    def fill(self, cells=None, recurrent=False, simplify=False, **kwargs):
        """Fills cells that have fill option by filling universe cells.

        The cells that initially were in this universe and had fill option
        are replaced by filling universe. Geometry of the cell being filled
        is used to bound cells of filler universe. Simplification won't be
        done by default.

        Parameters
        ----------
        cells : int or list[int]
            Names of cell or cells, which must be filled. Default: None - all
            universe cells will be considered.
        recurrent : bool
            Indicates if apply_fill of inner universes should be also invoked.
            In this case, apply_fill is first invoked for inner universes.
            Default: False.
        simplify : bool
            Whether to simplify universe cells after filling. Default: False.
        kwargs : dict
            Parameters for simplify method. See Body.simplify.

        Returns
        -------
        new_universe : Universe
            New filled universe.
        """
        if isinstance(cells, int):
            cells = [cells]
        new_cells = []
        for c in self._cells:
            name = c.get('name', 0)
            if cells and name not in cells:
                new_cells.append(c)
            else:
                new_cells.extend(
                    c.fill(recurrent=recurrent, simplify=simplify, **kwargs)
                )
        new_universe = self.copy()
        new_universe._cells = new_cells
        if simplify:
            new_universe = new_universe.simplify(**kwargs)
        return new_universe

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
            new_cell = c.simplify(
                box=box, split_disjoint=split_disjoint, min_volume=min_volume,
                trim_size=trim_size
            )
            if not new_cell.shape.is_empty():
                cells.append(new_cell)
        u = self.copy()
        u._cells = cells

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
        u = self.copy()
        u._cells = tr_cells
        return u

    def get_surfaces(self):
        """Gets all surfaces that discribe this universe.

        Returns
        -------
        surfaces : set[Surface]
            A set of all contained surfaces.
        """
        surfaces = set()
        for c in self._cells:
            cs = c.shape.get_surfaces()
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
            m = c.material()
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

    def select_universe(self, name):
        """Selects given universe.

        If no such universe present, KeyError is raised.

        Parameters
        ----------
        name : int or str
            Numeric or verbose name of universe to be selected.

        Returns
        -------
        universe : Universe
            Requested universe.
        """
        if name == self.name or name == self.verbose_name:
            return self
        for c in self._cells:
            fill = c.get('FILL')
            if fill:
                u = fill['universe']
                try:
                    return u.select_universe(name)
                except KeyError:
                    pass
        raise KeyError("No such universe: {0}".format(name))

    def select_cell(self, name):
        """Selects given cell.

        Cell must belong this universe.
        If no such cell present, KeyError is raised.

        Parameters
        ----------
        name : int
            Name of cell.

        Returns
        -------
        cell : Body
            Requested cell.
        """
        for c in self._cells:
            if c.get['name'] == name:
                return c
        raise KeyError("No such cell: {0}".format(name))

    def rename(self, start_cell=1, start_surf=1, start_mat=1, start_tr=1,
               name=None, verbose_name=None):
        """Renames universe entities.

        Modifies current universe. All nested universes must be renamed
        separately if needed.

        Parameters
        ----------
        start_cell : int
            Starting name for cells. Default: 1.
        start_surf : int
            Starting name for surfaces. Default: 1.
        start_mat : int
            Starting name for compositions (MCNP materials). Default: 1.
        start_tr : int
            Starting name for transformations. Default: 1.
        name : int
            Name of this universe. Only needed if name must be changed.
            Default: None.
        verbose_name : str
            Verbose name of this universe. Only needed if verbose name of the
            universe must be changed. Default: None.
        """
        for s in self.get_surfaces():
            s.options['name'] = start_surf
            start_surf += 1
        for comp in set(m.composition for m in self.get_materials()):
            comp._options['name'] = start_mat
            start_mat += 1
        transformations = set()
        for c in self._cells:
            c['name'] = start_cell
            start_cell += 1
            tr = c.get('TRCL', None)
            if tr and tr['name']:
                transformations.add(tr)
            fill = c.get('FILL', {})
            tr = fill.get('transform', None)
            if tr and tr['name']:
                transformations.add(tr)
        for tr in transformations:
            tr['name'] = start_tr
            start_tr += 1
        if name is not None:
            self._name = name
        if verbose_name is not None:
            self._verbose_name = verbose_name

    def check_names(self):
        """Checks if the universe entities have name clashes.

        First it checks if there is name clashes in cells, surfaces, materials
        and transformations that belong to the universe itself, and then with
        all other nested universes.

        Returns
        -------
        status : bool
            Status of the check. True if everything is OK, False, otherwise.
        report : dict
            Dictionary of check results. Here found clashes are reported.
        """
        names = {}
        self._collect_names(names)

    def _collect_names(self, results):
        if id(self) in results.keys():
            return
        names = {'cells': [], 'surfaces': [], 'compositions': [],
                 'transformations': [], 'name': self._name}
        for s in self.get_surfaces():
            names['surfaces'].append(s.options.get('name', 0))
        compositions = {m.composition for m in self.get_materials()}
        for c in compositions:
            names['compositions'].append(compositions._options.get('name', 0))
        transformations = set()
        for c in self:
            names['cells'].append(c.get('name', 0))
            tr = c.get('TRCL', None)
            if tr is not None:
                transformations.add(tr)
            fill = c.get('FILL', None)
            if fill:
                u = fill['universe']
                tr = fill.get('transform', None)
                if tr:
                    transformations.add(tr)
                u._collect_names(results)
        for tr in transformations:
            names['transformations'].append(tr._options.get('name', 0))
        results[id(self)] = names

    def save(self, filename, format_rules=None):
        """Saves the universe to file.

        Parameters
        ----------
        filename : str
            Name of file.
        format_rules : dict
            A dictionary of format rules.
        """
        raise NotImplementedError
