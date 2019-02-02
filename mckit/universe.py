# -*- coding: utf-8 -*-

from collections import OrderedDict, defaultdict

import numpy as np

from .body import Body
from .surface import create_surface
from .constants import MIN_BOX_VOLUME
from .geometry import GLOBAL_BOX, Box
from .material import Composition, Material
from .transformation import Transformation
from .card import Card

__all__ = [
    'Universe', 'produce_universes', 'NameClashError', 'get_cell_selector',
    'get_surface_selector'
]


class NameClashError(Exception):
    pass


def get_cell_selector(cell_names):
    """Produces cell selector function for specific cell names.

    Parameters
    ----------
    cell_names : int or iterable
        Names of cells to be selected.

    Returns
    -------
    selector : func
        Selector function.
    """
    if isinstance(cell_names, int):
        cell_names = {cell_names}
    else:
        cell_names = set(cell_names)

    def selector(cell):
        if cell.name() in cell_names:
            return [cell]
        else:
            return []
    return selector


def get_surface_selector(surface_names):
    """Produces surface selector function for specific surface names.

    Parameters
    ----------
    surface_names : int or iterable
        Names of surfaces to be selected.

    Returns
    -------
    selector : func
        Selector function
    """
    if isinstance(surface_names, int):
        surface_names = {surface_names}
    else:
        surface_names = set(surface_names)

    def selector(cell):
        surfs = cell.shape.get_surfaces()
        return [s for s in surfs if s.name() in surface_names]
    return selector


def produce_universes(cells):
    """Creates universes from cells.

    The function groups all cells that have equal value of 'universe' option,
    and the creates corresponding universes. Universe with name 0 is returned
    only.

    Parameters
    ----------
    cells : list[Body]
        A list of cells.

    Returns
    -------
    universe : Universe
        Main universe.
    """
    universes = defaultdict(lambda: Universe([]))
    for c in cells:
        uname = c.options.get('U', 0)
        universes[uname].add_cell(c, name_rule='keep')
        fill = c.options.get('FILL', None)
        if fill:
            fill['universe'] = universes[fill['universe']]
    return universes[0]


class Universe:
    """Describes universe - a set of cells.
    
    Universe is a set of cells from which it consist of. Each cell can be filled
    with other universe. In this case, cells of other universe are bounded by
    cell being filled.
    
    Parameters
    ----------
    cells : iterable
        A list of cells this universe consist of.
    name : int
        Name of the universe, usually used in MCNP. 0 - means global universe.
    verbose_name : str
        Optional verbose name. This name also can be used to retrieve inner
        universe.
    comment : list[str] or str
        String or list of strings that describes the universe.

    Methods
    -------
    add_cell(cell)
        Adds new cell to the universe.
    apply_fill(cell, universe)
        Applies fill operations to all or selected cells or universes.
    bounding_box(tol, box)
        Gets bounding box of the universe.
    check_names()
        Checks if there is name clashes.
    copy()
        Makes a copy of the universe.
    get_surfaces()
        Gets all surfaces of the universe.
    get_materials()
        Gets all materials of the universe.
    get_composition()
        Gets all compositions of the universe.
    get_transformations()
        Gets all transformations of the universe.
    get_universes()
        Gets all inner universes.
    name()
        Gets numeric name of the universe.
    rename(start_cell, start_surf, start_mat, start_tr)
        Renames all entities contained in the universe.
    save(filename)
        Saves the universe to file.
    select(cell, selector)
        Selects specified entities.
    simplify(box, split_disjoint, min_volume)
        Simplifies all cells of the universe.
    test_points(points)
        Tests to which cell each point belongs.
    transform(tr)
        Applies transformation tr to this universe. Returns a new universe.
    verbose_name()
        Gets verbose name of the universe.
    """

    def __init__(self, cells, name=0, verbose_name=None, comment=None):
        self._name = name
        self._comment = comment
        self._verbose_name = name if verbose_name is None else verbose_name
        self._cells = []

        for c in cells:
            self.add_cell(c, name_rule='keep')

    def __iter__(self):
        return iter(self._cells)

    def add_cell(self, cell, name_rule='new'):
        """Adds new cell to the universe.

        Modifies current universe.

        Parameters
        ----------
        cell : Body
            Cell to be added to the universe.
        name_rule : str
            Rule, what to do with entities' names. 'keep' - keep all names; in
            case of clashes NameClashError exception is raised. 'clash' - rename
            entity only in case of name clashes. 'new' - set new sequential name
            to all inserted entities.
        """
        surfs = self.get_surfaces()
        new_name = max(surfs.keys(), default=0) + 1
        surf_replace = {s: s for s in surfs.values()}
        cell_surfs = cell.shape.get_surfaces()
        for s in cell_surfs:
            if s not in surf_replace.keys():
                new_surf = s.copy()
                if name_rule == 'keep' and new_surf.name() in surfs.keys():
                    raise NameClashError(
                        "Surface name clash: {0}".format(s.name()))
                elif name_rule == 'new':
                    new_surf.rename(new_name)
                    new_name += 1
                elif name_rule == 'clash' and new_surf.name() in surfs.keys():
                    new_surf.rename(new_name)
                    new_name += 1
                surfs[new_surf.name()] = new_surf
                surf_replace[new_surf] = new_surf
        cell_names = {c.name() for c in self}
        new_cell = Body(cell.shape.replace_surfaces(surf_replace),
                        **cell.options)
        if name_rule == 'keep' and cell.name() in cell_names:
            raise NameClashError("Cell name clash: {0}".format(cell.name()))
        elif name_rule == 'new' or name_rule == 'clash' and cell.name() in cell_names:
            new_cell.rename(max(cell_names, default=0) + 1)
        new_cell.options['U'] = self
        self._cells.append(new_cell)

    def apply_fill(self, cell=None, universe=None, predicate=None):
        """Applies fill operations to all or selected cells or universes.

        Modifies current universe.

        Parameters
        ----------
        cell : Body or int
            Cell or name of cell which is filled by filling universe. The cell
            can only belong to this universe. Cells of inner universes are not
            taken into account.
        universe : Universe or int
            Filler-universe or its name. Cells, that have this universe as a
            filler will be filled. Only cells of this universe will be checked.
        predicate : func
            Function that accepts Body instance and return True, if this cell
            must be filled.
        """
        pass

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

    def check_names(self):
        """Checks, if there is name clashes.

        Returns
        -------
        result : bool
            True, if there is no name clashes.
        stat : dict
            Description of found clashes.
        """
        pass

    def copy(self):
        """Makes a copy of the universe."""
        return Universe(
            self._cells, name=self._name, verbose_name=self._verbose_name,
            comment=self._comment
        )

    def get_surfaces(self, recursive=False):
        """Gets all surfaces of the universe.

        Parameters
        ----------
        recursive : bool
            Whether to take surfaces of inner universes. Default: False -
            return surfaces of this universe only.

        Returns
        -------
        surfs : dict
            A dictionary of name->Surface.
        """
        surfs_set = set()
        for c in self:
            surfs_set.update(c.shape.get_surfaces())
            if recursive and 'FILL' in c.options.keys():
                surfs_set.update(c.options['FILL']['universe'].get_surfaces(
                    recursive).values())
        return {s.name(): s for s in surfs_set}

    def get_materials(self, recursive=False):
        """Gets all materials of the universe.

        Parameters
        ----------
        recursive : bool
            Whether to take materials of inner universes. Default: False -
            returns materials of this universe only.

        Returns
        -------
        mats : dict
            A dictionary of name->Material.
        """
        pass

    def get_composition(self, recursive=False):
        """Gets all compositions of the unvierse.

        Parameters
        ----------
        recursive : bool
            Whether to take composition of inner universes. Default: False -
            returns compositions of this universe only.

        Returns
        -------
        comps : dict
            A dictionary of name->Composition.
        """
        pass

    def get_transformations(self):
        """Gets all transformations of the universe."""
        pass

    def get_universes(self):
        """Gets all inner universes."""
        pass

    def name(self):
        """Gets numeric name of the universe."""
        return self._name

    def rename(self, start_cell=None, start_surf=None, start_mat=None,
               start_tr=None, name=None):
        """Renames all entities contained in the universe.

        All new names are sequential starting from the specified name. If name
        is None, than names of entities are leaved untouched.

        Parameters
        ----------
        start_cell : int
            Starting name for cells. Default: None.
        start_surf : int
            Starting name for surfaces. Default: None.
        start_mat : int
            Starting name for materials. Default: None.
        start_tr : int
            Starting name for transformations. Default: None.
        name : int
            Name for the universe. Default: None.
        """
        pass

    def save(self, filename, inner=True):
        """Saves the universe into file.

        Parameters
        ----------
        filename : str
            File name, universe to be saved to.
        inner : bool
            Whether to save inner universes too. If False, only this universe
            itself without fillers will be saved. Default: True.
        """
        pass

    def select(self, selector=None, inner=False):
        """Selects specified entities.

        Parameters
        ----------
        selector : func
            A function that accepts 1 argument, Body instance, and returns
            selected entities.
        inner : bool
            Whether to consider inner universes. Default: False - only this
            universe will be taken into account.

        Returns
        -------
        items : list
            List of selected items.
        """
        items = []
        taken_ids = set()
        for c in self:
            portion = selector(c)
            if inner:
                u = c.options.get('FILL', {}).get('universe', None)
                if u:
                    portion.extend(u.select(selector, True))
            for item in portion:
                if id(item) not in taken_ids:
                    taken_ids.add(id(item))
                    items.append(item)
        return items

    def simplify(self, box=GLOBAL_BOX, min_volume=1, split_disjoint=False):
        """Simplifies all cells of the universe.

        Modifies current universe.

        Parameters
        ----------
        box : Box
            Box, from which simplification process starts. Default: GLOBAL_BOX.
        min_volume : float
            Minimal volume of the box, when splitting process terminates.
        split_disjoint : bool
            Whether to split disjoint cells.
        """
        pass

    def test_points(self, points):
        """Finds cell to which each point belongs to.

        Parameters
        ----------
        points : array_like[float]
            An array of point coordinates. If there is only one point it has
            shape (3,); if there are n points, it has shape (n, 3).

        Returns
        -------
        result : np.ndarray[int]
            An array of cell indices to which a particular point belongs to.
            Its length equals to the number of points.
        """
        points = np.array(points)
        result = np.empty(points.size // 3)
        for i, c in enumerate(self._cells):
            test = c.shape.test_points(points)
            result[test == +1] = i
        return result

    def transform(self, tr):
        """Applies transformation tr to this universe. Returns a new universe.

        Parameters
        ----------
        tr : Transformation
            Transformation to be applied.

        Returns
        -------
        u_tr : Universe
            New transformed universe.
        """
        new_cells = [c.transform(tr) for c in self]
        return Universe(new_cells, name=self._name,
                        verbose_name=self._verbose_name, comment=self._comment)

    def verbose_name(self):
        """Gets verbose name of the universe."""
        return self._verbose_name

#     def copy(self):
#         """Makes a shallow copy of this universe."""
#         u = Universe([], name=self.name(), verbose_name=self.verbose_name,
#                      comment=self._comment)
#         u._cells = self._cells.copy()
#         return u
#
#     @property
#     def verbose_name(self):
#         return self._verbose_name
#
#     def bounding_box(self, tol=100, box=GLOBAL_BOX):
#         """Gets bounding box for the universe.
#
#         It finds all bounding boxes for universe cells and then constructs
#         box, that covers all cells' bounding boxes. The main purpose of this
#         method is to find boundaries of the model geometry to compare
#         transformation and surface objects for equality. It is recommended to
#         choose tol value not too small to reduce computation time.
#
#         Parameters
#         ----------
#         tol : float
#             Linear tolerance for the bounding box. The distance [in cm] between
#             every box's surface and universe in every direction won't exceed
#             this value. Default: 100 cm.
#         box : Box
#             Starting box for the search. The user must be sure that box covers
#             all geometry, because cells, that already contains corners of the
#             box will be considered as infinite and will be excluded from
#             analysis.
#
#         Returns
#         -------
#         bbox : Box
#             Universe bounding box.
#         """
#         boxes = []
#         for c in self._cells:
#             test = c.shape.test_points(box.corners)
#             if np.any(test == +1):
#                 continue
#             boxes.append(c.shape.bounding_box(tol=tol, box=box))
#         all_corners = np.empty((8 * len(boxes), 3))
#         for i, b in enumerate(boxes):
#             all_corners[i * 8: (i + 1) * 8, :] = b.corners
#         min_pt = np.min(all_corners, axis=0)
#         max_pt = np.max(all_corners, axis=0)
#         center = 0.5 * (min_pt + max_pt)
#         dims = max_pt - min_pt
#         return Box(center, *dims)
#
#     def fill(self, cells=None, recurrent=False, simplify=False, **kwargs):
#         """Fills cells that have fill option by filling universe cells.
#
#         The cells that initially were in this universe and had fill option
#         are replaced by filling universe. Geometry of the cell being filled
#         is used to bound cells of filler universe. Simplification won't be
#         done by default.
#
#         Parameters
#         ----------
#         cells : int or list[int]
#             Names of cell or cells, which must be filled. Default: None - all
#             universe cells will be considered.
#         recurrent : bool
#             Indicates if apply_fill of inner universes should be also invoked.
#             In this case, apply_fill is first invoked for inner universes.
#             Default: False.
#         simplify : bool
#             Whether to simplify universe cells after filling. Default: False.
#         kwargs : dict
#             Parameters for simplify method. See Body.simplify.
#
#         Returns
#         -------
#         new_universe : Universe
#             New filled universe.
#         """
#         if isinstance(cells, int):
#             cells = [cells]
#         new_cells = []
#         for c in self._cells:
#             name = c.get('name', 0)
#             if cells and name not in cells:
#                 new_cells.append(c)
#             else:
#                 new_cells.extend(
#                     c.fill(recurrent=recurrent, simplify=simplify, **kwargs)
#                 )
#         new_universe = self.copy()
#         new_universe._cells = new_cells
#         if simplify:
#             new_universe = new_universe.simplify(**kwargs)
#         return new_universe
#
#     def simplify(self, box=GLOBAL_BOX, split_disjoint=False,
#                  min_volume=MIN_BOX_VOLUME, trim_size=1):
#         """Simplifies this universe by simplifying all cells.
#
#         The simplification procedure goes in the following way.
#         # TODO: insert brief description!
#
#         Parameters
#         ----------
#         box : Box
#             Box where geometry should be simplified.
#         split_disjoint : bool
#             Whether to split disjoint geometries into separate geometries.
#         min_volume : float
#             The smallest value of box's volume when the process of box splitting
#             must be stopped.
#         trim_size : int
#             Max size of set to return. It is used to prevent unlimited growth
#             of the variant set.
#
#         Returns
#         -------
#         universe : Universe
#             Simplified version of this universe.
#         """
#         cells = []
#         for c in self._cells:
#             new_cell = c.simplify(
#                 box=box, split_disjoint=split_disjoint, min_volume=min_volume,
#                 trim_size=trim_size
#             )
#             if not new_cell.shape.is_empty():
#                 cells.append(new_cell)
#         u = self.copy()
#         u._cells = cells
#         return u
#
#     def get_surfaces(self):
#         """Gets all surfaces that discribe this universe.
#
#         Returns
#         -------
#         surfaces : set[Surface]
#             A set of all contained surfaces.
#         """
#         surfaces = set()
#         for c in self._cells:
#             cs = c.shape.get_surfaces()
#             surfaces.update(cs)
#         return surfaces
#
#     def get_materials(self):
#         """Gets all materials that belong to the universe.
#
#         Returns
#         -------
#         materials : set[Material]
#             A set of all materials that are contained in the universe.
#         """
#         materials = set()
#         for c in self._cells:
#             m = c.material()
#             if m is not None:
#                 materials.add(m)
#         return materials
#
#     def get_universes(self, recurrent=True):
#         """Gets names of all universes that are included in the universe.
#
#         Parameters
#         ----------
#         recurrent : bool
#             Whether to list universes that are not directly included in this one.
#             Default: True.
#
#         Returns
#         -------
#         universes : set[int or str]
#             A set of universe names. It can be either MCNP name or verbose name.
#         """
#         universes = set()
#         for c in self._cells:
#             fill = c.get('FILL')
#             if fill:
#                 universes.add(fill['universe'].verbose_name)
#                 if recurrent:
#                     universes.update(fill['universe'].get_universes(recurrent=True))
#         return universes
#
#     def select_universe(self, name):
#         """Selects given universe.
#
#         If no such universe present, KeyError is raised.
#
#         Parameters
#         ----------
#         name : int or str
#             Numeric or verbose name of universe to be selected.
#
#         Returns
#         -------
#         universe : Universe
#             Requested universe.
#         """
#         if name == self.name or name == self.verbose_name:
#             return self
#         for c in self._cells:
#             fill = c.get('FILL')
#             if fill:
#                 u = fill['universe']
#                 try:
#                     return u.select_universe(name)
#                 except KeyError:
#                     pass
#         raise KeyError("No such universe: {0}".format(name))
#
#     def select_cell(self, name):
#         """Selects given cell.
#
#         Cell must belong this universe.
#         If no such cell present, KeyError is raised.
#
#         Parameters
#         ----------
#         name : int
#             Name of cell.
#
#         Returns
#         -------
#         cell : Body
#             Requested cell.
#         """
#         for c in self._cells:
#             if c.get('name', None) == name:
#                 return c
#         raise KeyError("No such cell: {0}".format(name))
#
#     def rename(self, start_cell=1, start_surf=1, start_mat=1, start_tr=1,
#                name=None, verbose_name=None):
#         """Renames universe entities.
#
#         Modifies current universe. All nested universes must be renamed
#         separately if needed.
#
#         Parameters
#         ----------
#         start_cell : int
#             Starting name for cells. Default: 1.
#         start_surf : int
#             Starting name for surfaces. Default: 1.
#         start_mat : int
#             Starting name for compositions (MCNP materials). Default: 1.
#         start_tr : int
#             Starting name for transformations. Default: 1.
#         name : int
#             Name of this universe. Only needed if name must be changed.
#             Default: None.
#         verbose_name : str
#             Verbose name of this universe. Only needed if verbose name of the
#             universe must be changed. Default: None.
#         """
#         for s in self.get_surfaces():
#             s.options['name'] = start_surf
#             start_surf += 1
#         for comp in set(m.composition for m in self.get_materials()):
#             comp._options['name'] = start_mat
#             start_mat += 1
#         transformations = set()
#         for c in self._cells:
#             c['name'] = start_cell
#             start_cell += 1
#             tr = c.get('TRCL', None)
#             if tr and tr['name']:
#                 transformations.add(tr)
#             fill = c.get('FILL', {})
#             tr = fill.get('transform', None)
#             if tr and tr['name']:
#                 transformations.add(tr)
#         for tr in transformations:
#             tr['name'] = start_tr
#             start_tr += 1
#         if name is not None:
#             self._name = name
#         if verbose_name is not None:
#             self._verbose_name = verbose_name
#
#     def check_names(self):
#         """Checks if the universe entities have name clashes.
#
#         First it checks if there is name clashes in cells, surfaces, materials
#         and transformations that belong to the universe itself, and then with
#         all other nested universes.
#
#         Returns
#         -------
#         status : bool
#             Status of the check. True if everything is OK, False, otherwise.
#         report : dict
#             Dictionary of check results. Here found clashes are reported.
#         """
#         names = {}
#         self._collect_names(names)
#
#     def _collect_names(self, results):
#         if id(self) in results.keys():
#             return
#         names = {'cells': [], 'surfaces': [], 'compositions': [],
#                  'transformations': [], 'name': self._name}
#         for s in self.get_surfaces():
#             names['surfaces'].append(s.options.get('name', 0))
#         compositions = {m.composition for m in self.get_materials()}
#         for c in compositions:
#             names['compositions'].append(compositions._options.get('name', 0))
#         transformations = set()
#         for c in self:
#             names['cells'].append(c.get('name', 0))
#             tr = c.get('TRCL', None)
#             if tr is not None:
#                 transformations.add(tr)
#             fill = c.get('FILL', None)
#             if fill:
#                 u = fill['universe']
#                 tr = fill.get('transform', None)
#                 if tr:
#                     transformations.add(tr)
#                 u._collect_names(results)
#         for tr in transformations:
#             names['transformations'].append(tr._options.get('name', 0))
#         results[id(self)] = names
#
#     def save(self, filename, format_rules=None):
#         """Saves the universe to file.
#
#         Parameters
#         ----------
#         filename : str
#             Name of file.
#         format_rules : dict
#             A dictionary of format rules.
#         """
#         universe = [self.select_universe(name) for name in self.get_universes()]
#         items = [str(self._verbose_name)]
#         items.append('C cell section. Main universe.')
#         for c in self._cells:
#             items.append(str(c))
#         for u in universe:
#             items.append('C start of universe {0}'.format(u.name))
#             for c in u:
#                 items.append(str(c))
#             items.append('C end of universe {0}'.format(u.name))
#         items.append('')
#         items.append('C Surface section')
#         used_surfs = set()
#         for s in sorted(self.get_surfaces(), key=lambda x: x.options['name']):
#             if s not in used_surfs:
#                 items.append(str(s))
#                 used_surfs.add(s)
#         for u in universe:
#             items.append('C start of surfaces of universe {0}'.format(u.name))
#             for s in sorted(u.get_surfaces(), key=lambda x: x.options['name']):
#                 if s not in used_surfs:
#                     items.append(str(s))
#                     used_surfs.add(s)
#             items.append('C end of surfaces of universe {0}'.format(u.name))
#         items.append('')
#         items.append('C data section')
#         compositions = set()
#         for cell in self:
#             mat = cell.material()
#             if mat is not None:
#                 cc = mat.composition
#                 if cc._options['name'] not in compositions:
#                     compositions.add(cc._options['name'])
#                     items.append(str(cc))
#         for u in universe:
#             items.append('C start of materials of universe {0}'.format(u.name))
#             for cell in u:
#                 m = cell.material()
#                 if m is None: continue
#                 cc = m.composition
#                 if cc._options['name'] not in compositions:
#                     compositions.add(cc._options['name'])
#                     items.append(str(cc))
#             items.append('C end of materials of universe {0}'.format(u.name))
#         items.append('')
#
#         with open(filename, 'w') as f:
#             for item in items:
#                 f.write(item)
#                 f.write('\n')
#
#     def _get_last_cell_name(self):
#         """Gets last used cell name.
#
#         Returns
#         -------
#         name : int
#             Last cell name of the universe.
#         """
#         names = [c.name() for c in self._cells]
#         if not names:
#             return None
#         return np.max(names)
#
#     def _get_surf_replace_dictionary(self, surfaces, keep_name=False):
#         """Gets replace dictionary for surfaces.
#
#         Parameters
#         ----------
#         surfaces : iterable
#             Surfaces for which replace dictionary have to be created.
#         keep_name : bool
#             Whether to keep origin surface names. Default: False.
#
#         Returns
#         -------
#         replace_dict : dict
#             Replace dictionary.
#         """
#         universe_surfaces = {s: s for s in self.get_surfaces()}
#         replace_dict = {}
#         last_surf_name = np.max([s.name() for s in universe_surfaces.keys()])
#         for s in surfaces:
#             rep_sur = universe_surfaces.get(s, None)
#             if rep_sur is None:
#                 rep_sur = s.copy()
#                 if not keep_name:
#                     rep_sur.rename(last_surf_name + 1)
#                     last_surf_name += 1
#             replace_dict[s] = rep_sur
#         return replace_dict
#
#     def _register_cell(self, cell, keep_name=False):
#         """Registers new cell in the universe.
#
#         Parameters
#         ----------
#         cell : Body
#             Cell to be registered.
#         keep_name : bool
#             Whether to keep cell's name.
#
#         Returns
#         -------
#         new_cell : Body
#             New registered version of the cell.
#         """
#         last_name = self._get_last_cell_name()
#         replace_dict = self._get_surf_replace_dictionary(
#             cell.shape.get_surfaces(), keep_name=keep_name
#         )
#         new_cell = Body(
#             cell.shape.replace_surfaces(replace_dict),
#             **cell.options
#         )
#         if not keep_name:
#             new_cell.rename(last_name + 1)
#         return new_cell
#
#
# class EntityNamingError(Exception):
#     pass
