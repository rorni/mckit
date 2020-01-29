# -*- coding: utf-8 -*-
from typing import Iterable, Dict, Any, List
from collections import defaultdict
from attr import attrs, attrib
import numpy as np
from click import progressbar

from .body import Body, Shape
from .card import Card
# noinspection PyUnresolvedReferences,PyUnresolvedReferences,PyUnresolvedReferences
from mckit.box import GLOBAL_BOX, Box
from .transformation import Transformation
from .material import Material
from .surface import Plane

__all__ = [
    'Universe', 'produce_universes', 'NameClashError', 'cell_selector',
    'surface_selector'
]


class NameClashError(Exception):
    pass


def cell_selector(cell_names):
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


def surface_selector(surface_names):
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
    common_materials : set
        A set of common materials. Default: None.

    Methods
    -------
    add_cells(cell)
        Adds new cell to the universe.
    apply_fill(cell, universe)
        Applies fill operations to all or selected cells or universes.
    bounding_box(tol, box)
        Gets bounding box of the universe.
    copy()
        Makes a copy of the universe.
    find_common_materials()
        Finds all common materials among universes.
    get_surfaces()
        Gets all surfaces of the universe.
    get_materials()
        Gets all materials of the universe.
    get_compositions()
        Gets all compositions of the universe.
    get_transformations()
        Gets all transformations of the universe.
    get_universes()
        Gets all inner universes.
    name()
        Gets numeric name of the universe.
    name_clashes()
        Checks if there is name clashes.
    rename(start_cell, start_surf, start_mat, start_tr)
        Renames all entities contained in the universe.
    save(filename)
        Saves the universe to file.
    select(cell, selector)
        Selects specified entities.
    set_common_materials(com_mat)
        Sets new common materials for universe and all nested universes.
    simplify(box, split_disjoint, min_volume)
        Simplifies all cells of the universe.
    test_points(points)
        Tests to which cell each point belongs.
    transform(tr)
        Applies transformation tr to this universe. Returns a new universe.
    verbose_name()
        Gets verbose name of the universe.
    """

    def __init__(self, cells, name=0, verbose_name=None, comment=None,
                 name_rule='keep', common_materials=None):
        self._name = name
        self._comment = comment
        self._verbose_name = name if verbose_name is None else verbose_name
        self._cells = []
        if common_materials is None:
            common_materials = set()
        self._common_materials = common_materials

        self.add_cells(cells, name_rule=name_rule)

    @property
    def cells(self):
        return self._cells

    def __iter__(self):
        return iter(self._cells)

    def __len__(self):
        return len(self._cells)

    def __setitem__(self, key: int, value: Body):
        raise NotImplementedError("Renaming rules should be applied.")

    def __getitem__(self, item):
        return self._cells.__getitem__(item)

    # def __hash__(self):
    #     return reduce(xor, map(hash, self.cells), 0)
    #
    # def __eq__(self, other):
    #     return reduce(and_, map(eq, zip(self._cells, other.cells), True))

    def has_equivalent_cells(self, other):
        if len(self) != len(other):
            return False
        for i, c in enumerate(self):
            if not c.is_equivalent_to(other[i]):
                return False
        return True

    def add_cells(self, cells, name_rule='new'):
        """Adds new cell to the universe.

        Modifies current universe.

        Parameters
        ----------
        cells : Body or list[Body]
            An array of cells to be added to the universe.
        name_rule : str
            Rule, what to do with entities' names. 'keep' - keep all names; in
            case of clashes NameClashError exception is raised. 'clash' - rename
            entity only in case of name clashes. 'new' - set new sequential name
            to all inserted entities.
        """
        if isinstance(cells, Body):
            cells = [cells]
        surfs = self.get_surfaces()
        surf_replace = {s: s for s in surfs}
        comps = self._common_materials.union(self.get_compositions())
        comp_replace = {c: c for c in comps}
        cell_names = {c.name() for c in self}
        surf_names = {s.name() for s in surfs}
        comp_names = {c.name() for c in comps}
        for cell in cells:
            if cell.shape.is_empty():
                continue

            new_shape = self._get_cell_replaced_shape(
                cell, surf_replace, surf_names, name_rule
            )

            new_cell = Body(new_shape, **cell.options)
            mat = new_cell.material()
            if mat:
                new_comp = Universe._update_replace_dict(
                    mat._composition, comp_replace, comp_names, name_rule,
                    'Material'
                )
                new_cell.options['MAT'] = Material(
                    composition=new_comp, density=mat.density
                )

            if name_rule == 'keep' and cell.name() in cell_names:
                raise NameClashError("Cell name clash: {0}".format(cell.name()))
            elif name_rule == 'new' or name_rule == 'clash' and cell.name() in cell_names:
                new_name = max(cell_names, default=0) + 1
                new_cell.rename(new_name)
            cell_names.add(new_cell.name())
            new_cell.options['U'] = self
            self._cells.append(new_cell)

    def set_common_materials(self, common_materials):
        """Sets common materials for this one and all nested universes.

        Parameters
        ----------
        common_materials : set
            A set of common materials.
        """
        universes = self.get_universes()
        for u in universes:
            u._set_common_materials(common_materials)

    def _set_common_materials(self, common_materials):
        """Sets common materials for individual universe.

        Parameters
        ----------
        common_materials : set
            A set of common_materials
        """
        self._common_materials = common_materials
        cmd = {m: m for m in common_materials}
        for c in self:
            mat = c.material()
            if mat:
                comp = mat.composition
                if comp in common_materials:
                    c.options['MAT'] = Material(
                        composition=cmd[comp], density=mat.density
                    )

    @staticmethod
    def _get_cell_replaced_shape(cell, surf_replace, surf_names, name_rule):
        cell_surfs = cell.shape.get_surfaces()
        replace_dict = {}
        for s in cell_surfs:
            if isinstance(s, Plane):
                rev_s = Plane(-s._v, -s._k)
                if rev_s in surf_replace.keys():
                    rev_s = surf_replace[rev_s]
                    replace_dict[s] = Shape('C', rev_s)
                    continue
            replace_dict[s] = Universe._update_replace_dict(
                s, surf_replace, surf_names, name_rule, 'Surface'
            )
        return cell.shape.replace_surfaces(replace_dict)

    @staticmethod
    def _update_replace_dict(entity, replace, names, rule, err_desc):
        if entity not in replace.keys():
            new_entity = entity.copy()
            if rule == 'keep' and new_entity.name() in names:
                print(entity.mcnp_repr())
                for c in replace.keys():
                    print(c.mcnp_repr())
                raise NameClashError(
                    "{0} name clash: {1}".format(err_desc, entity.name())
                )
            elif rule == 'new' or rule == 'clash' and new_entity.name() in names:
                new_name = max(names, default=0) + 1
                new_entity.rename(new_name)
                names.add(new_name)
            replace[new_entity] = new_entity
            names.add(new_entity.name())
            return new_entity
        else:
            return replace[entity]

    @staticmethod
    def _fill_check(predicate):
        def _predicate(cell):
            fill = cell.options.get('FILL', None)
            if fill:
                return predicate(cell)
            else:
                return False

        return _predicate

    def alone(self):
        """Gets this universe alone, without inner universes.

        Returns
        -------
        u : Universe
            A copy of the universe with FILL cards removed.
        """
        cells = []
        for c in self:
            options = {k: v for k, v in c.options.items() if k != 'FILL'}
            cells.append(Body(c.shape, **options))
        return Universe(cells)

    def apply_fill(self, cell=None, universe=None, predicate=None, name_rule='new'):
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
        if not cell and not universe and not predicate:
            predicate = lambda c: True
        elif cell:
            predicate = lambda c: c.name() == cell
        elif universe:
            predicate = lambda c: c.options['FILL']['universe'].name() == universe
        predicate = self._fill_check(predicate)
        extra_cells = []
        del_indices = []
        for i, c in enumerate(self):
            if predicate(c):
                extra_cells.extend(c.fill())
                del_indices.append(i)
        for i in reversed(del_indices):
            self._cells.pop(i)
        self.add_cells(extra_cells, name_rule=name_rule)

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

    def copy(self):
        """Makes a copy of the universe."""
        return Universe(
            self._cells, name=self._name, verbose_name=self._verbose_name,
            comment=self._comment, common_materials=self._common_materials
        )

    def find_common_materials(self):
        """Finds common materials among universes included.

        Returns
        -------
        common_mats : set
            A set of common materials.
        """
        comp_count = defaultdict(lambda: 0)
        for u in self.get_universes():
            for c in u.get_compositions():
                comp_count[c] += 1
        common_mats = {c for c, cnt in comp_count.items() if cnt > 1}
        return common_mats

    def get_surfaces(self, inner=False):
        """Gets all surfaces of the universe.

        Parameters
        ----------
        inner : bool
            Whether to take surfaces of inner universes. Default: False -
            return surfaces of this universe only.

        Returns
        -------
        surfs : set
            A set of surfaces that belong to the universe.
        """
        surfs = set()
        for c in self:
            surfs.update(c.shape.get_surfaces())
            if inner and 'FILL' in c.options.keys():
                surfs.update(c.options['FILL']['universe'].get_surfaces(inner))
        return surfs

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

    def get_compositions(self, exclude_common=False):
        """Gets all compositions of the unvierse.

        Parameters
        ----------
        exclude_common : bool
            Exclude common compositions from the result. Default: False.

        Returns
        -------
        comps : set
            A set of Composition objects.
        """
        comps = set()
        for c in self:
            mat = c.material()
            if mat:
                comps.add(mat.composition)
        if exclude_common:
            comps.difference_update(self._common_materials)
        return comps

    def get_transformations(self):
        """Gets all transformations of the universe."""
        pass   # TODO dvp: add transformations

    def get_universes(self):
        """Gets all inner universes.

        Returns
        -------
        universes : set
            A set of universes.
        """
        universes = {self}
        for c in self:
            if 'FILL' in c.options:
                u = c.options['FILL']['universe']  # TODO dvp: add transformations
                universes.update(u.get_universes())
        return universes

    def name(self):
        """Gets numeric name of the universe."""
        return self._name

    def name_clashes(self):
        """Checks, if there is name clashes.

        Returns
        -------
        stat : dict
            Description of found clashes. If no clashes - the dictionary is
            empty.
        """
        univ = self.get_universes()
        stat = {}
        cells = {u: list(map(Card.name, u)) for u in univ}
        surfs = {u: list(map(Card.name, u.get_surfaces())) for u in univ}
        mats = {None: list(map(Card.name, self._common_materials))}
        for u in univ:
            mats[u] = list(map(Card.name, u.get_compositions().difference(self._common_materials)))
        univs = {u: [u.name()] for u in univ}
        cstat = Universe._produce_stat(cells)
        if cstat:
            stat['cell'] = cstat
        sstat = Universe._produce_stat(surfs)
        if sstat:
            stat['surf'] = sstat
        mstat = Universe._produce_stat(mats)
        if mstat:
            stat['material'] = mstat
        ustat = Universe._produce_stat(univs)
        if ustat:
            stat['universe'] = ustat
        return stat

    @staticmethod
    def _produce_stat(names):
        stat = defaultdict(list)
        for u, u_names in names.items():
            for name in u_names:
                stat[name].append(u)
        return Universe._clean_stat_dict(stat)

    @staticmethod
    def _clean_stat_dict(stat):
        new_stat = {}
        for k, v in stat.items():
            if len(v) > 1:
                # v.sort(key=Universe.name)
                new_stat[k] = set(v)
        return new_stat

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
        if name:
            self._name = name
        if start_cell:
            for c in self:
                c.rename(start_cell)
                start_cell += 1
        if start_surf:
            surfs = self.get_surfaces()
            for s in sorted(surfs, key=Card.name):
                s.rename(start_surf)
                start_surf += 1
        if start_mat:
            mats = self.get_compositions()
            for m in sorted(mats, key=Card.name):
                if m not in self._common_materials:
                    m.rename(start_mat)
                    start_mat += 1

    def save(self, filename, encoding="cp1251"):
        """Saves the universe into file.

        Parameters
        ----------
        filename : str, Path
            File name, universe to be saved to.
        encoding: str
            Encoding ot the output file
        """
        result = self.name_clashes()
        if result:
            raise NameClashError('Impossible to save model.')
        universes = self.get_universes()
        cells = []
        surfaces = []
        materials = list(sorted(self._common_materials, key=Card.name))
        for u in sorted(universes, key=Universe.name):
            cells.extend(sorted(u, key=Card.name))
            surfaces.extend(sorted(u.get_surfaces(), key=Card.name))
            materials.extend(sorted(u.get_compositions(True), key=Card.name))
        cards = [str(self.verbose_name())]
        cards.extend(map(Card.mcnp_repr, cells))
        cards.append('')
        cards.extend(map(Card.mcnp_repr, surfaces))
        cards.append('')
        cards.extend(map(Card.mcnp_repr, materials))
        cards.append('')
        with open(filename, mode='w', encoding=encoding) as f:
            f.write('\n'.join(cards))

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

    def simplify(self, box=GLOBAL_BOX, min_volume=1, split_disjoint=False,
                 verbose=True):
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
        verbose : bool
            Turns on verbose output. Default: True.
        """
        new_cells = []
        if verbose:
            fmt_fun = lambda x: "Simplifying cell #{0}".format(x.name() if x else x)
            uiter = progressbar(self, item_show_func=fmt_fun).__enter__()
        else:
            uiter = self

        for c in uiter:
            cs = c.simplify(box=box, min_volume=min_volume)
            if not cs.shape.is_empty():
                new_cells.append(cs)

        if verbose:
            print('Universe {0} simplification has been finished.'.format(self.name()))
            print('{0} empty cells were deleted.'.format(len(self._cells) - len(new_cells)))

        self._cells = new_cells

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
        return Universe(new_cells, name=self._name, name_rule='clash',
                        verbose_name=self._verbose_name, comment=self._comment)

    def verbose_name(self):
        """Gets verbose name of the universe."""
        return self._verbose_name

    @property
    def comment(self):
        return self._comment


@attrs
class _UniverseCellsGroup:
    universe: Universe = attrib()
    cells: List[Body] = attrib()


def produce_universes(cells: Iterable[Body]) -> Universe:
    """Creates groups from cells.

    The function groups all the by 'universe' option value,
    and creates corresponding groups.

    Parameters
    ----------
    cells : Iterable[Body]
        Cells to process.

    Returns
    -------
    universe : Universe
        The top level universe with name = 0.
    """
    groups: Dict[int, _UniverseCellsGroup] = {}
    for c in cells:
        universe_no: int = c.options.get('U', 0)
        if universe_no in groups:
            groups[universe_no].cells.append(c)
        else:
            new_group = _UniverseCellsGroup(universe=Universe([], universe_no), cells=[c])
            groups[universe_no] = new_group
    for c in cells:
        fill: Dict[str, Any] = c.options.get('FILL', None)
        if fill is not None:
            fill_universe_no = fill['universe']
            fill['universe'] = groups[fill_universe_no].universe
    for group in groups.values():
        group.universe.add_cells(group.cells, name_rule='keep')
    top_universe = groups[0].universe
    top_universe.set_common_materials(top_universe.find_common_materials())
    return top_universe


