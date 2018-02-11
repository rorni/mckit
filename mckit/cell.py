# -*- coding: utf-8 -*-

from functools import reduce
from itertools import product
from operator import xor

import numpy as np

from .constants import GLOBAL_BOX, MIN_BOX_VOLUME
from .fmesh import Box
from .surface import Surface


def _complement(arg):
    """Finds complement to the given set.

    | C | -1 |  0 | +1 |
    +---+----+----+----+
    |   | +1 |  0 | -1 |
    +---+----+----+----+

    Parameters
    ----------
    arg : int or np.ndarray[int]
        Argument for complement operation.

    Returns
    -------
    result : int or np.ndarray[int]
        The result of operation.
    """
    return -1 * arg


def _identity(arg):
    """Identity operation. Returns argument."""
    return arg


def _intersection(arg1, arg2):
    """Finds intersection.

    |  I | -1 |  0 | +1 |
    +----+----+----+----+
    | -1 | -1 | -1 | -1 |
    +----+----+----+----+
    |  0 | -1 |  0 |  0 |
    +----+----+----+----+
    | +1 | -1 |  0 | +1 |
    +----+----+----+----+

    Parameters
    ----------
    arg1, arg2 : int, np.ndarray[int]
        Operands.

    Returns
    -------
    result : int or np.ndarray[int]
        The result of operation.
    """
    return np.minimum(arg1, arg2)


def _union(arg1, arg2):
    """Finds union.

    |  U | -1 |  0 | +1 |
    +----+----+----+----+
    | -1 | -1 |  0 | +1 |
    +----+----+----+----+
    |  0 |  0 |  0 | +1 |
    +----+----+----+----+
    | +1 | +1 | +1 | +1 |
    +----+----+----+----+

    Parameters
    ----------
    arg1, arg2 : int, np.ndarray[int]
        Operands.

    Returns
    -------
    result : int or np.ndarray[int]
        The result of operation.
    """
    return np.maximum(arg1, arg2)


class GeometryNode:
    """Describes elementary operation.

    Parameters
    ----------
    operation : str
        An operation identifier to be applied to args. It can be 'U' for union
        and 'I' for intersection.
    positive : set
        A set of surfaces that have positive sense with respect to the part
        of space described by GeometryNode.
    negative : set
        A set of surfaces that have negative sense with respect to the part of
        space described by GeometryNode.
    mandatory : bool
        Indicate if this term is _mandatory. Important in union operations.
    """
    _operations = (_intersection, _complement, _union, _identity)
    _operation_code = {'I': 0, 'U': 2, 'C': 1, 'E': 3}
    _INTERSECTION = 0
    _COMPLEMENT = 1
    _UNION = 2
    _IDENTITY = 3

    def __init__(self, operation, *args, mandatory=True, node_id=None):
        if isinstance(operation, int) and 0 <= operation <= 3:
            self._opc = operation
        else:
            self._opc = self._operation_code[operation]

        if self._opc == self._COMPLEMENT or self._opc == self._IDENTITY:
            if len(args) > 1:
                raise ValueError('Incorrect number of parameters.')
            self._args = args[0]
        else:
            self._args = set(args)

        self._mandatory = mandatory
        self._id = node_id
        # calculate hash value
        self._calculate_hash()

    def _calculate_hash(self):
        if self._opc == self._COMPLEMENT or self._opc == self._IDENTITY:
            arg_hash = hash(self._args)
        else:
            arg_hash = reduce(xor, [hash(s) for s in self._args], 0)
        op_hash = self._opc
        self._hash_value = xor(arg_hash, op_hash)

    def _invert_opc(self):
        return (self._opc + 2) % 4

    def __hash__(self):
        return self._hash_value

    def __eq__(self, other):
        return isinstance(other, GeometryNode) and \
               other.is_of_type(self._opc) and self._args == other._args

    def __str__(self):
        if self._opc == self._COMPLEMENT:
            arg = self._args
            text = '-{0}'.format(arg.options['name'])
        elif self._opc == self._IDENTITY:
            arg = self._args
            text = str(arg.options['name'])
        elif self._opc == self._INTERSECTION:
            text = ' '.join([str(a) for a in self._args])
        elif self._opc == self._UNION:
            text = '(' + ':'.join([str(a) for a in self._args]) + ')'
        return text

    def get_simplest(self, other):
        if self.complexity() < other.complexity():
            return self
        else:
            return other

    def clean(self, del_unnecessary=False):
        """Cleans geometry description.

        Parameters
        ----------
        del_unnecessary : bool
            Indicate to delete nodes which are not _mandatory.

        Returns
        -------
        cleaned : GeometryNode
            Cleaned geometry.
        """
        if isinstance(self._args, Surface):
            return self
        args = set()
        opc = self._opc
        for a in self._args:
            ca = a.clean(del_unnecessary)
            if (a._mandatory or not del_unnecessary) and not ca.is_empty():
                if ca.is_of_type(opc):
                    args.update(ca._args)
                else:
                    args.add(ca)

        comp_set = {a.complement() for a in args}
        overlap = comp_set.intersection(args)
        if len(overlap) > 0:
            if opc == self._INTERSECTION:
                args = set()
            else:
                for o in overlap:
                    args.remove(o)

        if len(args) == 1:
            a = args.pop()
            opc = a._opc
            args = a._args if isinstance(a._args, set) else {a._args}

        return GeometryNode(opc, *args, mandatory=self._mandatory,
                            node_id=self._id)

    def is_empty(self):
        """Checks if this geometry is an empty set."""
        if self._args is None or isinstance(self._args, set) and len(self._args) == 0:
            return True
        else:
            return False
        #return len(self._args) == 0

    def is_of_type(self, opc):
        """Checks if the geometry node has the same type of operation."""
        return self._opc == opc

    def test_box(self, box, return_simple=False):
        """Checks if the geometry intersects the box.

        Parameters
        ----------
        box : Box
            Box for checking.
        return_simple : bool
            Indicate whether to return simpler form of the geometry. Unimportant
            surfaces for judging this box (and all contained boxes, of course)
            are removed then.

        Returns
        -------
        result : int
            Test result. It equals one of the following values:
            +1 if the box lies entirely inside the cell.
             0 if the box (probably) intersects the cell.
            -1 if the box lies outside the cell.
        simple_geoms : set[AdditiveGeometry]
            List of simple variants of this geometry. Optional.
        """
        if not return_simple:
            if isinstance(self._args, Surface):
                return self._operations[self._opc](self._args.test_box(box))
            else:
                tests = [a.test_box(box) for a in self._args]
                return reduce(self._operations[self._opc], tests)

        if isinstance(self._args, Surface):
            return self._args.test_box(box), {self}

        ans = {}
        for g in self._args:
            res, simp = g.test_box(box, return_simple=True)
            if res not in ans.keys():
                ans[res] = set()
            ans[res].update(simp)
        # Now calculate the result
        result = reduce(self._operations[self._opc], ans.keys())
        ans_set = ans.get(result, set())
        if result == 0:
            simp_geom = {GeometryNode(self._opc, *ans_set, node_id=self._id)}
        elif result == -1 and self._opc == 0 or result == +1 and self._opc == 1:
            simp_geom = set()
            for g in ans_set:
                simp_geom.add(GeometryNode(self._opc, g, node_id=self._id))
        else:
            for g in ans_set:
                g._mandatory = False
            simp_geom = {GeometryNode(self._opc, *ans_set, node_id=self._id)}
        return result, simp_geom

    def test_point(self, p):
        """Tests whether point(s) p belong to this geometry.

        Parameters
        ----------
        p : array_like[float]
            Coordinates of point(s) to be checked. If it is the only one point,
            then p.shape=(3,). If it is an array of points, then
            p.shape=(num_points, 3).

        Returns
        -------
        result : int or numpy.ndarray[int]
            If the point lies inside geometry, then +1 value is returned.
            If point lies on the boundary, 0 is returned.
            If point lies outside of the geometry, -1 is returned.
            Individual point - single value, array of points - array of
            ints of shape (num_points,) is returned.
        """
        if isinstance(self._args, Surface):
            r = self._operations[self._opc](self._args.test_point(p))
        else:
            tests = [a.test_point(p) for a in self._args]
            r = reduce(self._operations[self._opc], tests)
        return r

    def transform(self, tr):
        """Transforms this geometry.

        Parameters
        ----------
        tr : Transform
            Transformation object.

        Returns
        -------
        geom : GeometryNode
            Transformed geometry.
        """
        if isinstance(self._args, Surface):
            args = [self._args.transform(tr)]
        else:
            args = [a.transform(tr) for a in self._args]
        return GeometryNode(self._opc, *args, mandatory=self._mandatory,
                            node_id=self._id)

    def get_surfaces(self):
        """Gets all surfaces that describes the geometry.

        Returns
        -------
        surfaces : set[Surface]
            A set of surfaces that take part in geometry description.
        """
        surfaces = set()
        for arg in self._args:
            if isinstance(arg, Surface):
                surfaces.add(arg)
            else:
                surfaces.update(arg.get_surfaces())
        return surfaces

    def complexity(self):
        """Gets complexity of calculations."""
        if isinstance(self._args, Surface):
            return 1
        else:
            return sum(map(GeometryNode.complexity, self._args))

    def bounding_box(self, box=GLOBAL_BOX, tol=1.0):
        """Gets bounding box for this cell.

        Parameters
        ----------
        box : Box
            Initial box approach. If None, default box is centered around
            origin and have dimensions 1.e+4 cm.
        tol : float
            Absolute tolerance, when the process of box reduction has to be
            stopped.

        Returns
        -------
        box : Box
            The box that bounds this cell.
        """
        if self.test_box(box) != 0:
            raise ValueError("Initial box size is too small.")
        for dim in range(3):
            # adjust upper bound
            lower = 0
            while (box.scale[dim] - lower) > tol:
                ratio = 0.5 * (lower + box.scale[dim]) / box.scale[dim]
                box1, box2 = box.split(dim, ratio)
                t2 = self.test_box(box2)
                if t2 == -1:
                    box = box1
                else:
                    lower = box1.scale[dim]
            # adjust lower bound
            upper = 0
            while (box.scale[dim] - upper) > tol:
                ratio = 0.5 * (box.scale[dim] - upper) / box.scale[dim]
                box1, box2 = box.split(dim, ratio)
                t1 = self.test_box(box1)
                if t1 == -1:
                    box = box2
                else:
                    upper = box2.scale[dim]
        return box

    def volume(self, box=GLOBAL_BOX, min_volume=MIN_BOX_VOLUME,
               rand_points_num=1000):
        """Calculates volume of cell part inside the box.

        Parameters
        ----------
        box : Box
            The box inside which the cell volume is calculated. Default:
            GLOBAL_BOX - intended for total volume calculation.
        min_volume : float
            Minimal volume when the splitting process is stopped. Default:
            MIN_BOX_VOLUME.
        rand_points_num : int
            The number of random points used to estimate cell volume inside
            the box, that is less than min_volume.

        Returns
        -------
        vol : float
            Calculated volume.
        """
        result, simple_geoms = self.test_box(box, return_simple=True)
        geom = simple_geoms.pop()
        if result == +1:
            vol = box.volume()
        elif result == -1:
            vol = 0
        else:
            if box.volume() > min_volume:
                box1, box2 = box.split()
                vol1 = geom.volume(box1, min_volume=min_volume,
                                   rand_points_num=rand_points_num)
                vol2 = geom.volume(box2, min_volume=min_volume,
                                   rand_points_num=rand_points_num)
                vol = vol1 + vol2
            else:
                points = box.generate_random_points(rand_points_num)
                inside = np.count_nonzero(geom.test_point(points) == +1)
                vol = box.volume() * inside / rand_points_num
        return vol

    def simplify(self, box=GLOBAL_BOX, min_volume=MIN_BOX_VOLUME):
        """Finds simpler geometry representation.

        Parameters
        ----------
        box : Box
            Initial box for which simpler representation is being found.
        min_volume : float
            Minimal box volume, when splitting precess should stopped.

        Returns
        -------
        result : set[GeometryNode]
            A set of simpler geometries.
        """
        # TODO: review algorithm
        result, simple_geoms = self.test_box(box, return_simple=True)
        if result != 0 or box.volume() <= min_volume:
            return simple_geoms

        # This is the case result == 0.
        simple = set()
        # comple = []  It is not used for now. It is intended to count
        # complexities of geometries for their selection.
        for geom in simple_geoms:
            c = geom.complexity()
            if c <= 1:
                simple.add(geom)
                # comple.append(c)
                continue
            # If geometry is too complex and box is too large -> we split box.
            box1, box2 = box.split()
            simple_geoms1 = geom.simplify(box1, min_volume=min_volume)
            simple_geoms2 = geom.simplify(box2, min_volume=min_volume)
            # Now merge results
            for adg1, adg2 in product(simple_geoms1, simple_geoms2):
                ag = adg1.merge_nodes(adg2)
                c = ag.complexity()
                if c > 0:
                    simple.add(ag)
                    # comple.append(c)
        # simple_sort = [simple[i] for i in np.argsort(comple)[:15]]
        return simple  # _sort

    def merge_nodes(self, other):
        """Merges descriptions of two geometries.

        Parameters
        ----------
        other : AdditiveGeometry
            Other geometry.

        Returns
        -------
        new_geom : AdditiveGeometry
            New merged geometry.
        """
        # TODO: review algorithm
        if self._opc == self._COMPLEMENT or self._opc == self._IDENTITY:
            args = [self._args]
        else:
            s_dict = {t._id: t for t in self._args}
            o_dict = {t._id: t for t in other._args}
            for k, v in o_dict.items():
                if k not in s_dict.keys():
                    s_dict[k] = v
                else:
                    s_dict[k] = s_dict[k].merge_nodes(v)
            args = s_dict.values()
        m = self._mandatory or other._mandatory
        return GeometryNode(self._opc, *args, mandatory=m, node_id=self._id)

    def intersection(self, other):
        """Gets an intersection of geometries.

        Parameters
        ----------
        other : GeometryNode
            Other geometry.

        Returns
        -------
        result : GeometryNode
            A resulting intersection.
        """
        return GeometryNode('I', *(self._args | other._args))

    def union(self, other):
        """Gets an union of geometries.

        Parameters
        ----------
        other : GeometryNode
            Other geometry.

        Returns
        -------
        result : GeometryNode
            A resulting union.
        """
        return GeometryNode('U', *(self._args | other._args))

    def complement(self):
        """Gets a complement of the geometry.

        Returns
        -------
        result : GeometryNode
            A resulting complement.
        """
        opc = self._invert_opc()
        if isinstance(self._args, set):
            args = [a.complement() for a in self._args]
        else:
            args = [self._args]
        return GeometryNode(opc, *args, mandatory=self._mandatory,
                            node_id=self._id)

    @staticmethod
    def from_polish_notation(polish):
        """Creates AdditiveGeometry instance from reversed Polish notation.

        Parameters
        ----------
        polish : list
            List of surfaces and operations written in reversed Polish Notation.

        Returns
        -------
        a_geom : GeometryNode
            The geometry represented by AdditiveGeometry instance.
        """
        operands = []
        for op in polish:
            if isinstance(op, Surface):
                operands.append(GeometryNode('E', op))
            elif op == 'C':
                operands.append(operands.pop().complement())
            else:
                arg1 = operands.pop()
                arg2 = operands.pop()
                operands.append(GeometryNode(op, arg1, arg2))
        return operands.pop().clean()

    def _set_node_ids(self, start=1):
        """Give order values to all nodes."""
        self._id = start
        start += 1
        if isinstance(self._args, set):
            for g in self._args:
                start = g._set_node_ids(start)
        return start


class Cell(dict, GeometryNode):
    """Represents MCNP's cell.

    Parameters
    ----------
    geometry : list or AdditiveGeometry
        Geometry expression. It is either a list of Surface instances and
        operations (reverse Polish notation is used) or AdditiveGeometry object.
    options : dict
        A set of cell's options.

    Methods
    -------
    intersection(other)
        Returns an intersection of this cell with the other.
    populate(universe)
        Fills this cell by universe.
    simplify(box, split_disjoint, min_volume)
        Simplifies cell description.
    transform(tr)
        Applies transformation tr to this cell.
    union(other)
        Returns an union of this cell with the other.
    """
    def __init__(self, geometry, **options):
        if isinstance(geometry, list):
            geometry = GeometryNode.from_polish_notation(geometry)
        args = {geometry._args} if isinstance(geometry._args, Surface) else geometry._args
        GeometryNode.__init__(self, geometry._opc, *args)
        dict.__init__(self, options)

    def __str__(self):
        text = [str(self['name'])]
        text.append(GeometryNode.__str__(self))
        return ' '.join(text)

    def intersection(self, other):
        """Gets an intersection if this cell with the other.

        Other cell is a geometry that bounds this one. The resulting cell
        inherits all options of this one (the caller).

        Parameters
        ----------
        other : Cell
            Other cell.

        Returns
        -------
        cell : Cell
            The result.
        """
        geometry = GeometryNode.intersection(self, other)
        return Cell(geometry, **self)

    def union(self, other):
        """Gets an union if this cell with the other.

        The resulting cell inherits all options of this one (the caller).

        Parameters
        ----------
        other : Cell
            Other cell.

        Returns
        -------
        cell : Cell
            The result.
        """
        geometry = GeometryNode.union(self, other)
        return Cell(geometry, **self)

    def simplify(self, box=GLOBAL_BOX, split_disjoint=False,
                 min_volume=MIN_BOX_VOLUME):
        """Simplifies this cell by removing unnecessary surfaces.

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

        Returns
        -------
        simple_cell : Cell
            Simplified version of this cell.
        """
        self._set_node_ids()
        simple_geoms = super(Cell, self).simplify(box, min_volume)
        for i, g in enumerate(simple_geoms):
            print(i, ' -> ', g)

        geometries = set()
        for sg in simple_geoms:
            geometries.add(sg.clean(del_unnecessary=False))
        simplest = reduce(GeometryNode.get_simplest, geometries)
        # TODO: implement splitting of disjoint cells.
        return Cell(simplest, **self)

    def populate(self, universe=None):
        """Fills this cell by filling universe.

        If this cell doesn't contain fill options, the cell itself is returned
        as list of length 1. Otherwise a list of cells from filling universe
        bounded by cell being filled is returned.

        Parameters
        ----------
        universe : Universe
            Universe which cells fill this one. If None, universe from 'FILL'
            option will be used. If no such universe, the cell itself will be
            returned.

        Returns
        -------
        cells : list[Cell]
            Resulting cells.
        """
        if universe is None:
            if 'FILL' in self.keys():
                universe = self['FILL']
            else:
                return [self]
        cells = []
        for c in universe.cells:
            new_cell = c.intersection(self)  # because properties like MAT, etc
                                             # must be as in filling cell.
            if 'U' in self.keys():
                new_cell['U'] = self['U']    # except universe.
            cells.append(new_cell)
        return cells

    def transform(self, tr):
        """Applies transformation to this cell.

        Parameters
        ----------
        tr : Transform
            Transformation to be applied.

        Returns
        -------
        cell : Cell
            The result of this cell transformation.
        """
        geometry = GeometryNode.transform(self, tr)
        return Cell(geometry, **self)
