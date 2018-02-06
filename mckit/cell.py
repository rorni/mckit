# -*- coding: utf-8 -*-

from itertools import product
from functools import reduce
from operator import xor

import numpy as np

from .surface import Surface
from .constants import EX, EY, EZ, GLOBAL_BOX, MIN_BOX_VOLUME
from .fmesh import Box


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
        Indicate if this term is mandatory. Important in union operations.
    """
    _operations = (_intersection, _union)
    _operation_code = {'I': 0, 'U': 1, 0: 0, 1: 1}

    def __init__(self, operation, positive=set(), negative=set(),
                 mandatory=True, order=None):
        self._opc = self._operation_code[operation]
        self.positive = set()
        self.negative = set()
        self.mandatory = mandatory
        self.order = order

        pos_new = {p for p in self.positive if isinstance(p, Surface) or
                   not p.is_empty()}
        neg_new = {p for p in self.negative if isinstance(p, Surface) or
                   not p.is_empty()}
        if operation == _intersection and (len(pos_new) < len(positive) or
                                           len(neg_new) < len(negative)):
            positive = set()
            negative = set()
        else:
            positive = pos_new
            negative = neg_new

        if positive and negative and not positive.intersection(negative):
            self.positive.update(positive)
            self.negative.update(negative)
        else:
            self.positive.update(positive)
            self.negative.update(negative)
        # calculate hash value
        pos = reduce(xor, [hash(s) for s in self.positive], 0)
        neg = reduce(xor, [~hash(s) for s in self.negative], ~0)
        op_hash = 0 if self._opc == 0 else ~0
        self._hash_value = xor(xor(pos, neg), op_hash)

    def __hash__(self):
        return self._hash_value

    def __eq__(self, other):
        return other.is_of_type(self._opc) and \
               self.positive == other.positive and \
               self.negative == other.negative

    def __str__(self):
        sep = ' ' if self._opc == 0 else ':'
        toks = []
        for s in self.positive:
            if isinstance(s, GeometryNode):
                toks.append(str(s))
            else:
                toks.append(str(s.options['name']))
        for s in self.negative:
            if isinstance(s, GeometryNode):
                toks.append('#(' + str(s) + ')')
            else:
                toks.append('-' + str(s.options['name']))
        if self._opc == 1:
            toks = '(' + toks + ')'
        return sep.join(toks)

    def propagate_complement(self):
        new_neg = set()
        for g in self.negative:
            if isinstance(g, GeometryNode):
                self.positive.add(g.complement())
            else:
                new_neg.add(g)
        for g in self.positive:
            if isinstance(g, GeometryNode):
                g.propagate_complement()

    def clean(self, del_unnecessary=False):
        """Cleans geometry description.

        Parameters
        ----------
        del_unnecessary : bool
            Indicate to delete nodes which are not mandatory.

        Returns
        -------
        cleaned : GeometryNode
            Cleaned geometry.
        """
        pos_cleaned = set()
        neg_cleaned = set()
        for g in self.positive:
            if not g.mandatory and del_unnecessary:
                continue
            if isinstance(g, GeometryNode):
                if g.is_empty():
                    continue
                gc = g.clean(del_unnecessary)
                if gc.is_of_type(self._opc):
                    pos_cleaned.update(gc.positive)
                    neg_cleaned.update(gc.negative)
                else:
                    pos_cleaned.add(gc)
            else:
                pos_cleaned.add(g)
        for g in self.negative:
            if not g.mandatory and del_unnecessary:
                continue
            if isinstance(g, GeometryNode):
                if g.is_empty():
                    continue
                gc = g.clean(del_unnecessary)
                if gc.is_of_type(self._opc):
                    pos_cleaned.update(gc.positive)
                    neg_cleaned.update(gc.negative)
                else:
                    neg_cleaned.add(gc)
            else:
                neg_cleaned.add(g)
        return GeometryNode(self._opc, positive=pos_cleaned,
                            negative=neg_cleaned, mandatory=self.mandatory)

    def is_empty(self):
        """Checks if this geometry is an empty set."""
        return len(self.positive) + len(self.negative) == 0

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
            pos = [a.test_box(box) for a in self.positive]
            neg = [-1 * a.test_box(box) for a in self.negative]
            return reduce(self._operations[self._opc], pos + neg)

        pos = {}
        neg = {}
        for g in self.positive:
            if isinstance(g, Surface):
                res = g.test_box(box)
                simp = g
            else:
                res, simp = g.test_box(box, return_simple=True)
            if res not in pos.keys():
                pos[res] = set()
            pos[res].add(simp)
        for g in self.negative:
            if isinstance(g, Surface):
                res = g.test_box(box)
                simp = g
            else:
                res, simp = g.test_box(box, return_simple=True)
            res *= -1
            if res not in neg.keys():
                neg[res] = set()
            neg[res].add(simp)
        # Now calculate the result
        result = reduce(self._operations[self._opc], list(pos.keys()) + list(neg.keys()))
        pos_set = pos.get(result, set())
        neg_set = neg.get(result, set())
        if result == 0:
            simp_geom = {GeometryNode(self._opc, positive=pos_set,
                                      negative=neg_set,
                                      order=self.order)}
        elif result == -1 and self._opc == 0 or result == +1 and self._opc == 1:
            simp_geom = set()
            for g in pos_set:
                simp_geom.add(GeometryNode(self._opc, positive={g},
                                           order=self.order))
            for g in neg_set:
                simp_geom.add(GeometryNode(self._opc, negative={g},
                                           order=self.order))
        else:
            for g in pos_set | neg_set:
                g.mandatory = False
            simp_geom = {GeometryNode(self._opc, positive=pos_set,
                                      negative=neg_set, order=self.order)}
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
        pos = [a.test_point(p) for a in self.positive]
        neg = [-1 * a.test_point(p) for a in self.negative]
        return reduce(self._operations[self._opc], pos + neg)

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
        pos = {a.transform(tr) for a in self.positive}
        neg = {a.transform(tr) for a in self.negative}
        return GeometryNode(self._operations[self._opc], positive=pos,
                            negative=neg, mandatory=self.mandatory)

    def get_surfaces(self):
        """Gets all surfaces that describes the geometry.

        Returns
        -------
        surfaces : set[Surface]
            A set of surfaces that take part in geometry description.
        """
        surfaces = set()
        for arg in self.positive.union(self.negative):
            if isinstance(arg, Surface):
                surfaces.add(arg)
            else:
                surfaces.update(arg.get_surfaces())
        return surfaces

    def complexity(self):
        """Gets complexity of calculations."""
        comp = 0
        for arg in self.positive.union(self.negative):
            if isinstance(arg, Surface):
                comp += 1
            else:
                comp += arg.complexity()
        return comp

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
                ag = adg1.merge_geometries(adg2)
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
        pos = self._merge_node_sets(self.positive, other.positive)
        neg = self._merge_node_sets(self.negative, other.negative)
        return GeometryNode(self._opc, positive=pos, negative=neg,
                            mandatory=self.mandatory or other.mandatory,
                            order=self.order)

    @staticmethod
    def _merge_node_sets(a, b):
        a_dict = {t.order: t for t in a}
        b_dict = {t.order: t for t in b}
        for k, v in a_dict.items():
            if k not in b_dict.keys():
                b_dict[k] = v
            else:
                if isinstance(v, GeometryNode):
                    b_dict[k] = b_dict[k].merge_nodes(v)
                else:
                    b_dict[k].mandatory |= v
        return set(b_dict.values())

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
        return GeometryNode('I', positive={self, other})

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
        return GeometryNode('U', positive={self, other})

    def complement(self):
        """Gets a complement of the geometry.

        Parameters
        ----------
        go_deep : bool
            Indicate to propagate complement operation deeper.

        Returns
        -------
        result : GeometryNode
            A resulting complement.
        """
        opc = (self._opc + 1) % 2
        pos = self.negative
        neg = self.positive
        return GeometryNode(opc, positive=pos, negative=neg,
                            mandatory=self.mandatory, order=self.order)

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
                operands.append(op)
                operands.append(0)
            elif op == 'C':
                sign = operands.pop()
                operands.append((sign + 1) % 2)
            else:
                args = [set(), set()]
                for i in range(2):
                    sign = operands.pop()
                    args[sign].add(operands.pop())
                geom = GeometryNode(op, positive=args[0], negative=args[1])
                operands.append(geom)
                operands.append(0)
        a_geom = operands.pop()
        return a_geom

    def _give_orders(self, start=1):
        """Give order values to all nodes."""
        for g in self.positive | self.negative:
            if isinstance(g, Surface):
                g.order = start
                start += 1
            else:
                start = g._give_orders(start)
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
        GeometryNode.__init__(self, geometry._opc, positive=geometry.positive,
                              negative=geometry.negative)
        dict.__init__(self, options)

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
        self._give_orders()
        simple_geoms = super(Cell, self).simplify(box, min_volume)
        # remove not mandatory terms
        geometries = set()
        for sg in simple_geoms:
            terms = [t for t in sg.terms if t.mandatory]
            geometries.add(AdditiveGeometry(*terms))
        simplest = reduce(AdditiveGeometry.get_simplest, geometries)
        # TODO: implement splitting of disjoint cells.
        return simplest

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
