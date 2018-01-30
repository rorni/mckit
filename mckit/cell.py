# -*- coding: utf-8 -*-

from itertools import product
from functools import reduce

import numpy as np

from .surface import Surface
from .constants import EX, EY, EZ, GLOBAL_BOX, MIN_BOX_VOLUME
from .fmesh import Box


class Cell(dict):
    """Represents MCNP's cell.

    Parameters
    ----------
    geometry_expr : list
        Geometry expression. List of Surface instances and operations. Reverse
        Polish notation is used.
    options : dict
        A set of cell's options.

    Methods
    -------
    bounding_box()
        Gets bounding box for this cell.
    get_surfaces()
        Returns a set of surfaces that bound this cell.
    populate()
        Fills this cell by universe.
    simplify(split)
        Simplifies cell description.
    test_point(p)
        Tests whether point(s) p belong to this cell (lies inside it).
    test_box(box)
        Checks whether this cell intersects with the box.
    transform(tr)
        Applies transformation tr to this cell.
    """
    def __init__(self, geometry_expr, **options):
        dict.__init__(self, options)
        self._expression = geometry_expr.copy()

    def simplify(self, global_box=GLOBAL_BOX, tol=1.0, split=False):
        """Simplifies description of this cell.
        
        Parameters
        ----------
        global_box : Box
            Global box where cell is considered.
        tol : float
            Absolute tolerance, when the process of box reduction has to be 
            stopped.
        split : bool
            Indicate whether this cell should be split into simpler ones.
            
        Returns
        -------
        cell : Cell
            The simplified version of cell.
        """
        bbox = self.bounding_box(box=global_box, tol=tol, adjust=True)
        outer_boxes = bbox.get_outer_boxes(global_box=global_box)


    def bounding_box(self, box=GLOBAL_BOX, tol=1.0, adjust=False):
        """Gets bounding box for this cell.
        
        Parameters
        ----------
        box : Box
            Initial box approach. If None, default box is centered around
            origin and have dimensions 1.e+4 cm.
        tol : float
            Absolute tolerance, when the process of box reduction has to be 
            stopped.
        adjust : bool
            Specifies whether orientation of bounding box has to be corrected.
        
        Returns
        -------
        box : Box
            The box that bounds this cell.
        """
        if self.test_box(box) != 0:
            raise ValueError("Initial box size is too small.")
        for dim in range(3):
            # adjust upper bound
            mlt = 1
            ratio = 0.5
            while (1 - ratio) * box.scale[dim] > tol:
                box1, box2 = box.split(dim, ratio)
                t2 = self.test_box(box2)
                if t2 == -1:
                    box = box1
                    ratio = 0.5 * mlt
                else:
                    ratio = 0.5 * (1 + ratio)
                    mlt = (3 * ratio - 1) / ratio
            # adjust lower bound
            mlt = 1
            ratio = 0.5
            while ratio * box.scale[dim] > tol:
                box1, box2 = box.split(dim, ratio)
                t1 = self.test_box(box1)
                if t1 == -1:
                    box = box2
                    ratio = 0.5 * mlt
                else:
                    ratio = 0.5 * ratio
                    mlt = 0.5 * ratio / (1 - ratio)

        if adjust:
            # TODO: add correction of box orientation.
            pass
        return box

    def get_surfaces(self):
        """Gets a set of surfaces that bound this cell.

        Returns
        -------
        surfaces : set
            Surfaces that bound this cell.
        """
        surfaces = set()
        for op in self._expression:
            if isinstance(op, Surface):
                surfaces.add(op)
        return surfaces

    def populate(self):
        """Fills this cell by filling universe.

        If this cell doesn't contain fill options, the cell itself is returned
        as list of length 1. Otherwise a list of cells from filling universe
        bounded by cell being filled is returned.

        Returns
        -------
        cells : list
            Resulting cells.
        """
        raise NotImplemented

    def test_point(self, p):
        """Tests whether point(s) p belong to this cell.

        Parameters
        ----------
        p : array_like[float]
            Coordinates of point(s) to be checked. If it is the only one point,
            then p.shape=(3,). If it is an array of points, then
            p.shape=(num_points, 3).

        Returns
        -------
        result : int or numpy.ndarray[int]
            If the point lies inside cell, then +1 value is returned.
            If point lies on the boundary, 0 is returned.
            If point lies outside of the cell, -1 is returned.
            Individual point - single value, array of points - array of
            ints of shape (num_points,) is returned.
        """
        return self._geometry_test(p, 'test_point')

    def calculate_volume(self, box, geometry=None, accuracy=1, pool_size=10000):
        """Calculates volume of the cell inside the box.
        
        Parameters
        ----------
        box : Box
            The box.
        accuracy : float
            accuracy
        """
        if geometry is None:
            geometry = self._expression.copy()
        sense, new_geom = self._test_box(box, geometry.copy())
        if sense == +1:
            volume = box.volume()
        elif box.volume() <= accuracy**3:
            points = box.generate_random_points(pool_size)
            cell_result = self.test_point(points)
            volume = np.count_nonzero(cell_result == +1) / pool_size * box.volume()
        elif sense == 0:
            box1, box2 = box.split()
            volume = self.calculate_volume(box1, accuracy=accuracy, geometry=new_geom.copy()) + \
                     self.calculate_volume(box2, accuracy=accuracy, geometry=new_geom.copy())
        else:
            volume = 0
        return volume

    def test_box(self, box):
        """Checks whether this cell intersects with the box.

        Parameters
        ----------
        box : Box
            Region of space, which intersection with the cell is checked.

        Returns
        -------
        result : int
            Test result. It equals one of the following values:
            +1 if the box lies entirely inside the cell.
             0 if the box (probably) intersects the cell.
            -1 if the box lies outside the cell.
        """
        return self._geometry_test(box, 'test_box')

    def _geometry_test(self, arg, method_name):
        """Performs geometry test.

        Parameters
        ----------
        arg : array_like
            Objects which should be tested.
        method_name : str
            The name of Surface's method that must be invoked.

        Returns
        -------
        test_results : np.ndarray
            Test results.
        """
        stack = []
        for op in self._expression:
            if isinstance(op, Surface):
                stack.append(getattr(op, method_name)(arg))
            elif op == 'C':
                stack.append(_complement(stack.pop()))
            elif op == 'I':
                stack.append(_intersection(stack.pop(), stack.pop()))
            elif op == 'U':
                stack.append(_union(stack.pop(), stack.pop()))
        return stack.pop()

    def _test_box(self, box, geometry):
        op = geometry.pop()
        ng = []
        if isinstance(op, Surface):
            s = op.test_box(box)
        elif op == 'C':
            a, ng = self._test_box(box, geometry)
            s = _complement(a)
        elif op == 'I' or op == 'U':
            a, ng1 = self._test_box(box, geometry)
            b, ng2 = self._test_box(box, geometry)
            if op == 'I':
                s = _intersection(a, b)
            elif op == 'U':
                s = _union(a, b)
            if s == 0:
                ng.extend(ng1)
                ng.extend(ng2)
        else:
            s = op
        if s == 0:
            ng.append(op)
        else:
            ng = [s]
        return s, ng

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
        new_expr = []
        for op in self._expression:
            if isinstance(op, Surface):
                new_expr.append(op.transform(tr))
            else:
                new_expr.append(op)
        return Cell(new_expr, **self)


class GeometryTerm:
    """Geometry that is represented only by intersection of surfaces.

    Parameters
    ----------
    positive : set
        A set of surfaces that have positive sense with respect to the part
        of space described by GeometryTerm.
    negative : set
        A set of surfaces that have negative sense with respect to the part of
        space described by GeometryTerm.
    order : int
        A position in the list of terms of AdditiveGeometry instance. It is
        used for internal purposes.

    Methods
    -------
    intersection(other)
        Gets an intersection of this term with other.
    complement()
        Gets complement to this term.
    is_superset(other)
        Checks whether this term is a superset of other.
    is_subset(other)
        Checks whether this term is a subset of other.
    is_empty()
        Checks whether this term is an empty set.
    complexity()
        Gets the complexity of term description.
    test_box(box)
        Checks whether this term intersects with the box.

    Attributes
    ----------
    positive : set[Surface]
        A set of surfaces that have positive sense.
    negative : set[Surface]
        A set of surfaces that have negative sense.
    order : int
        Position in the terms list.
    """
    def __init__(self, positive=None, negative=None, order=None):
        self.positive = set()
        self.negative = set()
        self.order = order
        if positive and negative and not positive.intersection(negative):
            self.positive.update(positive)
            self.negative.update(negative)
        elif positive and not negative:
            self.positive.update(positive)
        elif negative and not positive:
            self.negative.update(negative)

    def intersection(self, other):
        """Gets an intersection of this term with other.

        Parameters
        ----------
        other : GeometryTerm
            Other geometry term.

        Returns
        -------
        result : GeometryTerm
            New resulting term.
        """
        if self.is_empty() or other.is_empty():
            return GeometryTerm(order=self.order)
        return GeometryTerm(positive=self.positive.union(other.positive),
                            negative=self.negative.union(other.negative),
                            order=self.order)

    def complement(self):
        """Gets a complement to this term.

        Returns
        -------
        comp : GeometryTerm
            A complement.
        """
        return GeometryTerm(positive=self.negative, negative=self.positive)

    def is_superset(self, other):
        """Checks whether this term is a superset of other.

        Parameters
        ----------
        other : GeometryTerm
            Other term.

        Returns
        -------
        test_result : bool
            True, if this term is a superset of other one.
        """
        return other.is_subset(self)

    def is_subset(self, other):
        """Checks whether this term is a subset of other.

        Parameters
        ----------
        other : GeometryTerm
            Other term.

        Returns
        -------
        test_result : bool
            True, if this term is a subset of other one.
        """
        if self.is_empty():
            return True
        if other.is_empty():
            return False
        ps = self.positive.issuperset(other.positive)
        ns = self.negative.issuperset(other.negative)
        return ps and ns

    def is_empty(self):
        """Checks if this set is empty."""
        return len(self.positive) == 0 and len(self.negative) == 0

    def test_box(self, box, return_simple=False):
        """Checks if this cell intersects with the box.

        Parameters
        ----------
        box : Box
            Box for checking.
        return_simple : bool
            Indicate whether to return simpler form of the term. Unimportant
            surfaces for judging this box (and all contained boxes, of course)
            are removed then.

        Returns
        -------
        result : int
            Test result. It equals one of the following values:
            +1 if the box lies entirely inside the cell.
             0 if the box (probably) intersects the cell.
            -1 if the box lies outside the cell.
        simple_terms : list[GeometryTerm]
            List of simple variants of this term. Optional.
        """
        if self.is_empty():
            return (-1, [GeometryTerm(order=self.order)]) if return_simple else -1

        pos = {}
        neg = {}
        for s in self.positive:
            res = s.test_box(box)
            if res not in pos.keys():
                pos[res] = set()
            pos[res].add(s)
        for s in self.negative:
            res = -1 * s.test_box(box)
            #      ^ -1 means take complement to the every entity in the set.
            if res not in neg.keys():
                neg[res] = set()
            neg[res].add(s)

        # Then take minimum element - this corresponds to the intersection
        # operation. See _intersection function below.
        result = min(list(pos.keys()) + list(neg.keys()))
        if return_simple:
            pos_set = pos.get(result, set())
            neg_set = neg.get(result, set())
            if result == 0:
                new_term = [GeometryTerm(positive=pos_set, negative=neg_set,
                                         order=self.order)]
            elif result == -1:
                new_term = []
                for s in pos_set:
                    new_term.append(GeometryTerm(positive={s},
                                                 order=self.order))
                for s in neg_set:
                    new_term.append(GeometryTerm(negative={s},
                                                 order=self.order))
            else:
                new_term = [GeometryTerm(order=self.order)]
            return result, new_term
        else:
            return result

    def complexity(self):
        """Gets complexity of term description.

        The complexity is the number of surfaces that are needed to describe
        the geometry of this term. If it is 0 - geometry is an empty set.

        Returns
        -------
        result : int
            Geometry complexity.
        """
        return len(self.positive) + len(self.negative)


class AdditiveGeometry:
    """A geometry that is described by a union of GeometryTerm's.

    Parameters
    ----------
    terms : list[GeometryTerm]
        A list of terms this geometry consists from.

    Methods
    -------
    intersection(other)
        Gets an intersection of this geometry with other.
    complement()
        Gets a complement to this geometry.
    union(other)
        Gets a union of this geometry with other.
    test_box(box)
        Checks if this geometry intersects the box.
    simplify(box)
        Simplifies this geometry.
    complexity()
        Gets the complexity of the additive geometry description.
    """
    def __init__(self, *terms):
        self.terms = []
        n = len(terms)
        for i in range(n):
            if terms[i].is_empty():
                continue
            for j in range(n):
                if i != j and terms[i].is_subset(terms[j]):
                    break
            else:
                self.terms.append(terms[i])

    def union(self, other):
        """Gets a union of this geometry with other.

        Parameters
        ----------
        other : AdditiveGeometry or GeometryTerm
            Other geometry for union.

        Returns
        -------
        result : AdditiveGeometry
            The result of union.
        """
        if isinstance(other, GeometryTerm):
            other = AdditiveGeometry(other)
        return AdditiveGeometry(*(self.terms + other.terms))

    def intersection(self, other):
        """Gets an intersection of this geometry with other.

        Parameters
        ----------
        other : AdditiveGeometry or GeometryTerm
            Other geometry for intersection.

        Returns
        -------
        result : AdditiveGeometry
            The result of intersection.
        """
        if isinstance(other, GeometryTerm):
            other = AdditiveGeometry(other)
        terms = [a.intersection(b) for a, b in product(self.terms, other.terms)]
        return AdditiveGeometry(*terms)

    def complement(self):
        """Gets complement to this set.

        Returns
        -------
        result : AdditiveGeometry
            The result of complement.
        """
        unions = []
        for t in self.terms:
            pos_t = [GeometryTerm(positive={a}) for a in t.negative] + \
                    [GeometryTerm(negative={a}) for a in t.positive]
            unions.append(AdditiveGeometry(*pos_t))
        return reduce(AdditiveGeometry.intersection, unions)

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
        simple_geoms : list[AdditiveGeometry]
            List of simple variants of this geometry. Optional.
        """
        if not return_simple:
            # max - gets a union of terms. See _union function below.
            return np.amax([t.test_box(box) for t in self.terms])

        terms = {}  # test_result -> list of terms that lead to this result.
        for t in self.terms:
            res, t_geoms = t.test_box(box, return_simple=True)
            if res not in terms.keys():
                terms[res] = []
            terms.append(t_geoms)

        result = max(terms.keys())
        if result == +1:
            simple_geoms = [AdditiveGeometry(*ts) for ts in product(*terms[1])]
        else:
            simple_geoms = [AdditiveGeometry(*terms[result])]
        return result, simple_geoms

    def simplify(self, box, split_disjoint=False, min_volume=MIN_BOX_VOLUME):
        """Simplifies this geometry by removing unnecessary surfaces.

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
        simple : AdditiveGeometry or list[AdditiveGeometry]
            Simple form of this geometry.
        """
        result, simple_geoms = self.test_box(box, return_simple=True)
        if result == -1:
            return simple_geoms

        # This is the case result == 0 or 1.
        simple = []
        for geom in simple_geoms:
            if geom.complexity() == 1 or box.volume() <= min_volume:
                simple.append(geom)
                continue
            # If geometry is too complex and box is too large -> we split box.
            box1, box2 = box.split()
            simp_geoms1 = geom.simplify(box1, split_disjoint=split_disjoint,
                                        min_volume=min_volume)
            simp_geoms2 = geom.simplify(box2, split_disjoint=split_disjoint,
                                        min_volume=min_volume)
            # Now merge results
            for adg1, adg2 in product(simp_geoms1, simp_geoms2):
                simple.append(adg1.merge_geometries(adg2))
        return simple

    def merge_geometries(self, other):
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
        # TODO: consider simplification of terms that occur only from -1 terms.
        # TODO: When merging at last they should be avoided if they have not
        # TODO: took part in merging procedure.
        self_t = {t.order: t for t in self.terms}
        other_t = {t.order: t for t in other.terms}
        for k, t in other_t.items():
            if k not in self_t.keys():
                self_t[k] = t
            else:
                self_t[k] = self_t[k].intersection(t)
        return AdditiveGeometry(*self_t.values())

    def complexity(self):
        """Gets complexity of geometry description."""
        complexity = 0
        for t in self.terms:
            complexity += t.complexity()
        return complexity

    @staticmethod
    def from_polish_notation(polish):
        """Creates AdditiveGeometry instance from reversed Polish notation.

        Parameters
        ----------
        polish : list
            List of surfaces and operations written in reversed Polish Notation.

        Returns
        -------
        a_geom : AdditiveGeometry
            The geometry represented by AdditiveGeometry instance.
        """
        operands = []
        for op in polish:
            if isinstance(op, Surface):
                operands.append(AdditiveGeometry(GeometryTerm(positive={op})))
            elif op == 'C':
                g = operands.pop()
                operands.append(g.complement())
            elif op == 'I':
                g1 = operands.pop()
                g2 = operands.pop()
                operands.append(g1.intersection(g2))
            elif op == 'U':
                g1 = operands.pop()
                g2 = operands.pop()
                operands.append(g1.union(g2))
        a_geom = operands.pop()
        if isinstance(a_geom, GeometryTerm):
            a_geom = AdditiveGeometry(a_geom)
        return a_geom


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

