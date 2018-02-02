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
    intersection(other)
        Returns an intersection of this cell with the other.
    populate(universe)
        Fills this cell by universe.
    simplify(box, split_disjoint, min_volume)
        Simplifies cell description.
    test_point(p)
        Tests whether point(s) p belong to this cell (lies inside it).
    test_box(box)
        Checks whether this cell intersects with the box.
    transform(tr)
        Applies transformation tr to this cell.
    union(other)
        Returns an union of this cell with the other.
    """
    def __init__(self, geometry_expr, **options):
        dict.__init__(self, options)
        if isinstance(geometry_expr, AdditiveGeometry):
            self._geometry = geometry_expr
        else:
            self._geometry = AdditiveGeometry(geometry_expr)

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
        geometry = self._geometry.intersection(other._geometry)
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
        geometry = self._geometry.union(other._geometry)
        return Cell(geometry, **self)

    def simplify(self, box=GLOBAL_BOX, split_disjoint=False,
                 min_volume=MIN_BOX_VOLUME):
        """Simplifies description of this cell.

        Parameters
        ----------
        box : Box
            Global box where cell is considered. Default: GLOBAL_BOX.
        split_disjoint : bool
            Indicate whether this cell should be split if it consists of two
            or more disjoint parts. Default: False - no split.
        min_volume : float
            Minimal volume, achieving which the process of box splitting is
            stopped and result is returned.

        Returns
        -------
        simple_cell : Cell or list[Cell]
            The simplified version of cell. If splitting take place, then list
            of simple cells is returned.
        """
        raise NotImplementedError

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
            raise NotImplementedError
        return box

    def get_surfaces(self):
        """Gets a set of surfaces that bound this cell.

        Returns
        -------
        surfaces : set
            Surfaces that bound this cell.
        """
        return self._geometry.get_surfaces()

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
        return self._geometry.test_point(p)

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
        return self._geometry.test_box(box)

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
        geometry = self._geometry.transform(tr)
        return Cell(geometry, **self)


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
    mandatory : bool
        Indicate whether this term is necessary for geometry. Default: True.

    Methods
    -------
    intersection(other)
        Gets an intersection of this term with other.
    complement()
        Gets complement to this term.
    get_surfaces()
        Gets a set of surfaces from which the term consists of.
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
    test_point(p)
        Checks whether point p belongs to geometry described by the term.
    transform(tr)
        Applies transformation tr to this term.

    Attributes
    ----------
    positive : set[Surface]
        A set of surfaces that have positive sense.
    negative : set[Surface]
        A set of surfaces that have negative sense.
    order : int
        Position in the terms list.
    """
    def __init__(self, positive=None, negative=None, order=None,
                 mandatory=True):
        self.positive = set()
        self.negative = set()
        self.order = order
        self.mandatory = mandatory
        if positive and negative and not positive.intersection(negative):
            self.positive.update(positive)
            self.negative.update(negative)
        elif positive and not negative:
            self.positive.update(positive)
        elif negative and not positive:
            self.negative.update(negative)

    def get_surfaces(self):
        """Gets a set of surfaces this term consists of."""
        surfaces = set()
        surfaces.update(self.positive)
        surfaces.update(self.negative)
        return surfaces

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
                            order=self.order,
                            mandatory=self.mandatory or other.mandatory)

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
                                                 order=self.order,
                                                 mandatory=False))
                for s in neg_set:
                    new_term.append(GeometryTerm(negative={s},
                                                 order=self.order,
                                                 mandatory=False))
            else:
                new_term = [GeometryTerm(order=self.order)]
            return result, new_term
        else:
            return result

    def test_point(self, p):
        """Tests whether point(s) p belong to this term geometry.

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
        pos_test = [s.test_point(p) for s in self.positive]
        neg_test = [_complement(s.test_point(p)) for s in self.negative]
        return reduce(_intersection, pos_test + neg_test)

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

    def transform(self, tr):
        """Applies transformation to this term.

        Parameters
        ----------
        tr : Transform
            Transformation to be applied.

        Returns
        -------
        term : GeometryTerm
            The result of this term transformation.
        """
        positive = {s.transform(tr) for s in self.positive}
        negative = {s.transform(tr) for s in self.negative}
        return GeometryTerm(positive=positive, negative=negative,
                            order=self.order, mandatory=self.mandatory)


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
    contains(other)
        Checks if this geometry contains other as a subset.
    equivalent(other)
        Checks if this geometry description is equivalent to the other one.
    get_surfaces()
        Gets a set of surfaces this geometry consists of.
    test_point(p)
        Tests if point p belongs to this geometry.
    transform(tr)
        Transforms this geometry.
    """
    def __init__(self, *terms):
        self.terms = []
        n = len(terms)
        for i in range(n):
            if terms[i].is_empty():
                continue
            for t in (self.terms + list(terms[i+1:])):
                if terms[i].is_subset(t):
                    break
            else:
                self.terms.append(terms[i])

    def __str__(self):
        out = []
        for t in self.terms:
            tl = {}
            if len(t.negative) > 0:
                tl['negative'] = {s.options['name'] for s in
                                  t.negative}
            if len(t.positive) > 0:
                tl['positive'] = {s.options['name'] for s in
                                  t.positive}
            out.append(str(tl))
        return '[' + ', '.join(out) + ']'

    def get_surfaces(self):
        """Gets a set of surfaces this geometry consists of."""
        surfaces = set()
        for t in self.terms:
            surfaces.update(t.get_surfaces())
        return surfaces

    def contains(self, other):
        """Checks if this geometry contains other as a subset.

        This is, say, a naive method of testing. It does not take into account
        the real surface shape. It is based only on surface names. In order to
        clarify if this geometry really contains the other, use simplify method.
        This geometry is believed to contain the other if every other term is
        a subset of some term of this geometry.

        Parameters
        ----------
        other : AdditiveGeometry
            Other geometry.

        Returns
        -------
        result : bool
            True if other geometry is a subset of this one. False otherwise.
        """
        for t in other.terms:
            for t0 in self.terms:
                if t.is_subset(t0):
                    break
            else:
                return False
        return True

    def equivalent(self, other):
        """Checks if this geometry description is equivalent to the other one.

        Parameters
        ----------
        other : AdditiveGeometry
            Other Geometry.

        Returns
        -------
        result : bool
            True if this geometry description is equivalent to the other.
            Equivalent means that the descriptions of geometries consist from
            the same terms.
        """
        return self.contains(other) and other.contains(self)

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
            terms[res].append(t_geoms)

        result = max(terms.keys())
        # print('=>', result, terms)
        if result == +1:
            simple_geoms = [AdditiveGeometry(ts[0]) for ts in terms[1]]
        elif result == -1:
            simple_geoms = [AdditiveGeometry(*ts) for ts in product(*terms[-1])]
        else:
            simple_geoms = [AdditiveGeometry(*[ts[0] for ts in terms[result]])]
        return result, simple_geoms

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
        test_term = [t.test_point(p) for t in self.tests]
        return reduce(_union, test_term)

    def transform(self, tr):
        """Applies transformation to this geometry.

        Parameters
        ----------
        tr : Transform
            Transformation to be applied.

        Returns
        -------
        geometry : AdditiveGeometry
            The result of this geometry transformation.
        """
        terms = [t.transform(tr) for t in self.terms]
        return AdditiveGeometry(*terms)

    def simplify(self, box=GLOBAL_BOX, split_disjoint=False,
                 min_volume=MIN_BOX_VOLUME):
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
        for i, t in enumerate(self.terms):
            t.order = i
        simple_geoms = self._simplify(box, split_disjoint, min_volume)
        # remove not mandatory terms
        simplest = []
        for sg in simple_geoms:
            terms = [t for t in sg.terms if t.mandatory]
            simplest.append(AdditiveGeometry(*terms))
        # TODO: implement choosing of the simplest geometry representation
        # TODO: implement splitting of disjoint cells.
        return simplest

    def _simplify(self, box, split_disjoint, min_volume):
        result, simple_geoms = self.test_box(box, return_simple=True)
        if result == -1 or box.volume() <= min_volume:
            return simple_geoms

        # This is the case result == 0 or 1.
        simple = []
        # comple = []  It is not used for now. It is intended to count
        # complexities of geometries for their selection.
        for geom in simple_geoms:
            c = geom.complexity()
            if c <= 1:
                simple.append(geom)
                # comple.append(c)
                continue
            # If geometry is too complex and box is too large -> we split box.
            box1, box2 = box.split()
            simp_geoms1 = geom._simplify(box1, split_disjoint=split_disjoint,
                                        min_volume=min_volume)
            simp_geoms2 = geom._simplify(box2, split_disjoint=split_disjoint,
                                        min_volume=min_volume)
            # Now merge results
            for adg1, adg2 in product(simp_geoms1, simp_geoms2):
                ag = adg1.merge_geometries(adg2)
                for a in simple:
                    if ag.equivalent(a):
                        break
                else:
                    c = ag.complexity()
                    if c > 0:
                        simple.append(ag)
                        # comple.append(c)
        # simple_sort = [simple[i] for i in np.argsort(comple)[:15]]
        return simple #_sort

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

