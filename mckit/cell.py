# -*- coding: utf-8 -*-

import numpy as np

from .surface import Surface
from .constants import EX, EY, EZ
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

    def bounding_box(self, ex=EX, ey=EY, ez=EZ, max_dim=1e+4, tol=1.0,
                     adjust=False):
        """Gets bounding box for this cell.
        
        Parameters
        ----------
        ex, ey, ez : array_like
            Basis vectors for initial bounding box.
        max_dim : float
            Initial dimension of bounding box.
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
        base = -0.5 * max_dim * (ex + ey + ez)
        box = Box(base, ex, ey, ez)
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

