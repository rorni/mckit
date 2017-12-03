# -*- coding: utf-8 -*-

import numpy as np

from .surface import Surface


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
             0 if the box (probably) intersects the region.
            -1 if the box lies outside the region.
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

