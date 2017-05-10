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
    test_point(p)
        Tests whether point(s) p belong to this cell (lies inside it).
    test_region(region)
        Checks whether this cell intersects with region.
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
        stack = []
        for op in self._expression:
            if isinstance(op, Surface):
                stack.append(op.test_point(p))
            elif op == 'C':
                stack.append(_complement(stack.pop()))
            elif op == 'I':
                stack.append(_intersection(stack.pop(), stack.pop()))
            elif op == 'U':
                stack.append(_union(stack.pop(), stack.pop()))
        return stack.pop()

    def test_region(self, region):
        """Checks whether this cell intersects with region.

        Parameters
        ----------
        region : array_like[float]
            Describes the region. Region is a cuboid with sides perpendicular to
            the coordinate axis. It has shape 8x3 - defines 8 points.

        Returns
        -------
        result : int
            Test result. It equals one of the following values:
            +1 if the region lies entirely inside the cell.
             0 if the cell (probably) intersects the region.
            -1 if the cell lies outside the region.
        """
        stack = []
        for op in self._expression:
            if isinstance(op, Surface):
                stack.append(op.test_region(region))
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

