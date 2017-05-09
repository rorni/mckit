# -*- coding: utf-8 -*-


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
        pass

    def get_surfaces(self):
        """Gets a set of surfaces that bound this cell.

        Returns
        -------
        surfaces : set
            Surfaces that bound this cell.
        """
        # TODO: implement get_surfaces method
        raise NotImplementedError

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
        # TODO: implement test_point method.
        raise NotImplementedError

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
            +1 if the cell lies entirely inside the region.
             0 if the cell (probably) intersects the region.
            -1 if the cell lies outside the region.
        """
        # TODO: implement test_region method.
        raise NotImplementedError

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
        # TODO: implement transform method.
        raise NotImplementedError

