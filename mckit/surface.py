# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import numpy as np


def create_surface(kind, *params, transform=None):
    """Creates new surface.

    Parameters
    ----------
    kind : str
        Surface kind designator. See MCNP manual.
    params : list[float]
        List of surface parameters.
    transform : Transformation
        Transformation to be applied to the surface being created.

    Returns
    -------
    surf : Surface
        New surface.
    """
    # TODO: implement creation of surface.
    raise NotImplementedError


class Surface(ABC):
    """Base class for all surface classes.

    Methods
    -------
    test_point(p)
        Checks the sense of point p with respect to this surface.
    transform(tr)
        Applies transformation tr to this surface.
    test_region(region)
        Checks whether this surface crosses the region.
    """
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    @abstractmethod
    def test_point(self, p):
        """Checks the sense of point(s) p.

        Parameters
        ----------
        p : array_like[float]
            Coordinates of point(s) to be checked. If it is the only one point,
            then p.shape=(3,). If it is an array of points, then
            p.shape=(num_points, 3).

        Returns
        -------
        sense : bool or numpy.ndarray[bool]
            If the point has positive sense, then True value is returned.
            Individual point - single bool value, array of points - array of
            bool of shape (num_points,) is returned.
        """

    @abstractmethod
    def transform(self, tr):
        """Applies transformation to this surface.

        Parameters
        ----------
        tr : Transform
            Transformation to be applied.

        Returns
        -------
        surf : Surface
            The result of this surface transformation.
        """

    @abstractmethod
    def test_region(self, region):
        """Checks whether this surface crosses the region.

        Region defines a rectangular cuboid. This method checks if this surface
        crosses the box, i.e. there is two points belonging to this region which
        have different sense with respect to this surface.

        Parameters
        ----------
        region : array_like[float]
            Describes the region. Region is a cuboid with sides perpendicular to
            the coordinate axis. It has shape 8x3 - defines 8 points.

        Returns
        -------
        result : int
            Test result. It equals one of the following values:
            +1 if every point inside region has positive sense.
             0 if there are both points with positive and negative sense inside
               region
            -1 if every point inside region has negative sense.
        """


class Plane(Surface):
    """Plane surface class.

    Parameters
    ----------
    normal : array_like[float]
        The normal to the plane being created.
    offset : float
        Free term.
    transform : Transformation
        Transformation to be applied to this plane.
    """
    def __init__(self, normal, offset, transform=None):
        # TODO: implement Plane instance creation.
        pass

    def test_point(self, p):
        # TODO: implement test_point method
        raise NotImplementedError

    def transform(self, tr):
        # TODO: implement transform method
        raise NotImplementedError

    def test_region(self, region):
        # TODO: implement test_region method
        raise NotImplementedError


class GQuadratic(Surface):
    """Generic quadratic surface class.

    Parameters
    ----------
    m : array_like[float]
        Matrix of coefficients of quadratic terms. m.shape=(3,3)
    v : array_like[float]
        Vector of coefficients of linear terms. v.shape=(3,)
    k : float
        Free term.
    transform : Transformation
        Transformation to be applied to this surface.
    """
    def __init(self, m, v, k, transform=None):
        # TODO: implement GQuadratic instance creation.
        pass

    def test_point(self, p):
        # TODO: implement test_point method
        raise NotImplementedError

    def transform(self, tr):
        # TODO: implement transform method
        raise NotImplementedError

    def test_region(self, region):
        # TODO: implement test_region method
        raise NotImplementedError
