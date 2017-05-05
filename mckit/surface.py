# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import fmin_tnc


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
        sense : int or numpy.ndarray[int]
            If the point has positive sense, then +1 value is returned.
            If point lies on the surface 0 is returned.
            If point has negative sense -1 is returned.
            Individual point - single value, array of points - array of
            ints of shape (num_points,) is returned.
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
        Surface.__init__(self)
        if transform is not None:
            v, k = transform.apply2plane(normal, offset)
        else:
            v = np.array(normal)
            k = offset
        self._v = v
        self._k = k

    def test_point(self, p):
        return np.sign(np.dot(p, self._v) + self._k).astype(int)

    def transform(self, tr):
        return Plane(self._v, self._k, transform=tr)

    def test_region(self, region):
        # Test sense of all region vertices.
        senses = self.test_point(region)
        # Returns 0 if both +1 and -1 values present.
        return np.sign(np.max(senses) + np.min(senses))


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
    def __init__(self, m, v, k, transform=None):
        Surface.__init__(self)
        if transform is not None:
            m, v, k = transform.apply2gq(m, v, k)
        else:
            m = np.array(m)
            v = np.array(v)
            k = k
        self._m = m
        self._v = v
        self._k = k

    def test_point(self, p):
        return np.sign(self._func(p)).astype(int)

    def transform(self, tr):
        return GQuadratic(self._m, self._v, self._k, transform=tr)

    def test_region(self, region):
        senses = self.test_point(region)
        sign = np.sign(np.max(senses) + np.min(senses))
        if sign != 0:
            bounds = [[lo, hi] for lo, hi in zip(np.amin(region, axis=0),
                                                 np.amax(region, axis=0))]
            for start_pt in region:
                end_pt = fmin_tnc(self._func, start_pt, fprime=self._grad,
                                  args=(sign,), bounds=bounds)[0]
                if self.test_point(end_pt) * sign < 0:
                    return 0
        return sign

    def _func(self, x, sign=+1):
        # TODO: Check mclight project for possible performance improvement.
        quad = np.sum(np.multiply(np.dot(x, self._m), x), axis=-1)
        return sign * (quad + np.dot(x, self._v) + self._k)

    def _grad(self, x, sign=+1):
        return sign * (0.5 * np.dot(x, self._m) + self._v)


class Torus(Surface):
    """Tori surface class.

    Parameters
    ----------
    center : array_like[float]
        The center of torus.
    axis : array_like[float]
        The axis of torus.
    R : float
        Major radius.
    a : float
        Radius parallel to torus axis.
    b : float
        Radius perpendicular to torus axis.
    transform : Transformation
    """
    def __init__(self, center, axis, R, a, b, transform=None):
        # TODO: implement torus creation.
        pass

    def test_point(self, p):
        # TODO: implement test_point
        raise NotImplementedError

    def test_region(self, region):
        # TODO: implement test_region
        raise NotImplementedError

    def transform(self, tr):
        # TODO: implement transform method
        raise NotImplementedError
