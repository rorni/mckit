# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import fmin_tnc

from .constants import *


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
    if kind[-1] == 'X':
        axis = EX
    elif kind[-1] == 'Y':
        axis = EY
    elif kind[-1] == 'Z':
        axis = EZ
    # -------- Plane -------------------
    if kind[0] == 'P':
        if len(kind) == 2:
            return Plane(axis, -params[0], transform=transform)
        else:
            return Plane(params[:3], -params[3], transform=transform)
    # -------- SQ -------------------
    elif kind == 'SQ':
        A, B, C, D, E, F, G, x0, y0, z0 = params
        m = np.diag([A, B, C])
        v = 2 * np.array([D - A*x0, E - B*y0, F - C*z0])
        k = A*x0**2 + B*y0**2 + C*z0**2 - 2 * (D*x0 + E*y0 + F*z0) + G
        return GQuadratic(m, v, k, transform)
    # -------- Sphere ------------------
    elif kind[0] == 'S':
        if kind == 'S':
            r0 = np.array(params[:3])
        elif kind == 'SO':
            r0 = ORIGIN
        else:
            r0 = axis * params[0]
        R = params[-1]
        #return GQuadratic(np.eye(3), -2 * r0, np.sum(r0**2) - R**2, transform)
        return Sphere(r0, R, transform)
    # -------- Cylinder ----------------
    elif kind[0] == 'C':
        A = 1 - axis
        if kind[1] == '/':
            Ax, Az = np.dot(A, EX), np.dot(A, EZ)
            r0 = params[0] * (Ax * EX + (1 - Ax) * EY) + \
                 params[1] * ((1 - Az) * EY + Az * EZ)
        else:
            r0 = ORIGIN
        R = params[-1]
        return GQuadratic(np.diag(A), -2 * r0, np.sum(r0**2) - R**2, transform)
    # -------- Cone ---------------
    elif kind[0] == 'K':
        if kind[1] == '/':
            r0 = np.array(params[:3])
        else:
            r0 = params[0] * axis
        t2 = params[-1]
        m = np.diag(1 - axis - t2 * axis)
        v05 = r0 * (1 - axis - axis * t2)
        return GQuadratic(m, -2 * v05, np.dot(v05, r0), transform)
    # ---------- GQ -----------------
    elif kind == 'GQ':
        A, B, C, D, E, F, G, H, J, k = params
        m = np.array([[A, 0.5*D, 0.5*F], [0.5*D, B, 0.5*E], [0.5*F, 0.5*E, C]])
        v = np.array([G, H, J])
        return GQuadratic(m, v, k, transform)
    # ---------- Torus ---------------------
    elif kind[0] == 'T':
        x0, y0, z0, R, a, b = params
        return Torus([x0, y0, z0], axis, R, a, b, transform=transform)
    # ---------- Axisymmetric surface defined by points ------
    else:
        if len(params) == 2:
            return Plane(axis, -params[0], transform=transform)
        elif len(params) == 4:
            h1, r1, h2, r2 = params
            if abs(h2 - h1) < RESOLUTION * max(abs(h1), abs(h2)):
                return Plane(axis, -0.5 * (h1 + h2), transform=transform)
            elif abs(r2 - r1) < RESOLUTION * max(abs(r2), abs(r1)):
                R = 0.5 * (abs(r1) + abs(r2))
                return GQuadratic(np.diag(1-axis), [0, 0, 0], -R**2, transform)
            else:
                h0 = (abs(r1) * h2 - abs(r2) * h1) / (abs(r1) - abs(r2))
                t2 = (abs(r1) - abs(r2))**2 / abs(h1 - h2)
                m = np.diag(1 - axis - t2 * axis)
                v = 2 * t2 * h0 * axis
                return GQuadratic(m, v, -t2 * h0**2, transform)
        elif len(params) == 6:
            # TODO: Implement creation of surface by 3 points.
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


class Sphere(Surface):
    """Sphere surface class.
    
    Parameters
    ----------
    center : array_like[float]
        Center of the sphere.
    radius : float
        The radius of the sphere.
    transform : Transformation
        Transformation to be applied to this sphere.
    """
    def __init__(self, center, radius, transform=None):
        Surface.__init__(self)
        if transform is not None:
            center = transform.apply2point(center)
        self._center = np.array(center)
        self._radius = radius

    def test_point(self, p):
        return np.sign(self._func(p)).astype(int)

    def transform(self, tr):
        return Sphere(self._center, self._radius, transform=tr)

    def test_region(self, region):
        return GQuadratic.test_region(self, region)

    def _func(self, x, sign=+1):
        dist = x - self._center
        quad = np.sum(np.multiply(dist, dist), axis=-1)
        return sign * (quad - self._radius**2)

    def _grad(self, x, sign=+1):
        return sign * 2 * (x - self._center)


class Cylinder(Surface):
    """Cylinder surface class.
    
    Parameters
    ----------
    pt : array_like[float]
        Point on the cylinder's axis.
    axis : array_like[float]
        Cylinder's axis direction.
    radius : float
        Cylinder's radius.
    transform : Transformation
        Transformation to be applied to the cylinder being created.
    """
    def __init__(self, pt, axis, radius, transform=None):
        Surface.__init__(self)
        if transform is not None:
            pt = transform.apply2point(pt)
            axis = transform.apply2vector(axis)
        self._pt = np.array(pt)
        self._axis = np.array(axis) / np.linalg.norm(axis)
        self._radius = radius

    def test_point(self, p):
        return np.sign(self._func(p)).astype(int)

    def transform(self, tr):
        return Cylinder(self._pt, self._axis, self._radius, transform=tr)

    def test_region(self, region):
        return GQuadratic.test_region(self, region)

    def _func(self, x, sign=+1):
        pass

    def _grad(self, x, sign=+1):
        pass


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
                                  args=(sign,), bounds=bounds, disp=0)[0]
                if self.test_point(end_pt) * sign < 0:
                    return 0
        return sign

    def _func(self, x, sign=+1):
        # TODO: Check mclight project for possible performance improvement.
        quad = np.sum(np.multiply(np.dot(x, self._m), x), axis=-1)
        return sign * (quad + np.dot(x, self._v) + self._k)

    def _grad(self, x, sign=+1):
        return sign * (2 * np.dot(x, self._m) + self._v)


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
        Surface.__init__(self)
        if transform is not None:
            center = transform.apply2point(center)
            axis = transform.apply2vector(axis)
        else:
            center = np.array(center)
            axis = np.array(axis)
        self._center = center
        self._axis = axis
        self._R = R
        self._a = a
        self._b = b

    def test_point(self, p):
        d = np.dot(p, self._axis) - np.dot(self._center, self._axis)
        c = np.sqrt(np.sum(p**2, axis=-1) - 2 * np.dot(p, self._center) + \
            np.dot(self._center, self._center) - d**2)
        sense = (d / self._a)**2 + ((c - self._R) / self._b)**2 - 1
        return np.sign(sense).astype(int)

    def test_region(self, region):
        # TODO: implement test_region
        raise NotImplementedError

    def transform(self, tr):
        return Torus(self._center, self._axis, self._R, self._a, self._b, tr)
