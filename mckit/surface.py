# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import fmin_slsqp

from .constants import *
from .fmesh import Box


def create_surface(kind, *params, **options):
    """Creates new surface.

    Parameters
    ----------
    kind : str
        Surface kind designator. See MCNP manual.
    params : list[float]
        List of surface parameters.
    options : dict
        Dictionary of surface's options. Possible values:
            transform = tr - transformation to be applied to the surface being
                             created. Transformation instance.

    Returns
    -------
    surf : Surface
        New surface.
    """
    kind = kind.upper()
    if kind[-1] == 'X':
        axis = EX
    elif kind[-1] == 'Y':
        axis = EY
    elif kind[-1] == 'Z':
        axis = EZ
    # -------- Plane -------------------
    if kind[0] == 'P':
        if len(kind) == 2:
            return Plane(axis, -params[0], **options)
        else:
            return Plane(params[:3], -params[3], **options)
    # -------- SQ -------------------
    elif kind == 'SQ':
        A, B, C, D, E, F, G, x0, y0, z0 = params
        m = np.diag([A, B, C])
        v = 2 * np.array([D - A*x0, E - B*y0, F - C*z0])
        k = A*x0**2 + B*y0**2 + C*z0**2 - 2 * (D*x0 + E*y0 + F*z0) + G
        return GQuadratic(m, v, k, **options)
    # -------- Sphere ------------------
    elif kind[0] == 'S':
        if kind == 'S':
            r0 = np.array(params[:3])
        elif kind == 'SO':
            r0 = ORIGIN
        else:
            r0 = axis * params[0]
        R = params[-1]
        return Sphere(r0, R, **options)
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
        return Cylinder(r0, axis, R, **options)
    # -------- Cone ---------------
    elif kind[0] == 'K':
        if kind[1] == '/':
            r0 = np.array(params[:3])
        else:
            r0 = params[0] * axis
        ta = np.sqrt(params[-1])
        return Cone(r0, axis, ta, **options)
    # ---------- GQ -----------------
    elif kind == 'GQ':
        A, B, C, D, E, F, G, H, J, k = params
        m = np.array([[A, 0.5*D, 0.5*F], [0.5*D, B, 0.5*E], [0.5*F, 0.5*E, C]])
        v = np.array([G, H, J])
        return GQuadratic(m, v, k, **options)
    # ---------- Torus ---------------------
    elif kind[0] == 'T':
        x0, y0, z0, R, a, b = params
        return Torus([x0, y0, z0], axis, R, a, b, **options)
    # ---------- Axisymmetric surface defined by points ------
    else:
        if len(params) == 2:
            return Plane(axis, -params[0], **options)
        elif len(params) == 4:
            # TODO: Use special classes instead of GQ
            h1, r1, h2, r2 = params
            if abs(h2 - h1) < RESOLUTION * max(abs(h1), abs(h2)):
                return Plane(axis, -0.5 * (h1 + h2), **options)
            elif abs(r2 - r1) < RESOLUTION * max(abs(r2), abs(r1)):
                R = 0.5 * (abs(r1) + abs(r2))
                return GQuadratic(np.diag(1-axis), [0, 0, 0], -R**2, **options)
            else:
                h0 = (abs(r1) * h2 - abs(r2) * h1) / (abs(r1) - abs(r2))
                t2 = (abs(r1) - abs(r2))**2 / abs(h1 - h2)
                m = np.diag(1 - axis - t2 * axis)
                v = 2 * t2 * h0 * axis
                return GQuadratic(m, v, -t2 * h0**2, **options)
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
    test_box(box)
        Checks whether this surface crosses the box.
    projection(p)
        Gets projection of point p on the surface.
    """
    def __init__(self, **options):
        self._box_results = {}
        self._box_stack = []
        self.options = options

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

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
        return np.sign(self._func(p)).astype(int)

    def test_box(self, box):
        """Checks whether this surface crosses the region.

        Box defines a rectangular cuboid. This method checks if this surface
        crosses the box, i.e. there is two points belonging to this box which
        have different sense with respect to this surface.

        Parameters
        ----------
        box : Box
            Describes the box.

        Returns
        -------
        result : int
            Test result. It equals one of the following values:
            +1 if every point inside the box has positive sense.
             0 if there are both points with positive and negative sense inside
               the box
            -1 if every point inside the box has negative sense.
        """
        corners = box.corners()
        senses = self.test_point(corners)
        sign = np.sign(np.max(senses) + np.min(senses))
        if sign != 0:
            bounds = box.bounds()
            for start_pt in corners:
                end_pt = fmin_slsqp(
                    self._func, start_pt, fprime=self._grad,
                    f_ieqcons=box.f_ieqcons(),
                    fprime_ieqcons=box.fprime_ieqcons(),
                    args=(sign,), bounds=bounds, disp=0
                )
                if self.test_point(end_pt) * sign < 0:
                    sign = 0
        return sign

    @abstractmethod
    def projection(self, p):
        """Gets projection of point(s) p on the surface.
        
        Finds a projection of point (or points) p on this surface. In case 
        the surface is not Plane instance only the projection closest to point
        p is found.
        
        Parameters
        ----------
        p : array_like[float]
            Coordinates of point(s).

        Returns
        -------
        proj : array_like
            Projected points.
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
    def _func(self, x, sign=+1):
        """Calculates deviation of point x from this surface.

        If the result is positive, then point has positive sense. Else -
        negative. sign parameter change sense to the opposite.

        Parameters
        ----------
        x : array_like[float]
            Coordinates of individual point to be tested (if shape (3,)) or
            of a set of points (if shape (N_pts, 3)).
        sign : int
            Multiplier to change test result. It is used in optimization
            problem to find function minimum.

        Returns
        -------
        result : float or array_like[float]
            Test result for individual point or points.
        """

    @abstractmethod
    def _grad(self, x, sign=+1):
        """Calculates gradient of function _func.

        Parameters
        ----------
        x : array_like[float]
            Coordinates of individual point to be tested (if shape (3,)) or
            of a set of points (if shape (N_pts, 3)).
        sign : int
            Multiplier to change test result. It is used in optimization
            problem to find function minimum.

        Returns
        -------
        result : array_like[float]
            Gradient values of function in every given point.
        """


class Plane(Surface):
    """Plane surface class.

    Parameters
    ----------
    normal : array_like[float]
        The normal to the plane being created.
    offset : float
        Free term.
    options : dict
        Dictionary of surface's options. Possible values:
            transform = tr - transformation to be applied to this plane.
                             Transformation instance.
    """
    def __init__(self, normal, offset, **options):
        if 'transform' in options.keys():
            tr = options.pop('transform')
            v, k = tr.apply2plane(normal, offset)
        else:
            v = np.array(normal)
            k = offset
        Surface.__init__(self, **options)
        self._v = v
        self._k = k

    def transform(self, tr):
        return Plane(self._v, self._k, transform=tr)

    def test_box(self, box):
        # Test sense of all region vertices.
        senses = self.test_point(box.corners())
        # Returns 0 if both +1 and -1 values present.
        return np.sign(np.max(senses) + np.min(senses))

    def projection(self, p):
        shape = (p.shape[0], 1) if len(p.shape) == 2 else (1,)
        d = np.reshape(np.dot(p, self._v) + self._k, shape)
        return p - np.multiply(d, self._v)

    def _func(self, x, sign=+1):
        p = sign * (np.dot(x, self._v) + self._k)
        return p

    def _grad(self, x, sign=+1):
        return self._v


class Sphere(Surface):
    """Sphere surface class.
    
    Parameters
    ----------
    center : array_like[float]
        Center of the sphere.
    radius : float
        The radius of the sphere.
    options : dict
        Dictionary of surface's options. Possible values:
            transform = tr - transformation to be applied to the sphere being
                             created. Transformation instance.
    """
    def __init__(self, center, radius, **options):
        if 'transform' in options.keys():
            tr = options.pop('transform')
            center = tr.apply2point(center)
        Surface.__init__(self, **options)
        self._center = np.array(center)
        self._radius = radius

    def projection(self, p):
        n = p - self._center
        n /= np.linalg.norm(n)
        return self._center + self._radius * n

    def transform(self, tr):
        return Sphere(self._center, self._radius, transform=tr)

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
    options : dict
        Dictionary of surface's options. Possible values:
            transform = tr - transformation to be applied to the cylinder being
                             created. Transformation instance.
    """
    def __init__(self, pt, axis, radius, **options):
        if 'transform' in options.keys():
            tr = options.pop('transform')
            pt = tr.apply2point(pt)
            axis = tr.apply2vector(axis)
        Surface.__init__(self, **options)
        self._pt = np.array(pt)
        self._axis = np.array(axis) / np.linalg.norm(axis)
        self._radius = radius

    def projection(self, p):
        shape = (p.shape[0], 1) if len(p.shape) == 2 else (1,)
        d = np.reshape(np.dot(p - self._pt, self._axis), shape)
        b = self._pt + np.multiply(d, self._axis)
        a = p - b
        return b + self._radius * a / np.linalg.norm(a)

    def transform(self, tr):
        return Cylinder(self._pt, self._axis, self._radius, transform=tr)

    def _func(self, x, sign=+1):
        a = x - self._pt
        an = np.dot(a, self._axis)
        quad = np.sum(np.multiply(a, a), axis=-1) - np.multiply(an, an)
        return sign * (quad - self._radius**2)

    def _grad(self, x, sign=+1):
        a = x - self._pt
        return sign * 2 * (a - np.dot(a, self._axis) * self._axis)


class Cone(Surface):
    """Cone surface class.

    Parameters
    ----------
    apex : array_like[float]
        Cone's apex.
    axis : array_like[float]
        Cone's axis.
    ta : float
        Tangent of angle between axis and generatrix.
    options : dict
        Dictionary of surface's options. Possible values:
            transform = tr - transformation to be applied to the cone being
                             created. Transformation instance.
    """
    def __init__(self, apex, axis, ta, **options):
        if 'transform' in options.keys():
            tr = options.pop('transform')
            apex = tr.apply2point(apex)
            axis = tr.apply2vector(axis)
        Surface.__init__(self, **options)
        self._apex = np.array(apex)
        self._axis = np.array(axis) / np.linalg.norm(axis)
        self._t2 = ta**2

    def projection(self, p):
        raise NotImplementedError

    def transform(self, tr):
        return Cone(self._apex, self._axis, np.sqrt(self._t2), transform=tr)

    def _func(self, x, sign=+1):
        a = x - self._apex
        an = np.dot(a, self._axis)
        quad = np.sum(np.multiply(a, a), axis=-1)
        return sign * (quad - np.multiply(an, an) * (1 + self._t2))

    def _grad(self, x, sign=+1):
        a = x - self._apex
        t2plus1 = 1 + self._t2
        return sign * 2 * (a - np.dot(a, self._axis) * self._axis * t2plus1)


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
    options : dict
        Dictionary of surface's options. Possible values:
            transform = tr - transformation to be applied to the surface being
                             created. Transformation instance.
    """
    def __init__(self, m, v, k, **options):
        if 'transform' in options.keys():
            tr = options.pop('transform')
            m, v, k = tr.apply2gq(m, v, k)
        else:
            m = np.array(m)
            v = np.array(v)
            k = k
        Surface.__init__(self, **options)
        self._m = m
        self._v = v
        self._k = k

    def projection(self, p):
        raise NotImplementedError

    def transform(self, tr):
        return GQuadratic(self._m, self._v, self._k, transform=tr)

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
    options : dict
        Dictionary of surface's options. Possible values:
            transform = tr - transformation to be applied to the torus being
                             created. Transformation instance.
    """
    def __init__(self, center, axis, R, a, b, **options):
        if 'transform' in options.keys():
            tr = options.pop('transform')
            center = tr.apply2point(center)
            axis = tr.apply2vector(axis)
        else:
            center = np.array(center)
            axis = np.array(axis)
        Surface.__init__(self, **options)
        self._center = center
        self._axis = axis / np.linalg.norm(axis)
        self._R = R
        self._a = a
        self._b = b
        # case of degenerate torus
        self._spec_pts = []
        if b > R:
            offset = a * np.sqrt(1 - (R / b)**2)
            self._spec_pts.append(self._center + offset * self._axis)
            self._spec_pts.append(self._center - offset * self._axis)

    def projection(self, p):
        raise NotImplementedError

    def test_box(self, box):
        # TODO: implement test_box
        bounds = box.bounds()
        # If special point is inside region, then torus boundary definitely
        # in the region.
        for spec_pt in self._spec_pts:
            result = True
            for v, (lo, hi) in zip(spec_pt, bounds):
                result = result and (lo < v < hi)
            if result:
                return 0
        return super(Torus, self).test_box(box)

    def transform(self, tr):
        return Torus(self._center, self._axis, self._R, self._a, self._b,
                     transform=tr)

    def _func(self, x, sign=+1):
        p = x - self._center
        pn = np.dot(p, self._axis)
        pp = np.sum(np.multiply(p, p), axis=-1)
        sq = np.sqrt(np.maximum(pp - pn**2, 0))
        return sign * (pn / self._a)**2 + ((sq - self._R) / self._b)**2 - 1

    def _grad(self, x, sign=+1):
        p = x - self._center
        pn = np.dot(p, self._axis)
        pp = np.sum(np.multiply(p, p), axis=-1)
        sq = np.sqrt(np.maximum(pp - pn ** 2, 0))
        if sq < RESOLUTION:
            add_term = 0
        else:
            add_term = self._R / (sq * self._b**2) * (p - pn * self._axis)
        return sign * 2 * (pn / self._a**2 * self._axis + \
               (p - pn * self._axis) / self._b**2 - add_term)
