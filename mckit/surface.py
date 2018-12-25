# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import numpy as np

from .constants import *
from .geometry import Plane      as _Plane,    \
                      Sphere     as _Sphere,   \
                      Cone       as _Cone,     \
                      Cylinder   as _Cylinder, \
                      Torus      as _Torus,    \
                      GQuadratic as _GQuadratic, \
                      GLOBAL_BOX, ORIGIN, EX, EY, EZ
from .printer import print_card, pretty_float
from .transformation import Transformation
from .utils import *
from .card import Card


__all__ = [
    'create_surface', 'Plane', 'Sphere', 'Cone', 'Torus', 'GQuadratic',
    'Cylinder'
]


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
            ta = np.sqrt(params[3])
        else:
            r0 = params[0] * axis
            ta = np.sqrt(params[1])
        sheet = 0 if len(params) % 2 == 0 else int(params[-1])
        return Cone(r0, axis, ta, sheet, **options)
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
                return Cylinder([0, 0, 0], axis, R, **options)
            else:
                if r1 * r2 < 0:
                    raise ValueError('Points must belong to the one sheet.')
                h0 = (abs(r1) * h2 - abs(r2) * h1) / (abs(r1) - abs(r2))
                ta = abs((r1 - r2) / (h1 - h2))
                s = round((h1 - h0) / abs(h1 - h0))
                return Cone(axis * h0, axis, ta, sheet=s, **options)
        elif len(params) == 6:
            # TODO: Implement creation of surface by 3 points.
            raise NotImplementedError


def create_replace_dictionary(surfaces, unique=None, box=GLOBAL_BOX, tol=1.e-10):
    """Creates surface replace dictionary for equal surfaces removing.

    Parameters
    ----------
    surfaces : set[Surface]
        A set of surfaces to be checked.
    unique: set[Surface]
        A set of surfaces that are assumed to be unique. If not None, than
        'surfaces' are checked for coincidence with one of them.
    box : Box
        A box, which is used for comparison.
    tol : float
        Tolerance

    Returns
    -------
    replace : dict
        A replace dictionary. surface -> (replace_surface, sense). Sense is +1
        if surfaces have the same direction of normals. -1 otherwise.
    """
    replace = {}
    uniq_surfs = set() if unique is None else unique
    for s in surfaces:
        for us in uniq_surfs:
            t = s.equals(us, box=box, tol=tol)
            if t != 0:
                replace[s] = (us, t)
                break
        else:
            uniq_surfs.add(s)
    return replace


class Surface(Card):
    """Base class for all surface classes.

    Methods
    -------
    equals(other, box, tol)
        Checks if this surface and surf are equal inside the box.
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
        Card.__init__(self, **options)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

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

    def mcnp_words(self):
        words = []
        mod = self.options.get('modifier', None)
        if mod:
            words.append(mod)
        words.append(str(self.name()))
        words.append('  ')
        return words


class Plane(Surface, _Plane):
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
        length = np.linalg.norm(v)
        v = v / length
        k /= length
        k = round_scalar(k, FLOAT_TOLERANCE, resolution=FLOAT_TOLERANCE)
        v = round_array(v, FLOAT_TOLERANCE, resolution=FLOAT_TOLERANCE)
        Surface.__init__(self, **options)
        _Plane.__init__(self, v, k)

    def __hash__(self):
        result = hash(self._k)
        for v in self._v:
            result ^= hash(v)
        return result

    def __eq__(self, other):
        if not isinstance(other, Plane):
            return False
        else:
            for x, y in zip(self._v, other._v):
                if x != y:
                    return False
            return self._k == other._k

    def reverse(self):
        """Gets the surface with reversed normal."""
        return Plane(-self._v, -self._k)

    def transform(self, tr):
        return Plane(self._v, self._k, transform=tr, **self.options)

    def mcnp_words(self):
        words = Surface.mcnp_words(self)
        if np.all(self._v == EX):
            words.append('PX')
        elif np.all(self._v == EY):
            words.append('PY')
        elif np.all(self._v == EZ):
            words.append('PZ')
        else:
            words.append('P')
            for v in self._v:
                p = significant_digits(v, FLOAT_TOLERANCE)
                words.append(' ')
                words.append(pretty_float(v, p))
        words.append(' ')
        p = significant_digits(self._k, FLOAT_TOLERANCE)
        words.append(pretty_float(-self._k, p))
        return print_card(words)


class Sphere(Surface, _Sphere):
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
        center = round_array(np.array(center), FLOAT_TOLERANCE,
                             resolution=FLOAT_TOLERANCE)
        radius = round_scalar(radius, FLOAT_TOLERANCE,
                              resolution=FLOAT_TOLERANCE)
        _Sphere.__init__(self, center, radius)

    def __hash__(self):
        result = hash(self._radius)
        for c in self._center:
            result ^= hash(c)
        return result

    def __eq__(self, other):
        if not isinstance(other, Sphere):
            return False
        else:
            for x, y in zip(self._center, other._center):
                if x != y:
                    return False
            return self._radius == other._radius

    def transform(self, tr):
        return Sphere(self._center, self._radius, transform=tr, **self.options)

    def mcnp_words(self):
        words = Surface.mcnp_words(self)
        if np.all(self._center == np.array([0.0, 0.0, 0.0])):
            words.append('SO')
        elif self._center[0] == 0.0 and self._center[1] == 0.0:
            words.append('SZ')
            words.append(' ')
            v = self._center[2]
            p = significant_digits(v, FLOAT_TOLERANCE)
            words.append(pretty_float(v, p))
        elif self._center[1] == 0.0 and self._center[2] == 0.0:
            words.append('SX')
            words.append(' ')
            v = self._center[0]
            p = significant_digits(v, FLOAT_TOLERANCE)
            words.append(pretty_float(v, p))
        elif self._center[0] == 0.0 and self._center[2] == 0.0:
            words.append('SY')
            words.append(' ')
            v = self._center[1]
            p = significant_digits(v, FLOAT_TOLERANCE)
            words.append(pretty_float(v, p))
        else:
            words.append('S')
            for v in self._center:
                words.append(' ')
                p = significant_digits(v, FLOAT_TOLERANCE)
                words.append(pretty_float(v, p))
        words.append(' ')
        v = self._radius
        p = significant_digits(v, FLOAT_TOLERANCE)
        words.append(pretty_float(v, p))
        return print_card(words)


class Cylinder(Surface, _Cylinder):
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
        axis = np.array(axis) / np.linalg.norm(axis)
        maxdir = np.argmax(np.abs(axis))
        if axis[maxdir] < 0:
            axis *= -1
        pt = np.array(pt)
        axis = round_array(axis, FLOAT_TOLERANCE, resolution=FLOAT_TOLERANCE)
        pt = pt - axis * np.dot(pt, axis)
        pt = round_array(pt, FLOAT_TOLERANCE, resolution=FLOAT_TOLERANCE)
        radius = round_scalar(radius, FLOAT_TOLERANCE, resolution=FLOAT_TOLERANCE)
        Surface.__init__(self, **options)
        _Cylinder.__init__(self, pt, axis, radius)

    def __hash__(self):
        result = hash(self._radius)
        for c in self._pt:
            result ^= hash(c)
        for a in self._axis:
            result ^= hash(a)
        return result

    def __eq__(self, other):
        if not isinstance(other, Cylinder):
            return False
        else:
            for x, y in zip(self._pt, other._pt):
                if x != y:
                    return False
            for x, y in zip(self._axis, other._axis):
                if x != y:
                    return False
            return self._radius == other._radius

    def transform(self, tr):
        return Cylinder(self._pt, self._axis, self._radius, transform=tr,
                        **self.options)

    def mcnp_words(self):
        words = Surface.mcnp_words(self)
        if np.all(self._axis == np.array([1.0, 0.0, 0.0])):
            if self._pt[1] == 0.0 and self._pt[2] == 0.0:
                words.append('CX')
            else:
                words.append('C/X')
                words.append(' ')
                v = self._pt[1]
                p = significant_digits(v, FLOAT_TOLERANCE)
                words.append(pretty_float(v, p))
                words.append(' ')
                v = self._pt[2]
                p = significant_digits(v, FLOAT_TOLERANCE)
                words.append(pretty_float(v, p))
        elif np.all(self._axis == np.array([0.0, 1.0, 0.0])):
            if self._pt[0] == 0.0 and self._pt[2] == 0.0:
                words.append('CY')
            else:
                words.append('C/Y')
                words.append(' ')
                v = self._pt[0]
                p = significant_digits(v, FLOAT_TOLERANCE)
                words.append(pretty_float(v, p))
                words.append(' ')
                v = self._pt[2]
                p = significant_digits(v, FLOAT_TOLERANCE)
                words.append(pretty_float(v, p))
        elif np.all(self._axis == np.array([0.0, 0.0, 1.0])):
            if self._pt[0] == 0.0 and self._pt[1] == 0.0:
                words.append('CZ')
            else:
                words.append('C/Z')
                words.append(' ')
                v = self._pt[0]
                p = significant_digits(v, FLOAT_TOLERANCE)
                words.append(pretty_float(v, p))
                words.append(' ')
                v = self._pt[1]
                p = significant_digits(v, FLOAT_TOLERANCE)
                words.append(pretty_float(v, p))
        else:
            nx, ny, nz = self._axis
            m = np.array([[1-nx**2, -nx*ny, -nx*nz],
                          [-nx*ny, 1-ny**2, -ny*nz],
                          [-nx*nz, -ny*nz, 1-nz**2]])
            v = np.zeros(3)
            k = -self._radius**2
            m, v, k = Transformation(translation=self._pt).apply2gq(m, v, k)
            return str(GQuadratic(m, v, k, **self.options))
        words.append(' ')
        v = self._radius
        p = significant_digits(v, FLOAT_TOLERANCE)
        words.append(pretty_float(v, p))
        return print_card(words)


class Cone(Surface, _Cone):
    """Cone surface class.

    Parameters
    ----------
    apex : array_like[float]
        Cone's apex.
    axis : array_like[float]
        Cone's axis.
    ta : float
        Tangent of angle between axis and generatrix.
    sheet : int
        Cone's sheet.
    options : dict
        Dictionary of surface's options. Possible values:
            transform = tr - transformation to be applied to the cone being
                             created. Transformation instance.
    """
    def __init__(self, apex, axis, ta, sheet=0, **options):
        if 'transform' in options.keys():
            tr = options.pop('transform')
            apex = tr.apply2point(apex)
            axis = tr.apply2vector(axis)
        axis = np.array(axis) / np.linalg.norm(axis)
        maxdir = np.argmax(np.abs(axis))
        if axis[maxdir] < 0:
            axis *= -1
            sheet *= -1
        apex = np.array(apex)
        axis = round_array(axis, FLOAT_TOLERANCE, resolution=FLOAT_TOLERANCE)
        apex = round_array(apex, FLOAT_TOLERANCE, resolution=FLOAT_TOLERANCE)
        ta = round_scalar(ta, np.sqrt(FLOAT_TOLERANCE))

        Surface.__init__(self, **options)
        # TODO: Do something with ta! It is confusing. _Cone accept ta, but returns t2.
        _Cone.__init__(self, apex, axis, ta, sheet)

    def __hash__(self):
        result = hash(self._t2) ^ hash(self._sheet)
        for c in self._apex:
            result ^= hash(c)
        for a in self._axis:
            result ^= hash(a)
        return result

    def __eq__(self, other):
        if not isinstance(other, Cone):
            return False
        else:
            for x, y in zip(self._apex, other._apex):
                print('{0:.15e}  {1:.15e}'.format(x, y))
                if x != y:
                    return False
            for x, y in zip(self._axis, other._axis):
                print('{0:.15e}  {1:.15e}'.format(x, y))
                if x != y:
                    return False
            print('{0:.15e}  {1:.15e}'.format(self._t2, other._t2))
            print('{0:.15e}  {1:.15e}'.format(self._sheet, other._sheet))
            return self._t2 == other._t2 and self._sheet == other._sheet

    def transform(self, tr):
        return Cone(self._apex, self._axis, np.sqrt(self._t2),
                    sheet=self._sheet, transform=tr, **self.options)

    def mcnp_words(self):
        words = Surface.mcnp_words(self)
        if np.all(self._axis == np.array([1.0, 0.0, 0.0])):
            if self._apex[1] == 0.0 and self._apex[2] == 0.0:
                words.append('KX')
                words.append(' ')
                v = self._apex[0]
                p = significant_digits(v, FLOAT_TOLERANCE)
                words.append(pretty_float(v, p))
            else:
                words.append('K/X')
                for v in self._apex:
                    words.append(' ')
                    p = significant_digits(v, FLOAT_TOLERANCE)
                    words.append(pretty_float(v, p))
        elif np.all(self._axis == np.array([0.0, 1.0, 0.0])):
            if self._apex[0] == 0.0 and self._apex[2] == 0.0:
                words.append('KY')
                words.append(' ')
                v = self._apex[1]
                p = significant_digits(v, FLOAT_TOLERANCE)
                words.append(pretty_float(v, p))
            else:
                words.append('K/Y')
                for v in self._apex:
                    words.append(' ')
                    p = significant_digits(v, FLOAT_TOLERANCE)
                    words.append(pretty_float(v, p))
        elif np.all(self._axis == np.array([0.0, 0.0, 1.0])):
            if self._apex[0] == 0.0 and self._apex[1] == 0.0:
                words.append('KZ')
                words.append(' ')
                v = self._apex[2]
                p = significant_digits(v, FLOAT_TOLERANCE)
                words.append(pretty_float(v, p))
            else:
                words.append('K/Z')
                for v in self._apex:
                    words.append(' ')
                    p = significant_digits(v, FLOAT_TOLERANCE)
                    words.append(pretty_float(v, p))
        else:
            nx, ny, nz = self._axis
            a = 1 + self._t2
            m = np.array([[1-a*nx**2, -a*nx*ny, -a*nx*nz],
                          [-a*nx*ny, 1-a*ny**2, -a*ny*nz],
                          [-a*nx*nz, -a*ny*nz, 1-a*nz**2]])
            v = np.zeros(3)
            k = 0
            m, v, k = Transformation(translation=self._apex).apply2gq(m, v, k)
            return str(GQuadratic(m, v, k, **self.options))
        words.append(' ')
        v = self._t2
        p = significant_digits(v, np.sqrt(FLOAT_TOLERANCE))
        words.append(pretty_float(v, p))
        if self._sheet != 0:
            words.append(' ')
            words.append('{0:d}'.format(self._sheet))
        return words


class GQuadratic(Surface, _GQuadratic):
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
        maxdir = np.argmax(np.abs(np.diag(m)))
        if np.diag(m)[maxdir] < 0:
            m *= -1
            v *= -1
            k *= -1
        m = round_array(m, FLOAT_TOLERANCE, resolution=FLOAT_TOLERANCE)
        v = round_array(v, FLOAT_TOLERANCE, resolution=FLOAT_TOLERANCE)
        k = round_scalar(k, FLOAT_TOLERANCE)

        Surface.__init__(self, **options)
        _GQuadratic.__init__(self, m, v, k)

    def __hash__(self):
        result = hash(self._k)
        for v in self._v:
            result ^= hash(v)
        for x in self._m.ravel():
            result ^= hash(x)
        return result

    def __eq__(self, other):
        if not isinstance(other, GQuadratic):
            return False
        else:
            for x, y in zip(self._v, other._v):
                if x != y:
                    return False
            for x, y in zip(self._m.ravel(), other._m.ravel()):
                if x != y:
                    return False
            return self._k == other._k

    def transform(self, tr):
        return GQuadratic(self._m, self._v, self._k, transform=tr,
                          **self.options)

    def mcnp_words(self):
        words = Surface.mcnp_words(self)
        a, b, c = np.diag(self._m)
        d = self._m[0, 1] + self._m[1, 0]
        e = self._m[1, 2] + self._m[2, 1]
        f = self._m[0, 2] + self._m[2, 0]
        g, h, j = self._v
        k = self._k
        for v in [a, b, c, d, e, f, g, h, j, k]:
            words.append(' ')
            p = significant_digits(v, FLOAT_TOLERANCE)
            words.append(pretty_float(v, p))
        return print_card(words)


class Torus(Surface, _Torus):
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
        axis = axis / np.linalg.norm(axis)
        maxdir = np.argmax(np.abs(axis))
        if axis[maxdir] < 0:
            axis *= -1
        axis = round_array(axis, FLOAT_TOLERANCE, resolution=FLOAT_TOLERANCE)
        center = round_array(center, FLOAT_TOLERANCE, resolution=FLOAT_TOLERANCE)
        R = round_scalar(R, FLOAT_TOLERANCE)
        a = round_scalar(a, FLOAT_TOLERANCE)
        b = round_scalar(b, FLOAT_TOLERANCE)

        Surface.__init__(self, **options)
        _Torus.__init__(self, center, axis, R, a, b)

    def __hash__(self):
        result = hash(self._R) ^ hash(self._a) ^ hash(self._b)
        for c in self._center:
            result ^= hash(c)
        for a in self._axis:
            result ^= hash(a)
        return result

    def __eq__(self, other):
        if not isinstance(other, Torus):
            return False
        else:
            for x, y in zip(self._center, other._center):
                if x != y:
                    return False
            for x, y in zip(self._axis, other._axis):
                if x != y:
                    return False
            return self._R == other._R and self._a == other._a and self._b == other._b

    def transform(self, tr):
        return Torus(self._center, self._axis, self._R, self._a, self._b,
                     transform=tr, **self.options)

    def mcnp_words(self):
        words = Surface.mcnp_words(self)
        if np.all(self._axis == np.array([1.0, 0.0, 0.0])):
            words.append('TX')
        elif np.all(self._axis == np.array([0.0, 1.0, 0.0])):
            words.append('TY')
        elif np.all(self._axis == np.array([0.0, 0.0, 1.0])):
            words.append('TZ')
        x, y, z = self._center
        for v in [x, y, z, self._R, self._a, self._b]:
            words.append(' ')
            p = significant_digits(v, FLOAT_TOLERANCE)
            words.append(pretty_float(v, p))
        return print_card(words)
