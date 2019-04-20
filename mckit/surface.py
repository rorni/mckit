# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import numpy as np

from . import constants 
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
            if abs(h2 - h1) < constants.RESOLUTION * max(abs(h1), abs(h2)):
                return Plane(axis, -0.5 * (h1 + h2), **options)
            elif abs(r2 - r1) < constants.RESOLUTION * max(abs(r2), abs(r1)):
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
        words.append(' ')
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
        self._k_digits = significant_digits(k, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)
        self._v_digits = significant_array(v, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)
        Surface.__init__(self, **options)
        _Plane.__init__(self, v, k)

    def copy(self):
        instance = Plane.__new__(Plane, self._v, self._k)
        instance._k_digits = self._k_digits
        instance._v_digits = self._v_digits
        Surface.__init__(instance, **self.options)
        _Plane.__init__(instance, self._v, self._k)
        return instance

    def __hash__(self):
        result = hash(self._get_k())
        for v in self._get_v():
            result ^= hash(v)
        return result

    def __eq__(self, other):
        if not isinstance(other, Plane):
            return False
        else:
            for x, y in zip(self._get_v(), other._get_v()):
                if x != y:
                    return False
            return self._get_k() == other._get_k()

    def _get_k(self):
        return round_scalar(self._k, self._k_digits)

    def _get_v(self):
        return round_array(self._v, self._v_digits)

    def reverse(self):
        """Gets the surface with reversed normal."""
        return Plane(-self._v, -self._k)

    def transform(self, tr):
        return Plane(self._v, self._k, transform=tr, **self.options)

    def mcnp_words(self):
        words = Surface.mcnp_words(self)
        if np.all(self._get_v() == EX):
            words.append('PX')
        elif np.all(self._get_v() == EY):
            words.append('PY')
        elif np.all(self._get_v() == EZ):
            words.append('PZ')
        else:
            words.append('P')
            for v, p in zip(self._get_v(), self._v_digits):
                words.append(' ')
                words.append(pretty_float(v, p))
        words.append(' ')
        words.append(pretty_float(-self._get_k(), self._k_digits))
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
        self._center_digits = significant_array(np.array(center), constants.FLOAT_TOLERANCE,
                                                resolution=constants.FLOAT_TOLERANCE)
        self._radius_digits = significant_digits(radius, constants.FLOAT_TOLERANCE,
                                                 resolution=constants.FLOAT_TOLERANCE)
        _Sphere.__init__(self, center, radius)

    def copy(self):
        instance = Sphere.__new__(Sphere, self._center, self._radius)
        instance._center_digits = self._center_digits
        instance._radius_digits = self._radius_digits
        Surface.__init__(instance, **self.options)
        _Sphere.__init__(instance, self._center, self._radius)
        return instance

    def __hash__(self):
        result = hash(self._get_radius())
        for c in self._get_center():
            result ^= hash(c)
        return result

    def __eq__(self, other):
        if not isinstance(other, Sphere):
            return False
        else:
            for x, y in zip(self._get_center(), other._get_center()):
                if x != y:
                    return False
            return self._get_radius() == other._get_radius()

    def _get_center(self):
        return round_array(self._center, self._center_digits)

    def _get_radius(self):
        return round_scalar(self._radius, self._radius_digits)

    def transform(self, tr):
        return Sphere(self._center, self._radius, transform=tr, **self.options)

    def mcnp_words(self):
        words = Surface.mcnp_words(self)
        if np.all(self._get_center() == np.array([0.0, 0.0, 0.0])):
            words.append('SO')
        elif self._get_center()[0] == 0.0 and self._get_center()[1] == 0.0:
            words.append('SZ')
            words.append(' ')
            v = self._get_center()[2]
            p = self._center_digits[2]
            words.append(pretty_float(v, p))
        elif self._get_center()[1] == 0.0 and self._get_center()[2] == 0.0:
            words.append('SX')
            words.append(' ')
            v = self._get_center()[0]
            p = self._center_digits[0]
            words.append(pretty_float(v, p))
        elif self._get_center()[0] == 0.0 and self._get_center()[2] == 0.0:
            words.append('SY')
            words.append(' ')
            v = self._get_center()[1]
            p = self._center_digits[1]
            words.append(pretty_float(v, p))
        else:
            words.append('S')
            for v, p in zip(self._center, self._center_digits):
                words.append(' ')
                words.append(pretty_float(v, p))
        words.append(' ')
        v = self._get_radius()
        p = self._radius_digits
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
        self._axis_digits = significant_array(axis, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)
        pt = pt - axis * np.dot(pt, axis)
        self._pt_digits = significant_array(pt, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)
        self._radius_digits = significant_digits(radius, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)
        Surface.__init__(self, **options)
        _Cylinder.__init__(self, pt, axis, radius)

    def copy(self):
        instance = Cylinder.__new__(Cylinder, self._pt, self._axis, self._radius)
        instance._axis_digits = self._axis_digits
        instance._pt_digits = self._pt_digits
        instance._radius_digits = self._radius_digits
        Surface.__init__(instance, **self.options)
        _Cylinder.__init__(instance, self._pt, self._axis, self._radius)
        return instance

    def __hash__(self):
        result = hash(self._get_radius())
        for c in self._get_pt():
            result ^= hash(c)
        for a in self._get_axis():
            result ^= hash(a)
        return result

    def __eq__(self, other):
        if not isinstance(other, Cylinder):
            return False
        else:
            for x, y in zip(self._get_pt(), other._get_pt()):
                if x != y:
                    return False
            for x, y in zip(self._get_axis(), other._get_axis()):
                if x != y:
                    return False
            return self._get_radius() == other._get_radius()

    def _get_pt(self):
        return round_array(self._pt, self._pt_digits)

    def _get_axis(self):
        return round_array(self._axis, self._axis_digits)

    def _get_radius(self):
        return round_scalar(self._radius, self._radius_digits)

    def transform(self, tr):
        return Cylinder(self._pt, self._axis, self._radius, transform=tr,
                        **self.options)

    def mcnp_words(self):
        words = Surface.mcnp_words(self)
        if np.all(self._get_axis() == np.array([1.0, 0.0, 0.0])):
            if self._get_pt()[1] == 0.0 and self._get_pt()[2] == 0.0:
                words.append('CX')
            else:
                words.append('C/X')
                words.append(' ')
                v = self._get_pt()[1]
                p = self._pt_digits[1]
                words.append(pretty_float(v, p))
                words.append(' ')
                v = self._get_pt()[2]
                p = self._pt_digits[2]
                words.append(pretty_float(v, p))
        elif np.all(self._get_axis() == np.array([0.0, 1.0, 0.0])):
            if self._get_pt()[0] == 0.0 and self._get_pt()[2] == 0.0:
                words.append('CY')
            else:
                words.append('C/Y')
                words.append(' ')
                v = self._get_pt()[0]
                p = self._pt_digits[0]
                words.append(pretty_float(v, p))
                words.append(' ')
                v = self._get_pt()[2]
                p = self._pt_digits[2]
                words.append(pretty_float(v, p))
        elif np.all(self._get_axis() == np.array([0.0, 0.0, 1.0])):
            if self._get_pt()[0] == 0.0 and self._get_pt()[1] == 0.0:
                words.append('CZ')
            else:
                words.append('C/Z')
                words.append(' ')
                v = self._get_pt()[0]
                p = self._pt_digits[0]
                words.append(pretty_float(v, p))
                words.append(' ')
                v = self._get_pt()[1]
                p = self._pt_digits[1]
                words.append(pretty_float(v, p))
        else:
            nx, ny, nz = self._axis
            m = np.array([[1-nx**2, -nx*ny, -nx*nz],
                          [-nx*ny, 1-ny**2, -ny*nz],
                          [-nx*nz, -ny*nz, 1-nz**2]])
            v = np.zeros(3)
            k = -self._radius**2
            m, v, k = Transformation(translation=self._pt).apply2gq(m, v, k)
            return GQuadratic(m, v, k, **self.options).mcnp_repr()
        words.append(' ')
        v = self._get_radius()
        p = self._radius_digits
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
        self._axis_digits = significant_array(axis, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)
        self._apex_digits = significant_array(apex, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)

        Surface.__init__(self, **options)
        # TODO: Do something with ta! It is confusing. _Cone accept ta, but returns t2.
        _Cone.__init__(self, apex, axis, ta, sheet)
        self._t2_digits = significant_digits(self._t2, constants.FLOAT_TOLERANCE)

    def copy(self):
        ta = np.sqrt(self._t2)
        instance = Cone.__new__(Cone, self._apex, self._axis, ta, self._sheet)
        instance._axis_digits = self._axis_digits
        instance._apex_digits = self._apex_digits
        instance._t2_digits = self._t2_digits
        Surface.__init__(instance, **self.options)
        _Cone.__init__(instance, self._apex, self._axis, ta, self._sheet)
        return instance

    def __hash__(self):
        result = hash(self._get_t2()) ^ hash(self._sheet)
        for c in self._get_apex():
            result ^= hash(c)
        for a in self._get_axis():
            result ^= hash(a)
        return result

    def __eq__(self, other):
        if not isinstance(other, Cone):
            return False
        else:
            for x, y in zip(self._get_apex(), other._get_apex()):
                if x != y:
                    return False
            for x, y in zip(self._get_axis(), other._get_axis()):
                if x != y:
                    return False
            return self._get_t2() == other._get_t2() and self._sheet == other._sheet

    def _get_axis(self):
        return round_array(self._axis, self._axis_digits)

    def _get_apex(self):
        return round_array(self._apex, self._apex_digits)

    def _get_t2(self):
        return round_scalar(self._t2, self._t2_digits)

    def transform(self, tr):
        return Cone(self._apex, self._axis, np.sqrt(self._t2),
                    sheet=self._sheet, transform=tr, **self.options)

    def mcnp_words(self):
        words = Surface.mcnp_words(self)
        if np.all(self._get_axis() == np.array([1.0, 0.0, 0.0])):
            if self._get_apex()[1] == 0.0 and self._get_apex()[2] == 0.0:
                words.append('KX')
                words.append(' ')
                v = self._apex[0]
                p = self._apex_digits[0]
                words.append(pretty_float(v, p))
            else:
                words.append('K/X')
                for v, p in zip(self._apex, self._apex_digits):
                    words.append(' ')
                    words.append(pretty_float(v, p))
        elif np.all(self._get_axis() == np.array([0.0, 1.0, 0.0])):
            if self._get_apex()[0] == 0.0 and self._get_apex()[2] == 0.0:
                words.append('KY')
                words.append(' ')
                v = self._apex[1]
                p = self._apex_digits[1]
                words.append(pretty_float(v, p))
            else:
                words.append('K/Y')
                for v, p in zip(self._apex, self._apex_digits):
                    words.append(' ')
                    words.append(pretty_float(v, p))
        elif np.all(self._get_axis() == np.array([0.0, 0.0, 1.0])):
            if self._get_apex()[0] == 0.0 and self._get_apex()[1] == 0.0:
                words.append('KZ')
                words.append(' ')
                v = self._apex[2]
                p = self._apex_digits[2]
                words.append(pretty_float(v, p))
            else:
                words.append('K/Z')
                for v, p in zip(self._apex, self._apex_digits):
                    words.append(' ')
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
            return GQuadratic(m, v, k, **self.options).mcnp_repr()
        words.append(' ')
        v = self._t2
        p = self._t2_digits
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
        self._m_digits = significant_array(m, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)
        self._v_digits = significant_array(v, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)
        self._k_digits = significant_digits(k, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)

        Surface.__init__(self, **options)
        _GQuadratic.__init__(self, m, v, k)

    def copy(self):
        instance = GQuadratic.__new__(GQuadratic, self._m, self._v, self._k)
        instance._m_digits = self._m_digits
        instance._v_digits = self._v_digits
        instance._k_digits = self._k_digits
        Surface.__init__(instance, **self.options)
        _GQuadratic.__init__(instance, self._m, self._v, self._k)
        return instance

    def __hash__(self):
        result = hash(self._get_k())
        for v in self._get_v():
            result ^= hash(v)
        for x in self._get_m().ravel():
            result ^= hash(x)
        return result

    def __eq__(self, other):
        if not isinstance(other, GQuadratic):
            return False
        else:
            for x, y in zip(self._get_v(), other._get_v()):
                if x != y:
                    return False
            for x, y in zip(self._get_m().ravel(), other._get_m().ravel()):
                if x != y:
                    return False
            return self._get_k() == other._get_k()

    def _get_m(self):
        return round_array(self._m, self._m_digits)

    def _get_v(self):
        return round_array(self._v, self._v_digits)

    def _get_k(self):
        return round_scalar(self._k, self._k_digits)

    def transform(self, tr):
        return GQuadratic(self._m, self._v, self._k, transform=tr,
                          **self.options)

    def mcnp_words(self):
        words = Surface.mcnp_words(self)
        words.append('GQ')
        m = self._get_m()
        a, b, c = np.diag(m)
        d = m[0, 1] + m[1, 0]
        e = m[1, 2] + m[2, 1]
        f = m[0, 2] + m[2, 0]
        g, h, j = self._get_v()
        k = self._get_k()
        for v in [a, b, c, d, e, f, g, h, j, k]:
            words.append(' ')
            p = significant_digits(v, constants.FLOAT_TOLERANCE, constants.FLOAT_TOLERANCE)
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
        self._axis_digits = significant_array(axis, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)
        self._center_digits = significant_array(center, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)
        self._R_digits = significant_digits(R, constants.FLOAT_TOLERANCE)
        self._a_digits = significant_digits(a, constants.FLOAT_TOLERANCE)
        self._b_digits = significant_digits(b, constants.FLOAT_TOLERANCE)

        Surface.__init__(self, **options)
        _Torus.__init__(self, center, axis, R, a, b)

    def copy(self):
        instance = Torus.__new__(Torus, self._center, self._axis, self._R, self._a, self._b)
        instance._axis_digits = self._axis_digits
        instance._center_digits = self._center_digits
        instance._R_digits = self._R_digits
        instance._a_digits = self._a_digits
        instance._b_digits = self._b_digits
        Surface.__init__(instance, **self.options)
        _Torus.__init__(instance, self._center, self._axis, self._R, self._a, self._b)
        return instance

    def __hash__(self):
        result = hash(self._get_R()) ^ hash(self._get_a()) ^ hash(self._get_b())
        for c in self._get_center():
            result ^= hash(c)
        for a in self._get_axis():
            result ^= hash(a)
        return result

    def __eq__(self, other):
        if not isinstance(other, Torus):
            return False
        else:
            for x, y in zip(self._get_center(), other._get_center()):
                if x != y:
                    return False
            for x, y in zip(self._get_axis(), other._get_axis()):
                if x != y:
                    return False
            return self._get_R() == other._get_R() and self._get_a() == other._get_a() and self._get_b() == other._get_b()

    def _get_axis(self):
        return round_array(self._axis, self._axis_digits)

    def _get_center(self):
        return round_array(self._center, self._center_digits)

    def _get_R(self):
        return round_scalar(self._R, self._R_digits)

    def _get_a(self):
        return round_scalar(self._a, self._a_digits)

    def _get_b(self):
        return round_scalar(self._b, self._b_digits)

    def transform(self, tr):
        return Torus(self._center, self._axis, self._R, self._a, self._b,
                     transform=tr, **self.options)

    def mcnp_words(self):
        words = Surface.mcnp_words(self)
        if np.all(self._get_axis() == np.array([1.0, 0.0, 0.0])):
            words.append('TX')
        elif np.all(self._get_axis() == np.array([0.0, 1.0, 0.0])):
            words.append('TY')
        elif np.all(self._get_axis() == np.array([0.0, 0.0, 1.0])):
            words.append('TZ')
        x, y, z = self._get_center()
        values = [x, y, z, self._get_R(), self._get_a(), self._get_b()]
        digits = [*self._center_digits, self._R_digits, self._a_digits, self._b_digits]
        for v, p in zip(values, digits):
            words.append(' ')
            words.append(pretty_float(v, p))
        return print_card(words)
