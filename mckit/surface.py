# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Text, Optional, Callable
import math
from abc import abstractmethod
import functools as ft
import numpy as np
from . import constants
from . utils.tolerance import tolerance_estimator

# noinspection PyUnresolvedReferences,PyPackageRequirements
from .geometry import Plane      as _Plane,    \
                      Sphere     as _Sphere,   \
                      Cone       as _Cone,     \
                      Cylinder   as _Cylinder, \
                      Torus      as _Torus,    \
                      GQuadratic as _GQuadratic, \
                      ORIGIN, EX, EY, EZ
from mckit.box import GLOBAL_BOX
from .printer import print_card, pretty_float
from .transformation import Transformation
from .utils import *
from .card import Card
from .constants import DROP_OPTIONS
import mckit.body


# noinspection PyUnresolvedReferences,PyPackageRequirements
__all__ = [
    'create_surface',
    'Plane',
    'Sphere',
    'Cone',
    'Torus',
    'GQuadratic',
    'Cylinder',
    'Surface',
    'ORIGIN',
    'EX',
    'EY',
    'EZ',
]

# TODO dvp: check if this solution is correct
# For Cone ta parameter the tolerance should be higher for round() satisfy tests
CONE_TA_TOLERANCE = constants.FLOAT_TOLERANCE


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
    params = np.asarray(params, dtype=float)
    kind = kind.upper()
    if kind[-1] == 'X':
        axis = EX
        assume_normalized = True
    elif kind[-1] == 'Y':
        axis = EY
        assume_normalized = True
    elif kind[-1] == 'Z':
        axis = EZ
        assume_normalized = True
    else:
        assume_normalized = False
    # -------- Plane -------------------
    if kind[0] == 'P':
        if len(kind) == 2:
            return Plane(axis, -params[0], assume_normalized=assume_normalized, **options)
        else:
            return Plane(params[:3], -params[3], assume_normalized=assume_normalized, **options)
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
        return Cylinder(r0, axis, R, assume_normalized=assume_normalized, **options)
    # -------- Cone ---------------
    elif kind[0] == 'K':
        if kind[1] == '/':
            r0 = np.array(params[:3], dtype=float)
            ta = params[3]
        else:
            r0 = params[0] * axis
            ta = params[1]
        sheet = 0 if len(params) % 2 == 0 else int(params[-1])
        return Cone(r0, axis, ta, sheet, assume_normalized=assume_normalized, **options)
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
                s = int(round((h1 - h0) / abs(h1 - h0)))  # TODO: dvp check this conversion: was without int()
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

    # def __hash__(self):
    #     return id(self)
    #
    # def __eq__(self, other):
    #     return id(self) == id(other)

    def __getstate__(self):
        return self.options

    def __setstate__(self, state):
        self.options = state

    @abstractmethod
    def copy(self) -> 'Surface':
        pass

    @property
    def transformation(self) -> Optional[Transformation]:
        transformation: Transformation = self.options.get('transform', None)
        return transformation

    @abstractmethod
    def apply_transformation(self) -> 'Plane':
        """
        Applies transformation specified for the surface.

        Returns
        -------
        A 1new surface with transformed parameters, If there's specified transformation,
        otherwise returns self.
        """

    def combine_transformations(self, tr: Transformation) -> Transformation:
        my_transformation: Transformation = self.transformation
        if my_transformation:
            tr = tr.apply2transform(my_transformation)
        return tr


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
    def is_close_to(
            self,
            other: 'Surface',
            estimator: Callable[[Any, Any], bool] = tolerance_estimator()
    ) -> bool:
        """
        Checks if this surface is close to other one with the given toleratnce values.
        """

    @abstractmethod
    def round(self) -> 'Surface':
        """
        Returns rounded version of self
        """

    def mcnp_words(self, pretty=False):
        words = []
        mod = self.options.get('modifier', None)
        if mod:
            words.append(mod)
        words.append(str(self.name()))
        words.append(' ')
        tr = self.transformation
        if tr is not None:
            words.append(tr.name())
            words.append(' ')
        return words

    def clean_options(self) -> Dict[Text, Any]:
        return filter_dict(self.options, DROP_OPTIONS)

    def compare_transformations(self, tr: Transformation) -> bool:
        my_transformation = self.transformation
        if my_transformation is not None:
            if tr is None:
                return False
            return my_transformation == tr
        else:
            return tr is None

    def has_close_transformations(
            self,
            tr: Transformation,
            estimator: Callable[[Any, Any], bool]
    ) -> bool:
        my_transformation = self.transformation
        if my_transformation is None:
            return tr is None
        if tr is None:
            return False
        return my_transformation.is_close(tr, estimator)


def internalize_ort(v: np.ndarray) -> np.ndarray:
    if v is EX or np.array_equal(v, EX):
        return EX, True
    elif v is EY or np.array_equal(v, EY):
        return EY, True
    elif v is EZ or np.array_equal(v, EZ):
        return EZ, True
    return v, False


def add_float(words: List[str], v: float, pretty: bool) -> None:
    words.append(' ')
    if pretty:
        words.append(pretty_float(v))
    else:
        words.append(prettify_float(v))


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
    def __init__(self, normal, offset, assume_normalized=False, **options):
        # if 'transform' in options.keys():
        #     tr = options.pop('transform')
        #     v, k = tr.apply2plane(normal, offset)
        # else:
        #     v = np.array(normal)
        #     k = offset
        v = np.asarray(normal, dtype=np.float)
        k = float(offset)
        if not assume_normalized:
            v, is_ort = internalize_ort(v)
            if not is_ort:
                length = np.linalg.norm(v)
                v = v / length
                k /= length
        # self._k_digits = significant_digits(k, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)
        # self._v_digits = significant_array(v, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)
        Surface.__init__(self, **options)
        _Plane.__init__(self, v, k)
        # self.normal = normal
        # self.offset = offset

    def round(self):
        result = self.apply_transformation()
        k_digits = significant_digits(result._k, constants.FLOAT_TOLERANCE, constants.FLOAT_TOLERANCE)
        v_digits = significant_array(result._v, constants.FLOAT_TOLERANCE, constants.FLOAT_TOLERANCE)
        k = round_scalar(result._k, k_digits)
        v = round_array(result._v, v_digits)
        return Plane(v, k, transform=None, assume_normalized=True, **result.options)

    def copy(self):
        # instance = Plane.__new__(Plane, self._v, self._k)
        # # instance._k_digits = self._k_digits
        # # instance._v_digits = self._v_digits
        # Surface.__init__(instance, **self.options)
        # _Plane.__init__(instance, self._v, self._k)
        # return instance
        options = filter_dict(self.options)
        instance = Plane(self._v, self._k, assume_normalized=True, **options)
        return instance

    __deepcopy__ = copy

    def apply_transformation(self) -> 'Plane':
        if 'transform' in self.options.keys():
            tr = self.options.pop('transform')
            v, k = tr.apply2plane(self._v, self._k)
        else:
            return self
        options = self.clean_options()
        return Plane(v, k, transform=None, assume_normalized=True, **options)

    def transform(self, tr: Transformation) -> 'Plane':
        if tr is None:
            return self
        tr = self.combine_transformations(tr)
        options = self.clean_options()
        return Plane(self._v, self._k, transform=tr, assume_normalized=True, **options)

    def _get_k(self):
        return self._k

    def _get_v(self):
        return self._v

    def reverse(self):
        """Gets the surface with reversed normal."""
        options = self.clean_options()
        return Plane(-self._v, -self._k, assume_normalized=True, **options)

    def mcnp_words(self, pretty=False):
        words = Surface.mcnp_words(self, pretty)
        if np.array_equal(self._v, EX):
            words.append('PX')
        elif np.array_equal(self._v, EY):
            words.append('PY')
        elif np.array_equal(self._v, EZ):
            words.append('PZ')
        else:
            words.append('P')
            for v in self._v:
                add_float(words, v, pretty)
        add_float(words, -self._k, pretty)  # TODO dvp: check why is offset negated in create_surface()?
        return print_card(words)

    def is_close_to(
            self,
            other: 'Surface',
            estimator: Callable[[Any, Any], bool] = tolerance_estimator()
    ) -> bool:
        if self is other:
            return True
        assert isinstance(other, Plane)
        if estimator((self._k, self._v), (other._k, other._v)):
            return self.has_close_transformations(other.transformation, estimator)
        return False

    def __hash__(self):
        # result = hash(self._get_k())
        # for v in self._get_v():
        #     result ^= hash(v)
        # return result
        return make_hash(self._k, self._v, self.transformation)

    def __eq__(self, other):
        # if not isinstance(other, Plane):
        #     return False
        # else:
        #     for x, y in zip(self._get_v(), other._get_v()):
        #         if x != y:
        #             return False
        #     return self._get_k() == other._get_k()
        if self is other:
            return True
        if not isinstance(other, Plane):
            return False
        if self._k == other._k:
            if np.array_equal(self._v, other._v):
                return self.compare_transformations(other.transformation)
        return False

    def __getstate__(self):
        # return self._v, self._k, self._k_digits, self._v_digits, Surface.__getstate__(self)
        return self._v, self._k, Surface.__getstate__(self)

    def __setstate__(self, state):
        v, k, options = state
        _Plane.__init__(self, v, k)
        Surface.__setstate__(self, options)
        # self._k_digits, self._v_digits = _k_digits, _v_digits

    def __repr__(self):
        return f"Plane({self._v}, {self._k}, {self.options if self.options else ''})"


# noinspection PyProtectedMember
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
        # if 'transform' in options.keys():
        #     tr = options.pop('transform')
        #     center = tr.apply2point(center)
        center = np.asarray(center, dtype=np.float)
        radius = float(radius)
        Surface.__init__(self, **options)
        # self._center_digits = significant_array(np.array(center), constants.FLOAT_TOLERANCE,
        #                                         resolution=constants.FLOAT_TOLERANCE)
        # self._radius_digits = significant_digits(radius, constants.FLOAT_TOLERANCE,
        #                                          resolution=constants.FLOAT_TOLERANCE)
        _Sphere.__init__(self, center, radius)

    def __getstate__(self):
        return self._center, self._radius, Surface.__getstate__(self)

    def __setstate__(self, state):
        c, r, options = state
        _Sphere.__init__(self, c, r)
        Surface.__setstate__(self, options)

    def copy(self):
        # instance = Sphere.__new__(Sphere, self._center, self._radius)
        # instance._center_digits = self._center_digits
        # instance._radius_digits = self._radius_digits
        # Surface.__init__(instance, **self.options)
        # _Sphere.__init__(instance, self._center, self._radius)
        # return instance
        return Sphere(self._center, self._radius,  **deepcopy(self.options))

    __deepcopy__ = copy

    def __repr__(self):
        return f"Sphere({self._center}, {self._radius}, {self.options if self.options else ''})"

    def __hash__(self):
        # result = hash(self._get_radius())
        # for c in self._get_center():
        #     result ^= hash(c)
        # return result
        return make_hash(self._radius, self._center, self.transformation)

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Sphere):
            return False
        if are_equal((self._radius, self._center), (other._radius, other._center)):
            return self.compare_transformations(other.transformation)
        return False

    def is_close_to(
            self,
            other: 'Sphere',
            estimator: Callable[[Any, Any], bool] = tolerance_estimator()
    ) -> bool:
        if self is other:
            return True
        if not isinstance(other, Sphere):
            return False
        if estimator((self._radius, self._center), (other._radius, other._center)):
            return self.has_close_transformations(other.transformation)
        return False

    def _get_center(self):
        return self._center

    def _get_radius(self):
        return self._radius

    def round(self) -> 'Surface':
        temp = self.apply_transformation()
        center_digits = significant_array(
            temp._center,
            constants.FLOAT_TOLERANCE,
            resolution=constants.FLOAT_TOLERANCE
        )
        center = round_array(temp._center, center_digits)
        radius_digits = significant_digits(
            temp._radius,
            constants.FLOAT_TOLERANCE,
            resolution=constants.FLOAT_TOLERANCE
        )
        radius = round_scalar(temp._radius, radius_digits)
        options = temp.clean_options()
        return Sphere(center, radius, transform=None, **options)

    def apply_transformation(self) -> 'Sphere':
        tr = self.transformation
        if tr is None:
            return self
        center = tr.apply2point(self._center)
        options = self.clean_options()
        return Sphere(center, self._radius, transform=None, **options)

    def transform(self, tr):
        if tr is None:
            return self
        tr = self.combine_transformations(tr)
        options = self.clean_options()
        return Sphere(self._center, self._radius, transform=tr, **options)

    def mcnp_words(self, pretty=False):
        words = Surface.mcnp_words(self, pretty)
        c = self._center
        r = self._radius
        if np.array_equal(self._center, ORIGIN):
            words.append('SO')
        elif c[0] == 0.0 and c[1] == 0.0:
            words.append('SZ')
            add_float(words, c[2], pretty)
        elif c[1] == 0.0 and c[2] == 0.0:
            words.append('SX')
            add_float(words, c[0], pretty)
        elif c[0] == 0.0 and c[2] == 0.0:
            words.append('SY')
            add_float(words, c[1], pretty)
        else:
            words.append('S')
            for v in c:
                add_float(words, v, pretty)
        add_float(words, self._radius, pretty)
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
    def __init__(self, pt, axis, radius, assume_normalized=False, **options):
        # if 'transform' in options.keys():
        #     tr = options.pop('transform')
        #     pt = tr.apply2point(pt)
        #     axis = tr.apply2vector(axis)
        axis = np.asarray(axis, dtype=np.float)
        if not assume_normalized:
            axis, is_ort = internalize_ort(axis)
            if not is_ort:
                axis /= np.linalg.norm(axis)
        maxdir = np.argmax(np.abs(axis))
        if axis[maxdir] < 0:
            axis *= -1
        pt = np.asarray(pt, dtype=np.float)
        # self._axis_digits = significant_array(axis, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)
        pt = pt - axis * np.dot(pt, axis)
        # self._pt_digits = significant_array(pt, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)
        # self._radius_digits = significant_digits(radius, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)
        Surface.__init__(self, **options)
        _Cylinder.__init__(self, pt, axis, radius)

    def __getstate__(self):
        return self._pt, self._axis, self._radius, Surface.__getstate__(self)

    def __setstate__(self, state):
        pt, axis, radius, options = state
        _Cylinder.__init__(self, pt, axis, radius)
        Surface.__setstate__(self, options)

    def copy(self):
        # instance = Cylinder.__new__(Cylinder, self._pt, self._axis, self._radius)
        # instance._axis_digits = self._axis_digits
        # instance._pt_digits = self._pt_digits
        # instance._radius_digits = self._radius_digits
        # Surface.__init__(instance, **self.options)
        # _Cylinder.__init__(instance, self._pt, self._axis, self._radius)
        # return instance
        return Cylinder(self._pt, self._axis, self._radius, assume_normalized=True, **deepcopy(self.options))

    __deepcopy__ = copy

    def __repr__(self):
        return f"Cylinder({self._pt}, {self._axis}, {self.options if self.options else ''})"

    def __hash__(self):
        # result = hash(self._get_radius())
        # for c in self._get_pt():
        #     result ^= hash(c)
        # for a in self._get_axis():
        #     result ^= hash(a)
        # return result
        return make_hash(self._radius, self._pt, self._axis, self.transformation)

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Cylinder):
            return False
        if self._radius == other._radius:
            if np.array_equal(self._pt, other._pt):
                if np.array_equal(self._axis, other._axis):
                    return self.compare_transformations(other.transformation)
        return False

    def is_close_to(
            self,
            other: 'Cylinder',
            estimator: Callable[[Any, Any], bool] = tolerance_estimator()
    ) -> bool:
        if self is other:
            return True
        if not isinstance(other, Cone):
            return False
        if estimator((self._radius, self._pt, self._axis), (other._radius, other._pt, other._axis)):
            return self.has_close_transformations(other.transformation, estimator)
        return False

    # def _get_pt(self):
    #     # return round_array(self._pt, self._pt_digits)
    #     return self._pt
    #
    # def _get_axis(self):
    #     # return round_array(self._axis, self._axis_digits)
    #     return self._axis
    #
    # def _get_radius(self):
    #     # return round_scalar(self._radius, self._radius_digits)
    #     return self._radius

    def round(self):
        temp = self.apply_transformation()
        pt = temp._pt
        pt_digits = significant_array(
            pt,
            constants.FLOAT_TOLERANCE,
            resolution=constants.FLOAT_TOLERANCE
        )
        pt = round_array(pt, pt_digits)
        axis = temp._axis
        axis_digits = significant_array(
            axis,
            constants.FLOAT_TOLERANCE,
            resolution=constants.FLOAT_TOLERANCE,
        )
        axis = round_array(axis, axis_digits)
        radius = temp._radius
        radius_digits = significant_digits(
            radius,
            constants.FLOAT_TOLERANCE,
            resolution=constants.FLOAT_TOLERANCE,
        )
        radius = round_scalar(radius, radius_digits)
        options = self.clean_options()
        return Cylinder(pt, axis, radius, transform=None, **options)

    def apply_transformation(self) -> 'Cylinder':
        if 'transform' in self.options.keys():
            tr = self.options.pop('transform')
            pt = tr.apply2point(self._pt)
            axis = tr.apply2vector(self._axis)
            options = self.clean_options()
            return Cylinder(pt, axis, self._radius, assume_normalized=True, **options)
        else:
            return self

    def transform(self, tr):
        if tr is None:
            return self
        tr = self.combine_transformations(tr)
        options = self.clean_options()
        return Cylinder(self._pt, self._axis, self._radius, transform=tr, assume_normalized=True, **options)

    def mcnp_words(self, pretty=False):
        words = Surface.mcnp_words(self)
        axis = self._axis
        pt = self._pt
        if np.array_equal(axis, EX):
            if pt[1] == 0.0 and pt[2] == 0.0:
                words.append('CX')
            else:
                words.append('C/X')
                add_float(words, pt[1], pretty)
                add_float(words, pt[2], pretty)
        elif np.array_equal(axis, EY):
            if pt[0] == 0.0 and pt[2] == 0.0:
                words.append('CY')
            else:
                words.append('C/Y')
                add_float(words, pt[0], pretty)
                add_float(words, pt[2], pretty)
        elif np.array_equal(axis, EZ):
            if pt[0] == 0.0 and pt[1] == 0.0:
                words.append('CZ')
            else:
                words.append('C/Z')
                add_float(words, pt[0], pretty)
                add_float(words, pt[1], pretty)
        else:
            nx, ny, nz = axis
            m = np.array([[1-nx**2, -nx*ny, -nx*nz],
                          [-nx*ny, 1-ny**2, -ny*nz],
                          [-nx*nz, -ny*nz, 1-nz**2]], dtype=float)
            v = np.zeros(3)
            k = -self._radius**2
            m, v, k = Transformation(translation=self._pt).apply2gq(m, v, k)
            return GQuadratic(m, v, k, **self.options).mcnp_repr(pretty)
        add_float(words, self._radius, pretty)
        return print_card(words)


# noinspection PyProtectedMember
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
    def __init__(
            self,
            apex: ndarray,
            axis: ndarray,
            ta: float,
            sheet: int = 0,
            assume_normalized: bool = False,
            **options
    ) -> None:
        # if 'transform' in options.keys():
        #     tr = options.pop('transform')
        #     apex = tr.apply2point(apex)
        #     axis = tr.apply2vector(axis)
        axis = np.asarray(axis, dtype=np.float)
        if not assume_normalized:
            axis, is_ort = internalize_ort(axis)
            if not is_ort:
                axis /= np.linalg.norm(axis)
        maxdir = np.argmax(np.abs(axis))
        if axis[maxdir] < 0:
            axis *= -1
            sheet *= -1
        apex = np.asarray(apex, dtype=np.float)
        # self._axis_digits = significant_array(axis, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)
        # self._apex_digits = significant_array(apex, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)
        Surface.__init__(self, **options)
        # TODO rnr: Do something with ta! It is confusing. _Cone accept ta, but returns t2.
        # TODO dvp: I did something, let's check
        _Cone.__init__(self, apex, axis, ta, sheet)
        # self._t2_digits = significant_digits(self._t2, constants.FLOAT_TOLERANCE)

    def apply_transformation(self) -> 'Cone':
        if 'transform' in self.options.keys():
            tr = self.options.pop('transform')
            apex = tr.apply2point(self._apex)
            axis = tr.apply2vector(self._axis)
            sheet = self._sheet
            options = self.clean_options()
            return Cone(apex, axis, self._t2, sheet, assume_normalized=True, **options)
        else:
            return self

    def round(self):
        res = self.apply_transformation()
        apex = res._apex
        apex_digits = significant_array(apex, constants.FLOAT_TOLERANCE, constants.FLOAT_TOLERANCE)
        apex = round_array(apex, apex_digits)
        axis = res._axis
        axis_digits = significant_array(axis, constants.FLOAT_TOLERANCE, constants.FLOAT_TOLERANCE)
        axis = round_array(axis, axis_digits)
        t2_digits = significant_digits(self._t2, CONE_TA_TOLERANCE, CONE_TA_TOLERANCE)
        t2 = round_scalar(self._t2, t2_digits)
        sheet = self._sheet
        options = self.clean_options()
        return Cone(apex, axis, t2, sheet, assume_normalized=True, **options)

    def copy(self):
        # ta = np.sqrt(self._t2)
        # instance = Cone.__new__(Cone, self._apex, self._axis, ta, self._sheet)
        # instance._axis_digits = self._axis_digits
        # instance._apex_digits = self._apex_digits
        # instance._t2_digits = self._t2_digits
        # Surface.__init__(instance, **self.options)
        # _Cone.__init__(instance, self._apex, self._axis, ta, self._sheet)
        # return instance
        return Cone(self._apex, self._axis, self._t2, self._sheet, assume_normalized=True, **deepcopy(self.options))

    __deepcopy__ = copy

    def __repr__(self):
        return f"Cone({self._apex}, {self._axis}, {self._t2}, {self._sheet}, {self.options if self.options else ''})"

    def __getstate__(self):
        return self._apex, self._axis, self._t2, self._sheet, Surface.__getstate__(self)

    def __setstate__(self, state):
        apex, axis, t2, sheet, options = state
        _Cone.__init__(self, apex, axis, t2, sheet)
        Surface.__setstate__(self, options)

    def __hash__(self):
        # result = hash(self._get_t2()) ^ hash(self._sheet)
        # for c in self._get_apex():
        #     result ^= hash(c)
        # for a in self._get_axis():
        #     result ^= hash(a)
        # return result
        return make_hash(self._t2, self._sheet, self._apex, self._axis)

    def __eq__(self, other):

        # noinspection DuplicatedCode
        if self is other:
            return True

        if not isinstance(other, Cone):
            return False

        return are_equal(
            (self._t2, self._sheet, self._apex, self._axis),
            (other._t2, other._sheet, other._apex, other._axis),
        )

    def is_close_to(
            self,
            other: 'Surface',
            estimator: Callable[[Any, Any], bool] = tolerance_estimator()
    ) -> bool:
        if self is other:
            return True
        if not isinstance(other, Cone):
            return False
        if estimator((self._apex, self._axis, self._t2), (other._apex, other._axis, other._t2)):
            return self.has_close_transformations(other.transformation, estimator)
        return False

    # def _get_axis(self):
    #     # return round_array(self._axis, self._axis_digits)
    #     return self._axis
    #
    # def _get_apex(self):
    #     # return round_array(self._apex, self._apex_digits)
    #     return self._apex
    #
    # def _get_t2(self):
    #     # return round_scalar(self._t2, self._t2_digits)
    #     return self._t2

    def transform(self, tr: Transformation) -> 'Cone':
        if tr is None:
            return self
        tr = self.combine_transformations(tr)
        options = self.clean_options()
        cone = Cone(self._apex, self._axis, self._t2, sheet=self._sheet, transform=tr, **options)
        # TODO dvp: check if the following code returning shape instead of Cone is necessary?
        # if self._sheet != 0:
        #     plane = Plane(self._axis, -np.dot(self._axis, self._apex), name=1, transform=tr)
        #     if self._sheet == +1:
        #         op = 'C'
        #     else:
        #         op = 'S'
        #     return mckit.body.Shape('U', cone, mckit.body.Shape(op, plane))
        return cone

    def mcnp_words(self, pretty=False):
        words = Surface.mcnp_words(self)
        axis = self._axis
        apex = self._apex
        if np.array_equal(axis, EX):
            if apex[1] == 0.0 and apex[2] == 0.0:
                words.append('KX')
                # words.append(' ')
                # v = self._apex[0]
                # p = self._apex_digits[0]
                # words.append(pretty_float(v, p))
                add_float(words, apex[0], pretty)
            else:
                words.append('K/X')
                for v in apex:
                    # words.append(' ')
                    # words.append(pretty_float(v, p))
                    add_float(words, v, pretty)
        elif np.array_equal(axis, EY):
            if apex[0] == 0.0 and apex[2] == 0.0:
                words.append('KY')
                # words.append(' ')
                # v = self._apex[1]
                # p = self._apex_digits[1]
                # words.append(pretty_float(v, p))
                add_float(words, apex[1], pretty)
            else:
                words.append('K/Y')
                for v in apex:
                    # words.append(' ')
                    # words.append(pretty_float(v, p))
                    add_float(words, v, pretty)
        elif np.array_equal(axis,  EZ):
            if apex[0] == 0.0 and apex[1] == 0.0:
                words.append('KZ')
                # words.append(' ')
                # v = self._apex[2]
                # p = self._apex_digits[2]
                # words.append(pretty_float(v, p))
                add_float(words, apex[2], pretty)
            else:
                words.append('K/Z')
                for v in apex:
                    # words.append(' ')
                    # words.append(pretty_float(v, p))
                    add_float(words, v, pretty)
        else:
            nx, ny, nz = axis
            a = 1 + self._t2
            m = np.array([[1-a*nx**2, -a*nx*ny, -a*nx*nz],
                          [-a*nx*ny, 1-a*ny**2, -a*ny*nz],
                          [-a*nx*nz, -a*ny*nz, 1-a*nz**2]])
            v = np.zeros(3)
            k = 0
            m, v, k = Transformation(translation=self._apex).apply2gq(m, v, k)
            return GQuadratic(m, v, k, **self.clean_options()).mcnp_repr(pretty)
        # words.append(' ')
        # v = self._t2
        # p = self._t2_digits
        # words.append(pretty_float(v, p))
        add_float(words, self._t2, pretty)
        if self._sheet != 0:
            words.append(' ')
            words.append(str(self._sheet))
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
        L = np.linalg.eigvalsh(m)
        factor = 1 / np.max(np.abs(L))
        self._m_digits = significant_array(m, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)
        self._v_digits = significant_array(v, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)
        self._k_digits = significant_digits(k, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)

        Surface.__init__(self, **options)
        _GQuadratic.__init__(self, m, v, k, factor)

    def __getstate__(self):
        return self._m, self._v, self._k, self._factor, Surface.__getstate__(self)

    def __setstate__(self, state):
        m, v, k, factor, options = state
        _GQuadratic.__init__(self, m, v, k, factor)
        Surface.__setstate__(self, options)

    def copy(self):
        instance = GQuadratic.__new__(GQuadratic, self._m, self._v, self._k, self._factor)
        instance._m_digits = self._m_digits
        instance._v_digits = self._v_digits
        instance._k_digits = self._k_digits
        Surface.__init__(instance, **self.options)
        _GQuadratic.__init__(instance, self._m, self._v, self._k, self._factor)
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

    def mcnp_words(self, pretty=False):
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


# noinspection PyProtectedMember
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
    def __init__(self, center, axis, R, a, b, assume_normalized=False, **options):
        # if 'transform' in options.keys():
        #     tr = options.pop('transform')
        #     center = tr.apply2point(center)
        #     axis = tr.apply2vector(axis)
        # else:
        center = np.asarray(center, dtype=float)
        axis = np.asarray(axis, dtype=float)
        if not assume_normalized:
            axis, is_ort = internalize_ort(axis)
            if not is_ort:
                axis /= np.linalg.norm(axis)
        maxdir = np.argmax(np.abs(axis))
        if axis[maxdir] < 0:
            axis *= -1
        # self._axis_digits = significant_array(axis, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)
        # self._center_digits = significant_array(center, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE)
        # self._R_digits = significant_digits(R, constants.FLOAT_TOLERANCE)
        # self._a_digits = significant_digits(a, constants.FLOAT_TOLERANCE)
        # self._b_digits = significant_digits(b, constants.FLOAT_TOLERANCE)
        Surface.__init__(self, **options)
        _Torus.__init__(self, center, axis, R, a, b)

    def round(self) -> 'Surface':
        temp = self.apply_transformation()
        center, axis = map(round_array, [temp._center, temp._axis])

        def r(x):
            return round_scalar(x, significant_digits(x, FLOAT_TOLERANCE))

        r, a, b = map(r, [temp._R, temp._a, temp._b])
        options = temp.clean_options()
        return Torus(center, axis, r, a, b, assume_normalized=True, **options)

    def apply_transformation(self) -> 'Torus':
        tr = self.transformation
        if tr is None:
            return self
        center = tr.apply2point(self._center)
        axis = tr.apply2vector(self._axis)
        # TODO dvp: should we check the transformation and result? The axis is to be along EX, EY, EZ.
        return Torus(center, axis, self._R, self._a, self._b, assume_normalized=True, **self.clean_options())

    def __getstate__(self):
        return self._center, self._axis, self._R, self._a, self._b, Surface.__getstate__(self)

    def __setstate__(self, state):
        center, axis, R, a, b, options = state
        _Torus.__init__(self, center, axis, R, a, b)
        Surface.__setstate__(self, options)

    def copy(self):
        # instance = Torus.__new__(Torus, self._center, self._axis, self._R, self._a, self._b)
        # instance._axis_digits = self._axis_digits
        # instance._center_digits = self._center_digits
        # instance._R_digits = self._R_digits
        # instance._a_digits = self._a_digits
        # instance._b_digits = self._b_digits
        # Surface.__init__(instance, **self.options)
        # _Torus.__init__(instance, self._center, self._axis, self._R, self._a, self._b)
        # return instance
        return Torus(
            self._center,
            self._axis,
            self._R,
            self._a,
            self._b,
            assume_normalized=True,
            **deepcopy(self.options)
        )

    def __hash__(self):
        # result = hash(self._get_R()) ^ hash(self._get_a()) ^ hash(self._get_b())
        # for c in self._get_center():
        #     result ^= hash(c)
        # for a in self._get_axis():
        #     result ^= hash(a)
        # return result
        return make_hash(self._center, self._axis, self._R, self._a, self._b)

    def __eq__(self, other: 'Torus'):
        if self is other:
            return True
        if not isinstance(other, Torus):
            return False
        if are_equal(
            (self._center, self._axis, self._R, self._a, self._b),
            (other._center, other._axis, other._R, other._a, other._b)
        ):
            return self.compare_transformations(other.transformation)
        return False

    def is_close_to(
            self,
            other: 'Torus',
            estimator: Callable[[Any, Any], bool] = tolerance_estimator()
    ) -> bool:
        if self is other:
            return True
        if not isinstance(other, Torus):
            return False
        if estimator(
            (self._center, self._axis, self._R, self._a, self._b),
            (other._center, other._axis, other._R, other._a, other._b),
        ):
            return self.has_close_transformations(other.transformation)
        return False

    # def _get_axis(self):
    #     return round_array(self._axis, self._axis_digits)
    #
    # def _get_center(self):
    #     return round_array(self._center, self._center_digits)
    #
    # def _get_R(self):
    #     return round_scalar(self._R, self._R_digits)
    #
    # def _get_a(self):
    #     return round_scalar(self._a, self._a_digits)
    #
    # def _get_b(self):
    #     return round_scalar(self._b, self._b_digits)

    def transform(self, tr):
        if tr is None:
            return self
        tr = self.combine_transformations(tr)
        options = self.clean_options()
        return Torus(self._center, self._axis, self._R, self._a, self._b,
                     transform=tr, **options)

    def mcnp_words(self, pretty=False):
        words = Surface.mcnp_words(self)
        estimator = tolerance_estimator()
        if estimator(self._axis, EX):
            words.append('TX')
        elif estimator(self._axis, EY):
            words.append('TY')
        elif estimator(self._axis, EZ):
            words.append('TZ')
        else:
            raise NotImplementedError("The axis of a torus should be along EX, EY or EZ")
        x, y, z = self._center
        for v in [x, y, z, self._R, self._a, self._b]:
            add_float(words, v, pretty)
        return print_card(words)

    def __repr__(self):
        return f"Torus({self._center}, {self._axis}, {self._R}, \
            {self._a}, {self._b}, {self.options if self.options else ''}"
