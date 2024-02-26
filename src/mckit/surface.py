"""Surface methods."""

from __future__ import annotations

from typing import Any, Callable, Optional, cast

from abc import abstractmethod

# noinspection PyPackageRequirements
import numpy as np

# noinspection PyPackageRequirements
import numpy.typing as npt

import mckit

from mckit.box import GLOBAL_BOX

# fmt:off
# noinspection PyUnresolvedReferences,PyPackageRequirements
from mckit.geometry import BOX as _BOX
from mckit.geometry import EX, EY, EZ, ORIGIN
from mckit.geometry import RCC as _RCC
from mckit.geometry import Cone as _Cone
from mckit.geometry import Cylinder as _Cylinder
from mckit.geometry import GQuadratic as _GQuadratic
from mckit.geometry import Plane as _Plane
from mckit.geometry import Sphere as _Sphere
from mckit.geometry import Torus as _Torus

from . import constants
from .card import Card
from .constants import DROP_OPTIONS
from .printer import pretty_float
from .transformation import Transformation
from .utils import (
    are_equal,
    filter_dict,
    round_array,
    round_scalar,
    significant_array,
    significant_digits,
)
from .utils.tolerance import DEFAULT_TOLERANCE_ESTIMATOR, FLOAT_TOLERANCE, MaybeClose

# fmt:on


# noinspection PyUnresolvedReferences,PyPackageRequirements
__all__ = [
    "create_surface",
    "Plane",
    "Sphere",
    "Cone",
    "Torus",
    "GQuadratic",
    "Cylinder",
    "Surface",
    "RCC",
    "BOX",
    "ORIGIN",
    "EX",
    "EY",
    "EZ",
]


VectorLike = npt.NDArray


# noinspection PyPep8Naming
def create_surface(kind: str, *_params: float, **options) -> Surface:
    """Creates new surface.

    Args:
        kind: Surface kind designator. See MCNP manual.
        _params: List of surface parameters.
        options: Dictionary of surface's options.
                In particular, transform  - transformation instance
                to be applied to the surface being created.

    Returns:
        New surface.

    Raises:
        NotImplementedError: when some logic is not implemented yet
        ValueError: on incompatible `params`
    """
    params = np.asarray(_params, dtype=float)
    kind = kind.upper()
    if kind[-1] == "X":
        axis = EX
    elif kind[-1] == "Y":
        axis = EY
    elif kind[-1] == "Z":
        axis = EZ
    else:
        axis = None
    surface = _create_surface_by_spec(axis, kind, options, params)

    if surface:
        return surface

    # ---------- Axis-symmetric surface defined by points ------
    if len(params) == 2:
        return Plane(axis, -params[0], **options)
    if len(params) == 4:
        # TODO: Use special classes instead of GQ
        h1, r1, h2, r2 = params
        if abs(h2 - h1) < constants.RESOLUTION * max(abs(h1), abs(h2)):
            return Plane(axis, -0.5 * (h1 + h2), **options)
        if abs(r2 - r1) < constants.RESOLUTION * max(abs(r2), abs(r1)):
            R = 0.5 * (abs(r1) + abs(r2))
            return Cylinder(np.array([0, 0, 0], dtype=float), axis, R, **options)
        if r1 * r2 < 0:
            raise ValueError("Points must belong to the one sheet.")
        h0 = (abs(r1) * h2 - abs(r2) * h1) / (abs(r1) - abs(r2))
        t2 = ((r1 - r2) / (h1 - h2)) ** 2
        s = round((h1 - h0) / abs(h1 - h0))
        return Cone(axis * h0, axis, t2, sheet=s, **options)
    # TODO: Implement creation of surface by 3 points.
    raise NotImplementedError()


def _create_surface_by_spec(axis, kind, options, params) -> Surface | None:  # noqa: PLR0911
    # -------- Plane -------------------
    if kind[0] == "P":
        return _create_plane(axis, kind, options, params)
    # -------- SQ -------------------
    if kind == "SQ":
        return _create_sq(options, params)
    # -------- Sphere ------------------
    if kind[0] == "S":
        return _create_sphere(axis, kind, options, params)
    # -------- Cylinder ----------------
    if kind[0] == "C":
        return _create_cylinder(axis, kind, options, params)
    # -------- Cone ---------------
    if kind[0] == "K":
        return _create_cone(axis, kind, options, params)
    # ---------- GQ -----------------
    if kind == "GQ":
        return _create_gquadratic(options, params)
    # ---------- Torus ---------------------
    if kind[0] == "T":
        return _create_torus(axis, options, params)
    # ---------- Macrobodies ---------------
    if kind == "RPP":
        return _create_rpp(options, params)
    if kind == "BOX":
        return _create_box(options, params)
    if kind == "RCC":
        return _create_rcc(options, params)
    return None


def _create_rcc(options, params) -> RCC:
    center = params[:3]
    axis = params[3:6]
    radius = params[6]
    return RCC(center, axis, radius, **options)


def _create_box(options, params) -> BOX:
    center = params[:3]
    dir_x = params[3:6]
    dir_y = params[6:9]
    dir_z = params[9:]
    return BOX(center, dir_x, dir_y, dir_z, **options)


def _create_rpp(options, params) -> BOX:
    x_min, x_max, y_min, y_max, z_min, z_max = params
    center = [x_min, y_min, z_min]
    dir_x = [x_max - x_min, 0, 0]
    dir_y = [0, y_max - y_min, 0]
    dir_z = [0, 0, z_max - z_min]
    return BOX(center, dir_x, dir_y, dir_z, **options)


def _create_torus(axis, options, params) -> Torus:
    x0, y0, z0, R, a, b = params
    return Torus([x0, y0, z0], axis, R, a, b, **options)


def _create_gquadratic(options, params) -> GQuadratic:
    A, B, C, D, E, F, G, H, J, k = params
    m = np.array([[A, 0.5 * D, 0.5 * F], [0.5 * D, B, 0.5 * E], [0.5 * F, 0.5 * E, C]])
    v = np.array([G, H, J])
    return GQuadratic(m, v, k, **options)


def _create_cone(axis, kind, options, params) -> Cone:
    if kind[1] == "/":
        r0 = np.array(params[:3], dtype=float)
        ta = params[3]
    else:
        r0 = params[0] * axis
        ta = params[1]
    sheet = 0 if len(params) % 2 == 0 else int(params[-1])
    return Cone(r0, axis, ta, sheet, **options)


def _create_cylinder(axis, kind, options, params) -> Cylinder:
    A = 1 - axis
    if kind[1] == "/":
        Ax, Az = np.dot(A, EX), np.dot(A, EZ)
        r0 = params[0] * (Ax * EX + (1 - Ax) * EY) + params[1] * ((1 - Az) * EY + Az * EZ)
    else:
        r0 = ORIGIN
    R = params[-1]
    return Cylinder(r0, axis, R, **options)


def _create_sphere(axis, kind, options, params) -> Sphere:
    if kind == "S":
        r0 = np.array(params[:3])
    elif kind == "SO":
        r0 = ORIGIN
    else:
        r0 = axis * params[0]
    R = params[-1]
    return Sphere(r0, R, **options)


def _create_sq(options: dict[str, Any], params: npt.NDArray) -> GQuadratic:
    A, B, C, D, E, F, G, x0, y0, z0 = params
    m = np.diag([A, B, C])
    v = 2 * np.array([D - A * x0, E - B * y0, F - C * z0])
    k = A * x0**2 + B * y0**2 + C * z0**2 - 2 * (D * x0 + E * y0 + F * z0) + G
    return GQuadratic(m, v, k, **options)


def _create_plane(
    axis: np.ndarray,
    kind: str,
    options: dict[str, Any],
    params: npt.NDArray,
) -> Plane:
    if len(kind) == 2:
        return Plane(axis, -params[0], **options)
    return Plane(params[:3], -params[3], **options)


def create_replace_dictionary(
    surfaces: set[Surface],
    unique: set[Surface] | None = None,
    box=GLOBAL_BOX,
    tol: float = 1.0e-10,
) -> dict[Surface, tuple[Surface, int]]:
    """Creates surface replace dictionary for equal surfaces removing.

    Args:
        surfaces : A set of surfaces to be checked.
        unique:  A set of surfaces that are assumed to be unique. If not None, then
                `surfaces` are checked for coincidence with one of them.
        box : Box
            A box, which is used for comparison.
        tol : float
            Tolerance

    Returns:
        A replacement dictionary. surface -> (replace_surface, sense). Sense is +1
        if surfaces have the same direction of normals. -1 otherwise.
    """
    replace = {}
    unique_surfaces = set() if unique is None else unique
    for s in surfaces:
        for us in unique_surfaces:
            # noinspection PyUnresolvedReferences
            t = s.equals(us, box=box, tol=tol)  # type: ignore[attr-defined]
            if t != 0:
                replace[s] = (us, t)
                break
        else:
            unique_surfaces.add(s)
    return replace


def _drop_empty_transformation(options: dict[str, Any]) -> None:
    if "transform" in options and not options["transform"]:  # empty transformation option
        del options["transform"]


class Surface(Card, MaybeClose):
    """Base class for all surface classes.

    Methods:
        equals(other, box, tol)
            Checks if this surface and surf are equal inside the box.
        test_point(p)
            Checks the sense of point p with respect to this surface.
        test_box(box)
            Checks whether this surface crosses the box.
        projection(p)
            Gets projection of point p on the surface.
    """

    def __init__(self, **options) -> None:
        """Create :class:`Surface` with the options.

        Args:
            options: kwargs - properties for the `Surface` card.
        """
        _drop_empty_transformation(options)
        Card.__init__(self, **options)

    @abstractmethod
    def copy(self) -> Surface:
        pass

    @property
    def transformation(self) -> Transformation | None:
        return cast(Optional[Transformation], self.options.get("transform", None))

    @abstractmethod
    def apply_transformation(self) -> Surface:
        """Applies transformation specified for the surface.

        Returns:
            A new surface with transformed parameters, if there's specified transformation,
            otherwise returns self.
        """

    def combine_transformations(self, tr: Transformation) -> Transformation:
        my_transformation: Transformation | None = self.transformation
        if my_transformation:
            return tr.apply2transform(my_transformation)
        return tr

    @abstractmethod
    def transform(self, tr: Transformation) -> Surface:
        """Applies transformation to this surface.

        Args:
            tr: Transformation to be applied.

        Returns:
            The result of this surface transformation.
        """

    @abstractmethod
    def is_close_to(
        self,
        other: Surface,
        estimator: Callable[[Any, Any], bool] = DEFAULT_TOLERANCE_ESTIMATOR,
    ) -> bool:
        """Checks if this surface is close to other one with the given tolerance values."""

    @abstractmethod
    def round(self) -> Surface:
        """Returns rounded version of self."""

    def mcnp_words(self, pretty: bool = False) -> list[str]:
        words = []
        mod = self.options.get("modifier", None)
        if mod:
            words.append(mod)
        words.append(str(self.name()))
        words.append(" ")
        # TODO dvp: add transformations processing in Universe.
        return words

    def clean_options(self) -> dict[str, Any]:
        result: dict[str, Any] = filter_dict(self.options, DROP_OPTIONS)
        return result

    def __getstate__(self):
        return self.options

    def __setstate__(self, state):
        self.options = state


def internalize_ort(v: np.ndarray) -> tuple[np.ndarray, bool]:
    if v is EX or np.array_equal(v, EX):
        return EX, True
    if v is EY or np.array_equal(v, EY):
        return EY, True
    if v is EZ or np.array_equal(v, EZ):
        return EZ, True
    return v, False


# noinspection PyProtectedMember
class RCC(Surface, _RCC):

    def __init__(self, center, direction, radius, **options):
        center = np.array(center)
        direction = np.array(direction)
        opt_surf = options.copy()
        opt_surf["name"] = 1
        norm = np.array(direction) / np.linalg.norm(direction)
        cyl = Cylinder(center, norm, radius, **opt_surf).apply_transformation()
        center2 = center + direction
        offset2 = -np.dot(norm, center2).item()
        offset3 = np.dot(norm, center).item()
        opt_surf["name"] = 2
        plane2 = Plane(norm, offset2, **opt_surf).apply_transformation()
        opt_surf["name"] = 3
        plane3 = Plane(-norm, offset3, **opt_surf).apply_transformation()
        _RCC.__init__(self, cyl, plane2, plane3)
        options.pop("transform", None)
        Surface.__init__(self, **options)
        self._hash = hash(cyl) ^ hash(plane2) ^ hash(plane3)

    def surface(self, number: int):
        args = self.surfaces
        if 1 <= number <= len(args):
            return args[number - 1]
        raise ValueError(f"There is no such surface in macrobody: {number}")

    def get_params(self):
        args = self.surfaces
        center = args[0]._pt - args[2]._k * args[0]._axis * np.dot(args[0]._axis, args[2]._v)
        direction = -(args[1]._k + args[2]._k) * args[1]._v
        radius = args[0]._radius
        return center, direction, radius

    def mcnp_words(self, pretty: bool = False) -> list[str]:
        words = Surface.mcnp_words(self, pretty)
        words.append("RCC")
        words.append(" ")
        center, direction, radius = self.get_params()
        values = list(center)
        values.extend(direction)
        values.append(radius)
        for v in values:
            fd = significant_digits(
                v, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE
            )
            words.append(pretty_float(v, fd))
            words.append(" ")
        return words

    def transform(self, tr: Transformation) -> RCC:
        """Transforms the shape.

        Args:
            tr:  Transformation to be applied.

        Returns:
            New RCC shape with the transformation stored in options.
        """
        center, direction, radius = self.get_params()
        # TODO(dvp): What if `self` already has transformation?
        #            Should we apply the both transformation on new one?
        return RCC(center, direction, radius, transform=tr)

    def apply_transformation(self) -> Surface:
        return self

    def is_close_to(
        self, other: Surface, estimator: Callable[[Any, Any], bool] = DEFAULT_TOLERANCE_ESTIMATOR
    ) -> Surface:
        pass

    def round(self) -> Surface:
        pass

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        if not isinstance(other, RCC):
            return False
        return self.surfaces == other.surfaces

    def __getstate__(self):
        surf_state = Surface.__getstate__(self)
        args = self.surfaces
        return args, self._hash, surf_state

    def __setstate__(self, state):
        args, hash_value, surf_state = state
        Surface.__setstate__(self, surf_state)
        _RCC.__init__(self, *args)
        self._hash = hash_value

    def copy(self):
        center, direction, radius = self.get_params()
        return RCC(center, direction, radius, **self.options)


# noinspection PyProtectedMember
class BOX(Surface, _BOX):
    """Macrobody BOX surface.

    Parameters
    ----------
    """

    def __init__(self, center, dir_x, dir_y, dir_z, **options):
        dir_x = np.array(dir_x)
        dir_y = np.array(dir_y)
        dir_z = np.array(dir_z)
        center = np.array(center)
        center2 = center + dir_x + dir_y + dir_z
        len_x = np.linalg.norm(dir_x)
        len_y = np.linalg.norm(dir_y)
        len_z = np.linalg.norm(dir_z)
        norm_x = dir_x / len_x
        norm_y = dir_y / len_y
        norm_z = dir_z / len_z
        offset_x = np.dot(norm_x, center).item()
        offset_y = np.dot(norm_y, center).item()
        offset_z = np.dot(norm_z, center).item()
        offset_2x = -np.dot(norm_x, center2).item()
        offset_2y = -np.dot(norm_y, center2).item()
        offset_2z = -np.dot(norm_z, center2).item()
        opt_surf = options.copy()
        opt_surf["name"] = 1
        surf1 = Plane(norm_x, offset_2x, **opt_surf).apply_transformation()
        opt_surf["name"] = 2
        surf2 = Plane(-norm_x, offset_x, **opt_surf).apply_transformation()
        opt_surf["name"] = 3
        surf3 = Plane(norm_y, offset_2y, **opt_surf).apply_transformation()
        opt_surf["name"] = 4
        surf4 = Plane(-norm_y, offset_y, **opt_surf).apply_transformation()
        opt_surf["name"] = 5
        surf5 = Plane(norm_z, offset_2z, **opt_surf).apply_transformation()
        opt_surf["name"] = 6
        surf6 = Plane(-norm_z, offset_z, **opt_surf).apply_transformation()
        _BOX.__init__(self, surf1, surf2, surf3, surf4, surf5, surf6)
        options.pop("transform", None)
        Surface.__init__(self, **options)
        self._hash = (
            hash(surf1) ^ hash(surf2) ^ hash(surf3) ^ hash(surf4) ^ hash(surf5) ^ hash(surf6)
        )

    def apply_transformation(self) -> Surface:
        pass

    def is_close_to(
        self, other: Surface, estimator: Callable[[Any, Any], bool] = DEFAULT_TOLERANCE_ESTIMATOR
    ) -> bool:
        pass

    def round(self) -> Surface:
        pass

    def surface(self, number: int):
        args = self.surfaces
        if 1 <= number <= len(args):
            return args[number - 1]
        raise ValueError(f"There is no such surface in macrobody: {number}")

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, BOX):
            return False
        args_this = self.surfaces
        args_other = other.surfaces
        return args_this == args_other

    @staticmethod
    def _get_plane_intersection(s1, s2, s3):
        matrix = np.zeros((3, 3))
        matrix[0, :] = -s1._v
        matrix[1, :] = -s2._v
        matrix[2, :] = -s3._v
        vector = np.array([s1._k, s2._k, s3._k])
        return np.linalg.solve(matrix, vector)

    def get_params(self):
        args = self.surfaces
        center = self._get_plane_intersection(args[1], args[3], args[5])
        point2 = self._get_plane_intersection(args[0], args[2], args[4])
        norm_x = args[0]._v
        norm_y = args[2]._v
        norm_z = args[4]._v
        diag = point2 - center
        dir_x = np.dot(norm_x, diag) * norm_x
        dir_y = np.dot(norm_y, diag) * norm_y
        dir_z = np.dot(norm_z, diag) * norm_z
        return center, dir_x, dir_y, dir_z

    def mcnp_words(self, pretty: bool = False) -> list[str]:
        words = Surface.mcnp_words(self, pretty)
        words.append("BOX")
        words.append(" ")
        center, dir_x, dir_y, dir_z = self.get_params()
        values = list(center)
        values.extend(dir_x)
        values.extend(dir_y)
        values.extend(dir_z)
        for v in values:
            fd = significant_digits(
                v, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE
            )
            words.append(pretty_float(v, fd))
            words.append(" ")
        return words

    def transform(self, tr: Transformation):
        """Transforms the shape.

        Args:
            tr:  Transformation to be applied.

        Returns:
            New shape.
        """
        center, dir_x, dir_y, dir_z = self.get_params()
        return BOX(center, dir_x, dir_y, dir_z, transform=tr)

    def __getstate__(self):
        surf_state = Surface.__getstate__(self)
        args = self.surfaces
        return args, self._hash, surf_state

    def __setstate__(self, state):
        args, hash_value, surf_state = state
        Surface.__setstate__(self, surf_state)
        _BOX.__init__(self, *args)
        self._hash = hash_value

    def copy(self):
        center, dir_x, dir_y, dir_z = self.get_params()
        return BOX(center, dir_x, dir_y, dir_z, **self.options)


class Plane(Surface, _Plane):
    """Plane surface class.

    Args:
        normal: The normal to the plane being created.
        offset: Free term.
        options:  Dictionary of surface's options. Possible values:
                  transform = transformation to be applied to this plane.
    """

    def __init__(
        self, normal: npt.NDArray[float], offset: float, **options: dict[str, Any]
    ) -> None:
        tr: Transformation | None = options.pop("transform", None)
        if tr:
            v, k = tr.apply2plane(normal, offset)
        else:
            v = np.asarray(normal, dtype=float)
            k = offset
        v, is_ort = internalize_ort(v)
        if not is_ort:
            length = np.linalg.norm(v)
            v /= length
            k /= length
        self._k_digits = significant_digits(
            k, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE
        )
        self._v_digits = significant_array(
            v, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE
        )
        Surface.__init__(self, **options)
        _Plane.__init__(self, v, k)
        # self._hash = compute_hash(self._k, self._v, self.transformation)

    # noinspection PyTypeChecker
    def round(self):
        result = self.apply_transformation()
        k_digits = significant_digits(
            result._k, constants.FLOAT_TOLERANCE, constants.FLOAT_TOLERANCE
        )
        v_digits = significant_array(
            result._v, constants.FLOAT_TOLERANCE, constants.FLOAT_TOLERANCE
        )
        k = round_scalar(result._k, k_digits)
        v = round_array(result._v, v_digits)
        return Plane(v, k, **result.options)

    def copy(self) -> Plane:
        """Create a copy of self.

        Skips Plane.__init__ (directly calls _Plane.__init__) to avoid time-consuming
        significant digits computation.

        Returns:
            New copy of self
        """
        instance = Plane.__new__(Plane, self._v, self._k)
        instance._k_digits = self._k_digits
        instance._v_digits = self._v_digits
        options = filter_dict(self.options)
        Surface.__init__(instance, **options)
        _Plane.__init__(instance, self._v, self._k)
        return instance

    def apply_transformation(self) -> Plane:
        tr = self.transformation
        if tr is None:
            return self
        v, k = tr.apply2plane(self._v, self._k)
        options = self.clean_options()
        return Plane(v, k, **options)

    def transform(self, tr: Transformation) -> Plane:
        if tr is None:
            return self
        tr = self.combine_transformations(tr)
        options = self.clean_options()
        options["transform"] = tr
        return Plane(self._v, self._k, **options)

    def reverse(self):
        """Gets the surface with reversed normal."""
        instance = Plane.__new__(Plane, -self._v, -self._k)
        instance._k_digits = self._k_digits
        instance._v_digits = self._v_digits
        options = self.clean_options()
        Surface.__init__(instance, **options)
        _Plane.__init__(instance, -self._v, -self._k)
        return instance

    def mcnp_words(self, pretty: bool = False) -> list[str]:
        words = Surface.mcnp_words(self, pretty)
        _v = self._get_v()
        if np.array_equal(_v, EX):
            words.append("PX")
        elif np.array_equal(_v, EY):
            words.append("PY")
        elif np.array_equal(_v, EZ):
            words.append("PZ")
        else:
            words.append("P")
            for v, p in zip(_v, self._v_digits):
                words.append(" ")
                words.append(pretty_float(v, p))
        words.append(" ")
        words.append(pretty_float(-self._get_k(), self._k_digits))
        return words

    def is_close_to(
        self,
        other: Surface,
        estimator: Callable[[Any, Any], bool] = DEFAULT_TOLERANCE_ESTIMATOR,
    ) -> bool:
        if self is other:
            return True
        if not isinstance(other, Plane):
            return False
        return estimator(
            (self._k, self._v, self.transformation),
            (other._k, other._v, other.transformation),
        )

    def __hash__(self):
        result = hash(self._get_k())
        for v in self._get_v():
            result ^= hash(v)
        return result

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Plane):
            return False
        return are_equal(
            (self._k, self._v, self.transformation),
            (other._k, other._v, other.transformation),
        )

    def __getstate__(self):
        return self._v, self._k, Surface.__getstate__(self)

    def __setstate__(self, state):
        v, k, options = state
        self.__init__(v, k, **options)

    def __repr__(self):
        return f"Plane({self._v}, {self._k}, {self.options if self.options else ''})"

    def _get_k(self):
        return round_scalar(self._k, self._k_digits)

    def _get_v(self):
        return round_array(self._v, self._v_digits)


# noinspection PyProtectedMember,PyTypeChecker
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
            transform = transformation to be applied to the sphere being
                             created. Transformation instance.
    """

    def __init__(
        self, center: npt.NDArray[float], radius: float, **options: dict[str, Any]
    ) -> None:
        tr: Transformation | None = options.pop("transform", None)
        if tr:
            center = tr.apply2point(center)
        else:
            center = np.asarray(center, dtype=float)
        radius = float(radius)
        Surface.__init__(self, **options)
        self._center_digits = significant_array(
            np.array(center), constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE
        )
        self._radius_digits = significant_digits(
            radius, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE
        )
        _Sphere.__init__(self, center, radius)

    def __getstate__(self):
        return self._center, self._radius, Surface.__getstate__(self)

    def __setstate__(self, state):
        c, r, options = state
        self.__init__(c, r, **options)

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
        for x, y in zip(self._get_center(), other._get_center()):
            if x != y:
                return False
        return self._get_radius() == other._get_radius()

    def __repr__(self):
        return f"Sphere({self._center}, {self._radius}, {self.options if self.options else ''})"

    def is_close_to(
        self,
        other: Sphere,
        estimator: Callable[[Any, Any], bool] = DEFAULT_TOLERANCE_ESTIMATOR,
    ) -> bool:
        if self is other:
            return True
        if not isinstance(other, Sphere):
            return False
        return estimator(
            (self._radius, self._center, self.transformation),
            (other._radius, other._center, other.transformation),
        )

    def round(self) -> Surface:
        temp = self.apply_transformation()
        center_digits = significant_array(
            temp._center,
            constants.FLOAT_TOLERANCE,
            resolution=constants.FLOAT_TOLERANCE,
        )
        center = round_array(temp._center, center_digits)
        radius_digits = significant_digits(
            temp._radius,
            constants.FLOAT_TOLERANCE,
            resolution=constants.FLOAT_TOLERANCE,
        )
        radius = round_scalar(temp._radius, radius_digits)
        options = temp.clean_options()
        return Sphere(center, radius, transform=None, **options)

    def apply_transformation(self) -> Sphere:
        tr = self.transformation
        if tr is None:
            return self
        center = tr.apply2point(self._center)
        options = self.clean_options()
        return Sphere(center, self._radius, transform=None, **options)

    def _get_center(self):
        return round_array(self._center, self._center_digits)

    def _get_radius(self):
        return round_scalar(self._radius, self._radius_digits)

    def transform(self, tr):
        return Sphere(self._center, self._radius, transform=tr, **self.options)

    def mcnp_words(self, pretty: bool = False):
        words = Surface.mcnp_words(self)
        if np.all(self._get_center() == np.array([0.0, 0.0, 0.0])):
            words.append("SO")
        elif self._get_center()[0] == 0.0 and self._get_center()[1] == 0.0:
            words.append("SZ")
            words.append(" ")
            v = self._get_center()[2]
            p = self._center_digits[2]
            words.append(pretty_float(v, p))
        elif self._get_center()[1] == 0.0 and self._get_center()[2] == 0.0:
            words.append("SX")
            words.append(" ")
            v = self._get_center()[0]
            p = self._center_digits[0]
            words.append(pretty_float(v, p))
        elif self._get_center()[0] == 0.0 and self._get_center()[2] == 0.0:
            words.append("SY")
            words.append(" ")
            v = self._get_center()[1]
            p = self._center_digits[1]
            words.append(pretty_float(v, p))
        else:
            words.append("S")
            for v, p in zip(self._center, self._center_digits):
                words.append(" ")
                words.append(pretty_float(v, p))
        words.append(" ")
        v = self._get_radius()
        p = self._radius_digits
        words.append(pretty_float(v, p))
        return words


# noinspection PyProtectedMember,PyUnresolvedReferences,DuplicatedCode,PyTypeChecker
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
            transform = transformation to be applied to the cylinder being
                             created. Transformation instance.
    """

    def __init__(
        self,
        pt: npt.NDArray[float],
        axis: npt.NDArray[float],
        radius: float,
        **options: dict[str, Any],
    ) -> None:
        tr: Transformation | None = options.pop("transform", None)
        if tr:
            pt = tr.apply2point(pt)
            axis = tr.apply2vector(axis)
        else:
            pt = np.asarray(pt, dtype=float)
            axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        max_dir = np.argmax(np.abs(axis))
        if axis[max_dir] < 0:
            axis *= -1
        self._axis_digits = significant_array(
            axis, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE
        )
        pt = pt - axis * np.dot(pt, axis)
        self._pt_digits = significant_array(
            pt, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE
        )
        self._radius_digits = significant_digits(
            radius, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE
        )
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

    def __repr__(self):
        return f"Cylinder({self._pt}, {self._axis}, {self._radius}, {self.options if self.options else ''})"

    def __hash__(self):
        result = hash(self._get_radius())
        for c in self._get_pt():
            result ^= hash(c)
        for a in self._get_axis():
            result ^= hash(a)
        return result

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Cylinder):
            return False
        for x, y in zip(self._get_pt(), other._get_pt()):
            if x != y:
                return False
        for x, y in zip(self._get_axis(), other._get_axis()):
            if x != y:
                return False
        return self._get_radius() == other._get_radius()

    def is_close_to(
        self,
        other: Cylinder,
        estimator: Callable[[Any, Any], bool] = DEFAULT_TOLERANCE_ESTIMATOR,
    ) -> bool:
        if self is other:
            return True
        if not isinstance(other, Cone):
            return False
        return estimator(
            (self._radius, self._pt, self._axis, self.transformation),
            (other._radius, other._pt, other._axis, other.transformation),
        )

    def _get_pt(self):
        return round_array(self._pt, self._pt_digits)

    def _get_axis(self):
        return round_array(self._axis, self._axis_digits)

    def _get_radius(self):
        return round_scalar(self._radius, self._radius_digits)

    def transform(self, tr):
        return Cylinder(self._pt, self._axis, self._radius, transform=tr, **self.options)

    def mcnp_words(self, pretty: bool = False):
        words = Surface.mcnp_words(self)
        if np.all(self._get_axis() == np.array([1.0, 0.0, 0.0])):
            if self._get_pt()[1] == 0.0 and self._get_pt()[2] == 0.0:
                words.append("CX")
            else:
                words.append("C/X")
                words.append(" ")
                v = self._get_pt()[1]
                p = self._pt_digits[1]
                words.append(pretty_float(v, p))
                words.append(" ")
                v = self._get_pt()[2]
                p = self._pt_digits[2]
                words.append(pretty_float(v, p))
        elif np.all(self._get_axis() == np.array([0.0, 1.0, 0.0])):
            if self._get_pt()[0] == 0.0 and self._get_pt()[2] == 0.0:
                words.append("CY")
            else:
                words.append("C/Y")
                words.append(" ")
                v = self._get_pt()[0]
                p = self._pt_digits[0]
                words.append(pretty_float(v, p))
                words.append(" ")
                v = self._get_pt()[2]
                p = self._pt_digits[2]
                words.append(pretty_float(v, p))
        elif np.all(self._get_axis() == np.array([0.0, 0.0, 1.0])):
            if self._get_pt()[0] == 0.0 and self._get_pt()[1] == 0.0:
                words.append("CZ")
            else:
                words.append("C/Z")
                words.append(" ")
                v = self._get_pt()[0]
                p = self._pt_digits[0]
                words.append(pretty_float(v, p))
                words.append(" ")
                v = self._get_pt()[1]
                p = self._pt_digits[1]
                words.append(pretty_float(v, p))
        else:
            nx, ny, nz = self._axis
            m = np.array(
                [
                    [1 - nx**2, -nx * ny, -nx * nz],
                    [-nx * ny, 1 - ny**2, -ny * nz],
                    [-nx * nz, -ny * nz, 1 - nz**2],
                ]
            )
            v = np.zeros(3)
            k = -self._radius**2
            m, v, k = Transformation(translation=self._pt).apply2gq(m, v, k)
            return GQuadratic(m, v, k, **self.options).mcnp_repr()
        words.append(" ")
        v = self._get_radius()
        p = self._radius_digits
        words.append(pretty_float(v, p))
        return words

    def round(self):
        temp = self.apply_transformation()
        pt = round_array(temp._pt)
        axis = round_array(temp._axis)
        radius = round_scalar(temp._radius)
        options = self.clean_options()
        return Cylinder(pt, axis, radius, transform=None, **options)

    def apply_transformation(self) -> Surface:
        tr = self.transformation
        if tr is None:
            return self
        pt = tr.apply2point(self._pt)
        axis = tr.apply2vector(self._axis)
        options = self.clean_options()
        # TODO dvp: actually may create Generic Quadratic. Should we use __new__() for this?
        return Cylinder(pt, axis, self._radius, **options)

    def __getstate__(self):
        return self._pt, self._axis, self._radius, Surface.__getstate__(self)

    def __setstate__(self, state):
        pt, axis, radius, options = state
        self.__init__(pt, axis, radius, **options)


# noinspection PyProtectedMember,PyUnresolvedReferences,DuplicatedCode,PyTypeChecker
class Cone(Surface, _Cone):
    """Cone surface class.

    Parameters
    ----------
    apex : array_like[float]
        Cone's apex.
    axis : array_like[float]
        Cone's axis.
    t2 : float
        Square of tangent of angle between axis and generatrix.
    sheet : int
        Cone's sheet.
    options : dict
        Dictionary of surface's options. Possible values:
            transform = transformation to be applied to the cone being
                             created. Transformation instance.
    """

    def __init__(
        self,
        apex: npt.NDArray[float],
        axis: npt.NDArray[float],
        t2: float,
        sheet: int = 0,
        **options,
    ) -> None:
        tr: Transformation | None = options.pop("transform", None)
        if tr:
            apex = tr.apply2point(apex)
            axis = tr.apply2vector(axis)
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        maxdir = np.argmax(np.abs(axis))
        if axis[maxdir] < 0:
            axis *= -1
            sheet *= -1
        apex = np.asarray(apex, dtype=float)
        self._axis_digits = significant_array(
            axis, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE
        )
        self._apex_digits = significant_array(
            apex, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE
        )
        Surface.__init__(self, **options)
        _Cone.__init__(self, apex, axis, t2, sheet)
        self._t2_digits = significant_digits(self._t2, constants.FLOAT_TOLERANCE)

    def apply_transformation(self) -> Surface:
        tr = self.transformation
        if tr is None:
            return self
        apex = tr.apply2point(self._apex)
        axis = tr.apply2vector(self._axis)
        sheet = self._sheet
        options = self.clean_options()
        return Cone(apex, axis, self._t2, sheet, **options)

    def round(self) -> Surface:
        res = self.apply_transformation()
        apex = round_array(res._apex)
        axis = round_array(res._axis)
        t2 = round_scalar(self._t2)
        sheet = self._sheet
        options = self.clean_options()
        return Cone(apex, axis, t2, sheet, **options)

    def copy(self):
        t2 = self._t2
        instance = Cone.__new__(Cone, self._apex, self._axis, t2, self._sheet)
        instance._axis_digits = self._axis_digits
        instance._apex_digits = self._apex_digits
        instance._t2_digits = self._t2_digits
        Surface.__init__(instance, **self.options)
        _Cone.__init__(instance, self._apex, self._axis, t2, self._sheet)
        return instance

    def __repr__(self):
        return f"Cone({self._apex}, {self._axis}, {self._t2}, {self._sheet}, {self.options if self.options else ''})"

    def __getstate__(self):
        return self._apex, self._axis, self._t2, self._sheet, Surface.__getstate__(self)

    def __setstate__(self, state):
        apex, axis, t2, sheet, options = state
        self.__init__(apex, axis, t2, sheet, **options)

    def __hash__(self):
        result = hash(self._get_t2()) ^ hash(self._sheet)
        for c in self._get_apex():
            result ^= hash(c)
        for a in self._get_axis():
            result ^= hash(a)
        return result

    def __eq__(self, other):
        # noinspection DuplicatedCode
        if self is other:
            return True

        if not isinstance(other, Cone):
            return False

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
        cone = Cone(self._apex, self._axis, self._t2, sheet=0, transform=tr, **self.options)
        if self._sheet != 0:
            plane = Plane(self._axis, -np.dot(self._axis, self._apex), name=1, transform=tr)
            if self._sheet == +1:
                op = "C"
            else:
                op = "S"
            return mckit.Shape("U", cone, mckit.Shape(op, plane))
        return cone

    def mcnp_words(self, pretty: bool = False) -> list[str]:
        words = Surface.mcnp_words(self)
        if np.all(self._get_axis() == np.array([1.0, 0.0, 0.0])):
            if self._get_apex()[1] == 0.0 and self._get_apex()[2] == 0.0:
                words.append("KX")
                words.append(" ")
                v = self._apex[0]
                p = self._apex_digits[0]
                words.append(pretty_float(v, p))
            else:
                words.append("K/X")
                for v, p in zip(self._apex, self._apex_digits):
                    words.append(" ")
                    words.append(pretty_float(v, p))
        elif np.all(self._get_axis() == np.array([0.0, 1.0, 0.0])):
            if self._get_apex()[0] == 0.0 and self._get_apex()[2] == 0.0:
                words.append("KY")
                words.append(" ")
                v = self._apex[1]
                p = self._apex_digits[1]
                words.append(pretty_float(v, p))
            else:
                words.append("K/Y")
                for v, p in zip(self._apex, self._apex_digits):
                    words.append(" ")
                    words.append(pretty_float(v, p))
        elif np.all(self._get_axis() == np.array([0.0, 0.0, 1.0])):
            if self._get_apex()[0] == 0.0 and self._get_apex()[1] == 0.0:
                words.append("KZ")
                words.append(" ")
                v = self._apex[2]
                p = self._apex_digits[2]
                words.append(pretty_float(v, p))
            else:
                words.append("K/Z")
                for v, p in zip(self._apex, self._apex_digits):
                    words.append(" ")
                    words.append(pretty_float(v, p))
        else:
            nx, ny, nz = self._axis
            a = 1 + self._t2
            m = np.array(
                [
                    [1 - a * nx**2, -a * nx * ny, -a * nx * nz],
                    [-a * nx * ny, 1 - a * ny**2, -a * ny * nz],
                    [-a * nx * nz, -a * ny * nz, 1 - a * nz**2],
                ]
            )
            v = np.zeros(3)
            k = 0
            m, v, k = Transformation(translation=self._apex).apply2gq(m, v, k)
            return GQuadratic(m, v, k, **self.options).mcnp_repr()
        words.append(" ")
        v = self._t2
        p = self._t2_digits
        words.append(pretty_float(v, p))
        if self._sheet != 0:
            words.append(" ")
            words.append(f"{self._sheet:d}")
        return words

    def is_close_to(
        self,
        other: Surface,
        estimator: Callable[[Any, Any], bool] = DEFAULT_TOLERANCE_ESTIMATOR,
    ) -> bool:
        if self is other:
            return True
        if not isinstance(other, Cone):
            return False
        return estimator(
            (self._apex, self._axis, self._t2, self.transformation),
            (other._apex, other._axis, other._t2, other.transformation),
        )


# noinspection PyProtectedMember,PyUnresolvedReferences,PyTypeChecker
class GQuadratic(Surface, _GQuadratic):
    """Generic quadratic surface class.

    Parameters
    ----------
    m : array_like[float]
        Matrix of coefficients of quadratic terms. m.shape=(3,3)
    v : array_like[float]
        Vector of coefficients of linear terms. v.shape=(3, )
    k : float
        Free term.
    options : dict
        Dictionary of surface's options. Possible values:
            transform = transformation to be applied to the surface being
                             created. Transformation instance.
    """

    def __init__(self, m, v, k, **options):
        tr: Transformation | None = options.pop("transform", None)
        if tr:
            m, v, k = tr.apply2gq(m, v, k)
        else:
            m = np.asarray(m, dtype=float)
            v = np.asarray(v, dtype=float)
        eigenvalues = np.linalg.eigvalsh(m)
        factor = 1.0 / np.max(np.abs(eigenvalues))
        self._m_digits = significant_array(
            m, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE
        )
        self._v_digits = significant_array(
            v, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE
        )
        self._k_digits = significant_digits(
            k, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE
        )
        Surface.__init__(self, **options)
        _GQuadratic.__init__(self, m, v, k, factor)

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

    def __eq__(self, other) -> bool:
        if not isinstance(other, GQuadratic):
            return False
        for x, y in zip(self._get_v(), other._get_v()):
            if x != y:
                return False
        for x, y in zip(self._get_m().ravel(), other._get_m().ravel()):
            if x != y:
                return False
        return self._get_k() == other._get_k()

    def _get_m(self) -> np.ndarray:
        return round_array(self._m, self._m_digits)

    def _get_v(self) -> np.ndarray:
        return round_array(self._v, self._v_digits)

    def _get_k(self) -> float:
        return round_scalar(self._k, self._k_digits)

    def __getstate__(self):
        return self._m, self._v, self._k, Surface.__getstate__(self)

    def __setstate__(self, state):
        m, v, k, options = state
        GQuadratic.__init__(self, m, v, k, **options)

    def transform(self, tr):
        return GQuadratic(self._m, self._v, self._k, transform=tr, **self.options)

    def mcnp_words(self, pretty: bool = False) -> list[str]:
        words = Surface.mcnp_words(self)
        words.append("GQ")
        m = self._get_m()
        a, b, c = np.diag(m)
        d = m[0, 1] + m[1, 0]
        e = m[1, 2] + m[2, 1]
        f = m[0, 2] + m[2, 0]
        g, h, j = self._get_v()
        k = self._get_k()
        for v in [a, b, c, d, e, f, g, h, j, k]:
            words.append(" ")
            p = significant_digits(v, constants.FLOAT_TOLERANCE, constants.FLOAT_TOLERANCE)
            words.append(pretty_float(v, p))
        return words

    def apply_transformation(self) -> Surface:
        tr = self.transformation
        if tr is None:
            return self
        m, v, k = tr.apply2gq(self._m, self._v, self._k)
        options = self.clean_options()
        return GQuadratic(m, v, k, **options)

    def round(self) -> Surface:
        temp: Surface = self.apply_transformation()
        m, v = map(round_array, [temp._m, temp._v])  # type: ignore[attr-defined]
        k = round_scalar(temp._k)
        # TODO dvp: handle cases when the surface can be represented with specialized quadratic surface: Cone etc.
        return GQuadratic(m, v, k, **self.clean_options())

    def __repr__(self):
        options = str(self.options) if self.options else ""
        return f"GQuadratic({self._m}, {self._v}, {self._k}, {self._factor}, {options})"

    def is_close_to(
        self,
        other: Surface,
        estimator: Callable[[Any, Any], bool] = DEFAULT_TOLERANCE_ESTIMATOR,
    ) -> bool:
        if self is other:
            return True
        if not isinstance(other, GQuadratic):
            return False  # TODO dvp: handle cases when other is specialized quadratic surface Sphere, Cone etc.
        return estimator(
            (self._k, self._v, self._m, self.transformation),
            (other._k, other._v, other._m, other.transformation),
        )


# noinspection PyProtectedMember,PyUnresolvedReferences,PyTypeChecker
class Torus(Surface, _Torus):
    """Tori surface class.

    Parameters
    ----------
    center : array_like[float]
        The center of torus.
    axis : array_like[float]
        The axis of torus.
    r : float
        Major radius.
    a : float
        Radius parallel to torus axis.
    b : float
        Radius perpendicular to torus axis.
    options : dict
        Dictionary of surface's options. Possible values:
            transform = transformation to be applied to the torus being
                             created. Transformation instance.
    """

    def __init__(self, center, axis, r, a, b, **options):
        tr: Transformation | None = options.pop("transform", None)
        if tr:
            center = tr.apply2point(center)
            axis = tr.apply2vector(axis)
        else:
            center = np.asarray(center, dtype=float)
            axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        maxdir = np.argmax(np.abs(axis))
        if axis[maxdir] < 0:
            axis *= -1
        self._axis_digits = significant_array(
            axis, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE
        )
        self._center_digits = significant_array(
            center, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE
        )
        self._R_digits = significant_digits(r, constants.FLOAT_TOLERANCE)
        self._a_digits = significant_digits(a, constants.FLOAT_TOLERANCE)
        self._b_digits = significant_digits(b, constants.FLOAT_TOLERANCE)
        Surface.__init__(self, **options)
        _Torus.__init__(self, center, axis, r, a, b)

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
        result = hash(self._get_r()) ^ hash(self._get_a()) ^ hash(self._get_b())
        for c in self._get_center():
            result ^= hash(c)
        for a in self._get_axis():
            result ^= hash(a)
        return result

    def __eq__(self, other):
        if not isinstance(other, Torus):
            return False
        for x, y in zip(self._get_center(), other._get_center()):
            if x != y:
                return False
        for x, y in zip(self._get_axis(), other._get_axis()):
            if x != y:
                return False
        return (
            self._get_r() == other._get_r()
            and self._get_a() == other._get_a()
            and self._get_b() == other._get_b()
        )

    def _get_axis(self):
        return round_array(self._axis, self._axis_digits)

    def _get_center(self):
        return round_array(self._center, self._center_digits)

    def _get_r(self):
        return round_scalar(self._R, self._R_digits)

    def _get_a(self):
        return round_scalar(self._a, self._a_digits)

    def _get_b(self):
        return round_scalar(self._b, self._b_digits)

    def __getstate__(self):
        return self._center, self._axis, self._R, self._a, self._b, Surface.__getstate__(self)

    def __setstate__(self, state):
        center, axis, R, a, b, options = state
        self.__init__(center, axis, R, a, b, **options)

    def transform(self, tr: Transformation) -> Torus:
        return Torus(
            self._center, self._axis, self._R, self._a, self._b, transform=tr, **self.options
        )

    def mcnp_words(self, pretty: bool = False) -> list[str]:
        words = Surface.mcnp_words(self)
        if np.all(self._get_axis() == np.array([1.0, 0.0, 0.0])):
            words.append("TX")
        elif np.all(self._get_axis() == np.array([0.0, 1.0, 0.0])):
            words.append("TY")
        elif np.all(self._get_axis() == np.array([0.0, 0.0, 1.0])):
            words.append("TZ")
        x, y, z = self._get_center()
        values = [x, y, z, self._get_r(), self._get_a(), self._get_b()]
        digits = [*self._center_digits, self._R_digits, self._a_digits, self._b_digits]
        for v, p in zip(values, digits):
            words.append(" ")
            words.append(pretty_float(v, p))
        return words

    def round(self) -> Surface:
        temp = self.apply_transformation()
        center, axis = map(round_array, [temp._center, temp._axis])  # type: ignore[attr-defined]

        def r(x):
            return round_scalar(x, significant_digits(x, FLOAT_TOLERANCE))

        r, a, b = map(r, [temp._R, temp._a, temp._b])  # type: ignore[attr-defined]
        options = temp.clean_options()
        return Torus(center, axis, r, a, b, **options)

    def apply_transformation(self) -> Surface:
        tr = self.transformation
        if tr is None:
            return self
        center = tr.apply2point(self._center)
        axis = tr.apply2vector(self._axis)
        # TODO dvp: should we check the transformation and result? The axis is to be along EX, EY, EZ.
        return Torus(
            center,
            axis,
            self._R,
            self._a,
            self._b,
            **self.clean_options(),
        )

    def is_close_to(
        self,
        other: Torus,
        estimator: Callable[[Any, Any], bool] = DEFAULT_TOLERANCE_ESTIMATOR,
    ) -> bool:
        if self is other:
            return True
        if not isinstance(other, Torus):
            return False
        return estimator(
            (self._center, self._axis, self._R, self._a, self._b, self.transformation),
            (
                other._center,
                other._axis,
                other._R,
                other._a,
                other._b,
                other.transformation,
            ),
        )

    def __repr__(self):
        return f"Torus({self._center}, {self._axis}, {self._R}, \
            {self._a}, {self._b}, {self.options if self.options else ''}"
