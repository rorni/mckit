# -*- coding: utf-8 -*-
"""Represents transformations."""
from typing import Any

import numpy as np

from .card import Card

# noinspection PyUnresolvedReferences,PyPackageRequirements
from .geometry import ORIGIN
from .utils import make_hash
from .utils.tolerance import EstimatorType, MaybeClose, tolerance_estimator

__all__ = ["Transformation", "IDENTITY_ROTATION"]

IDENTITY_ROTATION = np.eye(3)

ANGLE_TOLERANCE = 0.001
COS_TH = np.sin(ANGLE_TOLERANCE)
ZERO_COS_TOLERANCE = 2.0e-16


class Transformation(Card, MaybeClose):
    """Geometry transformation object.

    Parameters
    ----------
    translation : array_like[float]
        Translation vector of the transformation. Default: zero vector - no
        translation.
    rotation : array_like[float]
        Represents rotation matrix of the transformation. It can be either
        matrix itself with shape 3x3 or a list of transformation parameters.
        In the last case the parameters are treated as follows: 1-3 - ex', 4-6 -
        ey', 7-9 - ez', where ex', ey' and ez' - basis vectors of the local
        coordinate system. Basis vectors should be pairwise orthogonal. If there
        is a small non-orthogonality - less than 0.001 radian - vectors are
        corrected. Default: identity matrix - non rotation.
    indegrees : bool
        How rotation parameters should be treated. If True - rotation matrix
        parameters are given in degrees. Otherwise rotation parameters are basis
        vectors of local coordinate system. Default: False.
    inverted : bool
        How translation vector should be interpreted. If True - it is the origin
        of local coordinate system defined in the global one. Otherwise - the
        origin of global coordinate system defined in the local one.
        Default: False
    options : dict
        Other options, like name, comment, etc.

    Methods
    -------
    apply2gq(m, v, k)
        Gets parameters of generic quadratic surface in the main coordinate
        system.
    apply2plane(v, k)
        Gets parameters of plane in the main coordinate system.
    apply2point(p)
        Gets coordinates of point p in the global coordinate system.
    apply2vector(v)
        Gets coordinates of vector v in the global coordinate system.
    apply2transform(tr)
        Gets resulting transformation.
    reverse()
        Reverses this transformation, and returns the result.
    """

    def __init__(
        self,
        translation=ORIGIN,
        rotation=None,
        indegrees=False,
        inverted=False,
        **options: Any,
    ):

        Card.__init__(self, **options)

        if translation is not ORIGIN:
            translation = np.asarray(translation, dtype=float)

        if translation.shape != (3,):
            raise ValueError(
                f"Transaction #{self.name()}: wrong length of translation vector."
            )

        if rotation is None:
            u = IDENTITY_ROTATION
        else:
            u = np.asarray(rotation, dtype=float)
            if indegrees:
                u = np.cos(np.multiply(u, np.pi / 180.0))
            zero_cosines_idx = np.abs(u) < ZERO_COS_TOLERANCE
            u[zero_cosines_idx] = 0.0
            # TODO: Implement creation from reduced rotation parameter set.
            if u.shape == (9,):
                u = u.reshape((3, 3), order="F")
            if u.shape != (3, 3):
                raise ValueError(
                    f'Transaction{"" if self.is_anonymous else " #" + str(self.name())}: \
                      wrong number of rotation parameters: {u}.'
                )
            if np.array_equal(u, IDENTITY_ROTATION):
                u = IDENTITY_ROTATION
            else:
                # normalize auxiliary CS basis and orthogonalize it.
                u, r = np.linalg.qr(u)
                # QR decomposition returns orthogonal matrix u - which is corrected
                # rotation matrix, and upper triangular matrix r. On the main
                # diagonal r contains lengths of corresponding (negative if the
                # corrected vector is directed opposite to the initial one) input
                # basis vectors. Other elements are cosines of angles between
                # different basis vectors.

                # cos(pi/2 - ANGLE_TOLERANCE) = sin(ANGLE_TOLERANCE) - maximum
                # value of cosine of angle between two basis vectors.
                if (
                    abs(r[0, 1]) > COS_TH
                    or abs(r[0, 2]) > COS_TH
                    or abs(r[1, 2]) > COS_TH
                ):
                    raise ValueError(
                        f"Transaction #{self.name()}: non-orthogonality is greater than 0.001 rad."
                    )
                # To preserve directions of corrected basis vectors.
                for i in range(3):
                    u[:, i] = u[:, i] * np.sign(r[i, i])
        self._u = u
        if inverted:
            if u is IDENTITY_ROTATION:
                self._t = -translation
            else:
                self._t = -np.dot(u, translation)
        else:
            self._t = translation.copy()
        # self._t = -np.dot(u, translation) if inverted  else translation.copy()
        self._hash = make_hash(self._t, self._u)

    def mcnp_words(self, pretty=False):
        name = self.name()
        if name is None:
            name = 0
        words = ["*", f"TR{name}"]
        words.extend(self.get_words(pretty))
        return words

    def get_words(self, pretty=False):
        words = []
        for v in self._t:
            words.append(" ")
            words.append(
                "{:.10g}".format(v)
            )  # TODO dvp: check if precision 13 is necessary
            # add_float(words, v, pretty)
        if self._u is not IDENTITY_ROTATION:
            for v in self._u.transpose().ravel():
                words.append(" ")
                words.append("{:.10g}".format(np.arccos(v) * 180 / np.pi))
        return words

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Transformation):
            return False
        if np.array_equal(self._u, other._u):
            if np.array_equal(self._t, other._t):
                return True
        return False

    def __getitem__(self, key):
        return self.options.get(key, None)

    def __setitem__(self, key, value):
        self.options[key] = value

    def apply2gq(self, m1, v1, k1):
        """Gets parameters of generic quadratic surface in the main CS.

        Parameters
        ----------
        m1 : array_like[float]
            A 3x3 matrix which defines coefficients of quadratic terms of
            generic quadratic surface equation in the auxiliary coordinate
            system.
        v1 : array_like[float]
            A vector of size 3, which defines coefficients of linear terms of
            generic quadratic equation in the auxiliary coordinate system.
        k1 : float
            Free term of generic quadratic equation in the auxiliary coordinate
            system.

        Returns
        -------
        m : numpy.ndarray
            A 3x3 matrix which defines quadratic coefficients of GQ surface
            equation in the main coordinate system.
        v : numpy.ndarray
            A vector of size 3 which defines linear coefficients of GQ surface
            equation in the main coordinate system.
        k : float
            Free term of GQ surface equation in the main coordinate system.
        """
        m = np.dot(np.dot(self._u, m1), np.transpose(self._u))
        v = np.dot(self._u, v1) - 2 * np.dot(m, self._t)
        k = k1 - np.dot(v, self._t) - np.dot(self._t, np.dot(m, self._t))
        return m, v, k

    def apply2plane(self, v1, k1):
        """Gets parameters of plane surface in the main coordinate system.

        Parameters
        ----------
        v1 : array_like
            A vector of size 3 which defines vector, normal to the plane in
            the auxiliary coordinate system.
        k1 : float
            Free term of plane equation in the auxiliary coordinate
            system.

        Returns
        -------
        v : numpy.ndarray
            A vector of size 3 which defines vector, normal to the plane surface
            in the main coordinate system.
        k : float
            Free term of plane surface equation in the main coordinate system.
        """
        v = np.dot(self._u, v1)
        k = k1 - np.dot(v, self._t)
        return v, k

    def apply2point(self, p1):
        """Gets coordinates of point p1 in the main coordinate system.

        Parameters
        ----------
        p1 : array_like[float]
            Coordinates of the point(s) in the auxiliary coordinate system.
            It has shape (3,) if there is the only point or (N, 3) - if there
            are N points.

        Returns
        -------
        p : numpy.ndarray
            Coordinates of the point(s) in the main coordinate system.
        """
        # Matrix U is transposed to change U p1 -> p1 U^T - to preserve shape
        # of p1 and p.
        return np.dot(p1, np.transpose(self._u)) + self._t

    def apply2vector(self, v1):
        """Gets coordinates of vector v1 in the main coordinate system.

        Parameters
        ----------
        v1 : array_like[float]
            Coordinates of the vector(s) in the auxiliary coordinate system.

        Returns
        -------
        v : numpy.ndarray
            Coordinates of the vector(s) in the main coordinate system.
        """
        # In contrast with apply2point - no translation is needed.
        return np.dot(v1, np.transpose(self._u))

    def apply2transform(self, tr):
        """Gets new transformation.

        Suppose there are three coordinate systems r0, r1, r2. Transformation
        tr: r2 -> r1; and this transformation: r1 -> r. Thus the resulting
        transformation: r2 -> r. In other words the result is a sequence of
        two transformations: tr and this. tr is applied first.

        Parameters
        ----------
        tr : Transformation
            Transformation to be modified.

        Returns
        -------
        new_tr : Transformation
            New transformation - the result.
        """
        rot = np.dot(self._u, tr._u)
        trans = self.apply2point(tr._t)
        return Transformation(translation=trans, rotation=rot)

    def reverse(self):
        """Reverses this transformation.

        Gets new transformation which is complement to this one.

        Returns
        -------
        tr : Transform
            Reversed version of this transformation.
        """
        u1 = np.transpose(self._u)
        t1 = -np.dot(u1, self._t)
        return Transformation(translation=t1, rotation=u1)

    def is_close_to(
        self, other: Any, estimator: EstimatorType = tolerance_estimator()
    ) -> bool:
        if self is other:
            return True
        if not isinstance(other, Transformation):
            return False
        return estimator((self._t, self._u), (other._t, other._u))

    def __repr__(self):
        return f"Transformation(translation={self._t}, rotation={self._u})"
