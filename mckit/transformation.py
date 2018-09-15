# -*- coding: utf-8 -*-
"""Represents transformations."""

import numpy as np

from .geometry import ORIGIN
from .printer import print_card

__all__ = ['Transformation', 'IDENTITY_ROTATION']

IDENTITY_ROTATION = np.eye(3)

ANGLE_TOLERANCE = 0.001


class Transformation:
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
        vectors of local coordinate system. Dafault: False.
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
    def __init__(self, translation=ORIGIN, rotation=None,
                 indegrees=False, inverted=False, **options):

        translation = np.array(translation, dtype=float)
        if translation.shape != (3,):
            raise ValueError('Wrong length of translation vector.')

        if rotation is None:
            u = IDENTITY_ROTATION
        else:
            u = np.array(rotation, dtype=float)
            if indegrees:
                u = np.cos(np.multiply(u, np.pi / 180.0))
            # TODO: Implement creation from reduced rotation parameter set.
            if u.shape == (9,):
                u = u.reshape((3, 3), order='F')
            if u.shape != (3, 3):
                raise ValueError('Wrong number of rotation parameters.')
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
            cos_th = np.sin(ANGLE_TOLERANCE)
            if abs(r[0, 1]) > cos_th or abs(r[0, 2]) > cos_th or \
               abs(r[1, 2]) > cos_th:
                raise ValueError('Non-orthogonality is greater than 0.001 rad.')
            # To preserve directions of corrected basis vectors.
            for i in range(3):
                u[:, i] = u[:, i] * np.sign(r[i, i])
        self._u = u
        self._t = -np.dot(u, translation) if inverted else translation.copy()
        self._options = options

    @staticmethod
    def _get_precision(u, t, box, tol):
        u1 = u.transpose()
        u = 180.0 / np.pi * np.arccos(u)
        t1 = t
        prec = np.finfo(float).precision
        while True:
            u2 = np.cos(np.pi * np.round(u, prec) / 180.0).transpose()
            t2 = np.round(t, prec)
            if np.linalg.norm(t2 - t1) >= tol:
                break
            diffs = [np.dot(u1 - u2, c) + np.dot(u2, t2) - np.dot(u1, t1)
                     for c in box.corners]
            if max(diffs) >= tol:
                break
        return prec + 1

    def _calculate_hash(self, u, t):
        self._hash = 0
        for v in t:
            self._hash ^= hash(v)
        for v in u.ravel():
            self._hash ^= hash(v)

    def __str__(self):
        return print_card(['*TR{0}'.format(self['name'])] + self.get_words())

    def get_words(self):
        words = []
        for v in self._t:
            words.append(' ')
            words.append('{0:.12e}'.format(v))
        for v in self._u.ravel():
            words.append(' ')
            words.append('{0:.12e}'.format(np.arccos(v) * 180 / np.pi))
        return words

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    def __getitem__(self, key):
        return self._options.get(key, None)

    def __setitem__(self, key, value):
        self._options[key] = value

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

