# -*- coding: utf-8 -*-
"""Represents transformations."""

import numpy as np


__all__ = ['Transformation']


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
    reverse()
        Reverses this transformation, and returns the result.
    """
    def __init__(self):
        # TODO: Implement transformation creation.
        pass

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
        # TODO: Implement gq transformation method
        raise NotImplementedError

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
            A vector of size 3 which defines vetor, normal to the plane surface
            in the main coordinate system.
        k : float
            Free term of plane surface equation in the main coordinate system.
        """
        # TODO: Implement plane transformation method
        raise NotImplementedError

    def apply2point(self, p1):
        """Gets coordinates of point p1 in the main coordinate system.

        Parameters
        ----------
        p1 : array_like[float]
            Coordinates of the point(s) in the auxiliary coordinate system.

        Returns
        -------
        p : numpy.ndarray
            Coordinates of the point in the main coordinate system.
        """
        # TODO: Implement point transformation method
        raise NotImplementedError

    def apply2vector(self, v1):
        """Gets coordinates of vector v1 in the main coordinate system.

        Parameters
        ----------
        v1 : array_like[float]
            Coordinates of the vector in the auxiliary coordinate system.

        Returns
        -------
        v : numpy.ndarray
            Coordinates of the vector in the main coordinate system.
        """
        # TODO: Implement vector transformation method
        raise NotImplementedError

    def reverse(self):
        """Reverses this transformation.

        Gets new transformation which is complement to this one.

        Returns
        -------
        tr : Transform
            Reversed version of this transformation.
        """
        # TODO: Implement reverse transformation method.
        raise NotImplementedError

