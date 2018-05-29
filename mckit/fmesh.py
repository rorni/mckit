# -*- coding: utf-8 -*-

from itertools import product
from abc import ABC, abstractmethod

import numpy as np

from .constants import GLOBAL_BOX, EX, EY, EZ
from .transformation import Transformation

from .geometry import Box


class AbstractMesh:
    pass


class RectMesh:
    """Represents rectangular mesh.

    Parameters
    ----------
    origin : array_like[float]
        Bottom left behind corner for the Rectangular Mesh.
    xbins, ybins, zbins : array_like[float]
        Bins of mesh in every direction. The last bin value gives dimension of
        mesh in this direction.
    transform : Trnasformation
        Transformation for the mesh. Default: None.

    Methods
    -------
    shape() - gets the shape of mesh.
    get_voxel(i, j, k) - gets the voxel of RectMesh with indices i, j, k.
    """
    def __init__(self, origin, xbins, ybins, zbins, transform=None):
        self._xbins = np.array(xbins)
        self._ybins = np.array(ybins)
        self._zbins = np.array(zbins)
        self._origin = np.array(origin)
        self._ex = EX
        self._ey = EY
        self._ez = EZ
        if transform is not None:
            self._ex = transform.apply2vector(self._ex)
            self._ey = transform.apply2vector(self._ey)
            self._ez = transform.apply2vector(self._ez)
            self._origin = transform.apply2point(self._origin)

    @property
    def shape(self):
        """Gets the shape of the mesh."""
        return self._xbins.size - 1, self._ybins.size - 1, self._zbins.size - 1

    def calculate_volumes(self, cells, verbose=False, min_volume=1.e-3):
        """Calculates volumes of cells"""
        pass

    def voxel_index(self, point):
        """Gets index of voxel that contains specified point.

        Parameters
        ----------
        point : array_like[float]
            Coordinates of the point to be checked.

        Returns
        -------
        i, j, k : int
            Indices along each dimension of voxel, where the point is located.
        """
        point = np.array(point)
        x_proj = np.dot(self._ex, point)
        y_proj = np.dot(self._ey, point)
        z_proj = np.dot(self._ez, point)
        i = np.searchsorted(self._xbins, x_proj) - 1
        j = np.searchsorted(self._ybins, y_proj) - 1
        k = np.searchsorted(self._zbins, z_proj) - 1
        if isinstance(i, int):
            return i, j, k
        else:
            indices = []
            for x, y, z



class CylMesh:
    """Represents cylindrical mesh.

    Parameters
    ----------
    origin : array_like[float]
        Bottom center of the cylinder, that bounds the mesh.
    axis : array_like[float]
        Cylinder's axis.
    vec : array_like[float]
        Vector defining along with axis the plane for theta=0.
    rbins, zbins, tbins: array_like[float]
        Bins of mesh in radial, extend and angle directions respectively.
        Angles are specified in revolutions.
    """
    def __init__(self, origin, axis, vec, rbins, zbins, tbins):
        self._origin = np.array(origin)
        self._axis = np.array(axis)
        self._vec = np.array(vec)
        self._voxels = {}


class FMesh:
    '''Fmesh tally object.
    
    Parameters
    ----------
    xbins, ybins, zbins : arraylike[float]
        Bin boundaries in X, Y and Z directions respectively.
    ebins : arraylike[float]
        Bin boundaries for energies.
    data : arraylike[float]
        Fmesh data - quantity estimation averaged over voxel volume. It has
        shape (Ne-1)x(Nx-1)x(Ny-1)x(Nz-1), where Ne, Nx, Ny and Nx - the number
        of corresponding bin boundaries.
    error : arraylike[float]
        Fmesh relative error. Shape - see data.
    particle : str or list
        Particle type for which data was calculated.
    transform : Transformation
        Transformation to be applied to the spatial mesh.
    multiplier : None
        Data transformation.
        
    Methods
    -------
    get_slice() - gets specific slice of data.
    get_spectrum(point) - gets energy spectrum at the specified point. 
    '''
    def __init__(self, xbins, ybins, zbins, ebins, data, error=None,
                 particle='N', transform=None, multiplier=None):
        self.xbins = np.array(xbins)
        self.ybins = np.array(ybins)
        self.zbins = np.array(zbins)
        self.ebins = np.array(ebins)
        self.data = np.array(data)
        self.error = np.array(error)
        self.particle = particle
        self.transform = transform
        self.multiplier = multiplier

    def get_spectrum(self, point):
        '''Gets energy spectrum at the specified point.
        
        Parameters
        ----------
        point : arraylike[float]
            Point energy spectrum must be get at.
            
        Returns
        -------
        en, flux: arraylike[float]
            energy flux and points, where it is defined.
        '''
        pass
