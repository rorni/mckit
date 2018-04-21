# -*- coding: utf-8 -*-

from itertools import product

import numpy as np

#from .constants import GLOBAL_BOX, EX, EY, EZ
from .transformation import Transformation

from .geometry import Box


class RectMesh(Box):
    """Represents rectangular mesh.

    Parameters
    ----------
    base : array_like[float]
        Base point of the voxel. It is one of its vertices. It has shape 3.
    ex, ey, ez : array_like[float]
        Basis vectors that give directions of box's edges. They must be
        orthogonal. For now it is user's responsibility to ensure the
        orthogonality. Their length has no affect on the dimensions of Mesh.
        Last value in xbins, ybins, zbins has.
    xbins, ybins, zbins : array_like[float]
        Bins of mesh in every direction. The last bin value gives dimension of
        mesh in this direction.

    Methods
    -------
    shape() - gets the shape of mesh.
    get_voxel(i, j, k) - gets the voxel of RectMesh with indices i, j, k.
    """
    def __init__(self, base, ex, ey, ez, xbins, ybins, zbins):
        ex = np.array(ex) / np.linalg.norm(ex) * xbins[-1]
        ey = np.array(ey) / np.linalg.norm(ey) * ybins[-1]
        ez = np.array(ez) / np.linalg.norm(ez) * zbins[-1]
        Box.__init__(base, ex, ey, ez)
        self.xbins = np.array(xbins)
        self.ybins = np.array(ybins)
        self.zbins = np.array(zbins)

    def shape(self):
        """Gets the shape of the mesh."""
        return self.xbins.size - 1, self.ybins.size - 1, self.zbins.size - 1

    def get_voxel(self, i, j, k):
        """Gets box that corresponds to voxel with indices i, j, k.

        Parameters
        ----------
        i, j, k : int
            Indices of the voxel.

        Returns
        -------
        voxel : Box
            The voxel.
        """
        base = np.array([self.xbins[i], self.ybins[j], self.zbins[k]])
        ex = self.ex * (self.xbins[i + 1] - self.xbins[i]) / self.xbins[-1]
        ey = self.ey * (self.ybins[j + 1] - self.ybins[j]) / self.ybins[-1]
        ez = self.ez * (self.zbins[k + 1] - self.zbins[k]) / self.zbins[-1]
        return Box(base, ex, ey, ez)


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
