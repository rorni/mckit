# -*- coding: utf-8 -*-

import numpy as np

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
