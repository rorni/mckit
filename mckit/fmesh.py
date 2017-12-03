# -*- coding: utf-8 -*-

import numpy as np
from itertools import product

from .transformation import Transformation


class Box:
    """Box object.

    Parameters
    ----------
    base : array_like[float]
        Base point of the box. It is one of its vertices. It has shape 3.
    ex, ey, ez : array_like[float]
        Basis vectors that give directions of box's edges. They must be
        orthogonal. For now it is user's responsibility to ensure the
        orthogonality. The length of basis vectors denote corresponding
        dimensions of the box.

    Methods
    -------
    bounds() - gets bounds in global coordinate system.
    corners() - gets coordinates of all corners in global CS.
    f_ieqcons(x, *arg) - gets constraint function of the Box.
    fprime_ieqcons(x, *arg) - gets derivatives of constraint function of the Box.
    random_points(n) - generates n random points inside the box.
    volume() - gets volume of the box.
    """
    def __init__(self, base, ex, ey, ez):
        self.base = np.array(base)
        self.ex = np.array(ex)
        self.ey = np.array(ey)
        self.ez = np.array(ez)
        self._tr = Transformation(translation=self.base,
                                  rotation=np.concatenate((ex, ey, ez)))
        self._scale = np.array([np.linalg.norm(self.ex),
                                np.linalg.norm(self.ey),
                                np.linalg.norm(self.ez)])

    def corners(self):
        """Gets coordinates of all corners in global coordinate system."""
        corners = np.zeros((8, 3))
        multipliers = product([0, 1], [0, 1], [0, 1])
        for i, (x, y, z) in enumerate(multipliers):
            corners[i, :] = self.base + x * self.ex + y * self.ey + z * self.ez
        return corners

    def bounds(self):
        """Gets bounds in global coordinate system."""
        corners = self.corners()
        bounds = [[lo, hi] for lo, hi in zip(np.amin(corners, axis=0),
                                             np.amax(corners, axis=0))]
        return bounds

    def random_points(self, n):
        """Generates n random points inside the box."""
        points = np.random.random((n, 3)) * self._scale
        return self._tr.apply2point(points)

    def volume(self):
        """Gets volume of the box."""
        return np.multiply.reduce(self._scale)

    def f_ieqcons(self):
        """Gets constraint function of the Box.

        Returns
        -------
        f(x, *args) : callable
            A function that denotes inequality constraints corresponding to the
            Box. This function returns array (6,) where each element >=0 if the
            point x is inside the Box.
        """
        n = np.vstack((self.ex, self.ey, self.ez, -self.ex, -self.ey, -self.ez))
        p1 = self.base.reshape((1, 3))
        p2 = (self.base + self.ex + self.ey + self.ez).reshape((1, 3))
        p0 = np.vstack((np.repeat(p1, 3, axis=0), np.repeat(p2, 3, axis=0)))
        p = np.sum(np.multiply(n, p0), axis=1)
        #print(n)
        #print(p0)
        #print(p)
        return lambda x, *args: np.dot(n, x) - p

    def fprime_ieqcons(self):
        """Gets derivatives of constraint function of the Box.

        Returns
        -------
        fprime(x, *args) : callable
            A function that denotes partial derivatives of constraints
            corresponding to the Box. This function returns matrix (6, 3) -
            of partial derivatives of the Box's surface equations at point x.
        """
        n = np.vstack((self.ex, self.ey, self.ez, -self.ex, -self.ey, -self.ez))
        return lambda x, *args: n


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
