# -*- coding: utf-8 -*-

from itertools import product

import numpy as np

from .geometry import EX, EY, EZ, Box
from .transformation import Transformation


class AbstractMesh:
    pass


class RectMesh:
    """Represents rectangular mesh.

    Parameters
    ----------
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
    def __init__(self, xbins, ybins, zbins, transform=None):
        self._xbins = np.array(xbins)
        self._ybins = np.array(ybins)
        self._zbins = np.array(zbins)
        self._ex = EX
        self._ey = EY
        self._ez = EZ
        self._origin = np.array([self._xbins[0], self._ybins[0], self._zbins[0]])
        self._tr = None
        if transform is not None:
            self.transform(transform)

    def __eq__(self, other):
        return self is other

    @property
    def shape(self):
        """Gets the shape of the mesh."""
        return self._xbins.size - 1, self._ybins.size - 1, self._zbins.size - 1

    def transform(self, tr):
        """Transforms this mesh.

        Parameters
        ----------
        tr : Transformation
            Transformation to be applied.
        """
        self._origin = tr.apply2point(self._origin)
        self._ex = tr.apply2vector(self._ex)
        self._ey = tr.apply2vector(self._ey)
        self._ez = tr.apply2vector(self._ez)
        if self._tr is not None:
            self._tr = tr.apply2transform(self._tr)
        else:
            self._tr = tr

    def calculate_volumes(self, cells, with_mat_only=True, verbose=False, min_volume=1.e-3):
        """Calculates volumes of cells.

        Parameters
        ----------
        cells : list[Body]
            List of cells.
        verbose : bool
            Verbose output during calculations.
        min_volume : float
            Minimum volume for cell volume calculations

        Returns
        -------
        volumes : dict
            Volumes of cells for every voxel. It is dictionary cell -> vol_matrix.
            vol_matrix is SparseData instance - volume of cell for each voxel.
        """
        volumes = {}
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    box = self.get_voxel(i, j, k)
                    for c in cells:
                        if with_mat_only and c.material() is None:
                            continue
                        vol = c.shape.volume(box=box, min_volume=min_volume)
                        if vol > 0:
                            if c not in volumes.keys():
                                volumes[c] = SparseData(self.shape)
                            volumes[c][i, j, k] = vol
                    if verbose and volumes[c][i, j, k]:
                        print('Voxel ({0}, {1}, {2})'.format(i, j, k))
        return volumes

    def get_voxel(self, i, j, k):
        """Gets voxel.

        Parameters
        ----------
        i, j, k : int
            Indices of the voxel.

        Returns
        -------
        voxel : Box
            The box that describes the voxel.
        """
        cx = 0.5 * (self._xbins[i] + self._xbins[i+1])
        cy = 0.5 * (self._ybins[j] + self._ybins[j+1])
        cz = 0.5 * (self._zbins[k] + self._zbins[k+1])
        center = np.array([cx, cy, cz])
        if self._tr:
            center = self._tr.apply2point(center)
        xdim = self._xbins[i+1] - self._xbins[i]
        ydim = self._ybins[j+1] - self._ybins[j]
        zdim = self._zbins[k+1] - self._zbins[k]
        return Box(center, xdim, ydim, zdim, ex=self._ex, ey=self._ey, ez=self._ez)

    def voxel_index(self, point, local=False):
        """Gets index of voxel that contains specified point.

        Parameters
        ----------
        point : array_like[float]
            Coordinates of the point to be checked.
        local : bool
            If point is specified in local coordinate system.

        Returns
        -------
        i, j, k : int
            Indices along each dimension of voxel, where the point is located.
        """
        if self._tr and not local:
            point = self._tr.reverse().apply2point(point)
        else:
            point = np.array(point)
        x_proj = np.dot(point, EX)
        y_proj = np.dot(point, EY)
        z_proj = np.dot(point, EZ)
        i = np.searchsorted(self._xbins, x_proj) - 1
        j = np.searchsorted(self._ybins, y_proj) - 1
        k = np.searchsorted(self._zbins, z_proj) - 1
        if len(point.shape) == 1:
            return self.check_indices(i, j, k)
        else:
            print('i=', i, 'j=', j, 'k=', k)
            indices = []
            for x, y, z in zip(i, j, k):
                indices.append(self.check_indices(x, y, z))
            return indices

    def check_indices(self, i, j, k):
        """Check if the voxel with such indices really exists.

        Parameters
        ----------
        i, j, k : int
            Indices along x, y and z dimensions.

        Returns
        -------
        index_tuple : tuple(int)
            A tuple of indices if such voxel exists. None otherwise.
        """
        i = self._check_x(i)
        j = self._check_y(j)
        k = self._check_z(k)
        if i is None or j is None or k is None:
            return None
        else:
            return i, j, k

    def _check_x(self, i):
        if i < 0 or i >= self._xbins.size - 1:
            return None
        return i

    def _check_y(self, j):
        if j < 0 or j >= self._ybins.size - 1:
            return None
        return j

    def _check_z(self, k):
        if k < 0 or k >= self._zbins.size - 1:
            return None
        return k

    def slice_axis_index(self, X=None, Y=None, Z=None):
        """Gets index and axis of slice.

        Parameters
        ----------
        X, Y, Z : float
            Point of slice in local coordinate system.

        Returns
        -------
        axis : int
            Number of axis.
        index : int
            Index along axis.
        x, y : ndarray[float]
            Centers of bins along free axes.
        """
        none = 0
        for i, a in enumerate([X, Y, Z]):
            if a is not None:
                none += 1
                axis = i
        if none != 1:
            raise ValueError('Wrong number of fixed spatial variables.')

        if X is not None:
            index = self._check_x(np.searchsorted(self._xbins, X) - 1)
        elif Y is not None:
            index = self._check_y(np.searchsorted(self._ybins, Y) - 1)
        elif Z is not None:
            index = self._check_z(np.searchsorted(self._zbins, Z) - 1)
        else:
            index = None

        if index is None:
            raise ValueError('Specified point lies outside of the mesh.')

        if axis > 0:
            x = 0.5 * (self._xbins[1:] + self._xbins[:-1])
        else:
            x = 0.5 * (self._ybins[1:] + self._ybins[:-1])
        if axis < 2:
            y = 0.5 * (self._zbins[1:] + self._zbins[:-1])
        else:
            y = 0.5 * (self._ybins[1:] + self._ybins[:-1])

        return axis, index, x, y


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
        self._rbins = np.array(rbins)
        self._zbins = np.array(zbins)
        self._tbins = np.array(tbins)
        self._voxels = {}
        # raise NotImplementedError

    @property
    def shape(self):
        """Gets the shape of the mesh."""
        return self._rbins.size - 1, self._zbins.size - 1, self._tbins.size - 1

    def transform(self, tr):
        """Transforms this mesh.

        Parameters
        ----------
        tr : Transformation
            Transformation to be applied.
        """
        raise NotImplementedError

    def calculate_volumes(self, cells, with_mat_only=True, verbose=False, min_volume=1.e-3):
        """Calculates volumes of cells.

        Parameters
        ----------
        cells : list[Body]
            List of cells.
        verbose : bool
            Verbose output during calculations.
        min_volume : float
            Minimum volume for cell volume calculations

        Returns
        -------
        volumes : dict
            Volumes of cells for every voxel. It is dictionary cell -> vol_matrix.
            vol_matrix is SparseData instance - volume of cell for each voxel.
        """
        raise NotImplementedError

    def get_voxel(self, i, j, k):
        """Gets voxel.

        Parameters
        ----------
        i, j, k : int
            Indices of the voxel.

        Returns
        -------
        voxel : Box
            The box that describes the voxel.
        """
        raise NotImplementedError

    def voxel_index(self, point, local=False):
        """Gets index of voxel that contains specified point.

        Parameters
        ----------
        point : array_like[float]
            Coordinates of the point to be checked.
        local : bool
            If point is specified in local coordinate system.

        Returns
        -------
        i, j, k : int
            Indices along each dimension of voxel, where the point is located.
        """
        raise NotImplementedError

    def check_indices(self, i, j, k):
        """Check if the voxel with such indices really exists.

        Parameters
        ----------
        i, j, k : int
            Indices along x, y and z dimensions.

        Returns
        -------
        index_tuple : tuple(int)
            A tuple of indices if such voxel exists. None otherwise.
        """
        i = self._check_r(i)
        j = self._check_z(j)
        k = self._check_t(k)
        if i is None or j is None or k is None:
            return None
        else:
            return i, j, k

    def _check_r(self, i):
        if i < 0 or i >= self._rbins.size - 1:
            return None
        return i

    def _check_z(self, j):
        if j < 0 or j >= self._zbins.size - 1:
            return None
        return j

    def _check_t(self, k):
        if k < 0 or k >= self._tbins.size - 1:
            return None
        return k

    def slice_axis_index(self, R=None, Z=None, T=None):
        """Gets index and axis of slice.

        Parameters
        ----------
        R, Z, T : float
            Point of slice in local coordinate system.

        Returns
        -------
        axis : int
            Number of axis.
        index : int
            Index along axis.
        x, y : ndarray[float]
            Centers of bins along free axes.
        """
        raise NotImplementedError


class SparseData:
    """Describes sparse spatial data.

    Parameters
    ----------
    shape : tuple
        Shape of the data.

    Properties
    ----------
    shape : tuple[int]
        Mesh shape.
    size : int
        The quantity of nonzero elements.

    Methods
    -------
    copy()
        Makes a copy of the data.
    dense()
        Converts SparseData object to dense matrix (numpy array).
    total()
        Finds a sum of all data elements.
    """
    def __init__(self, shape):
        self._shape = shape
        self._data = {}

    def __setitem__(self, index, value):
        """Adds new data element by its index."""
        if len(index) != len(self._shape):
            raise ValueError("Wrong shape.")
        for i, b in zip(index, self._shape):
            if i < 0 or i >= b:
                raise IndexError('Index value is out of range')
        if value == 0:
            self._data.pop(index, None)
        else:
            self._data[index] = value

    def __getitem__(self, index):
        """Gets data value with specified indices."""
        if len(index) != len(self._shape):
            raise ValueError("Wrong shape.")
        for i, b in zip(index, self._shape):
            if i < 0 or i >= b:
                raise IndexError('Index value is out of range')
        return self._data.get(index, 0)

    def __iter__(self):
        return iter(self._data.items())

    def copy(self):
        """Makes a copy of the data."""
        result = SparseData(self._shape)
        result._data = self._data.copy()
        return result

    def dense(self):
        """Converts SparseData object to dense matrix (numpy array)."""
        result = np.zeros(self._shape)
        for index, value in self._data.items():
            result[index] = value
        return result

    def total(self):
        """Finds a sum of all data elements."""
        result = 0.0
        for v in self._data.values():
            result += np.sum(v)
        return result

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return len(self._data.keys())

    def __add__(self, other):
        result = self.copy()
        result += other
        return result

    def __mul__(self, other):
        result = self.copy()
        result *= other
        return result

    def __truediv__(self, other):
        result = self.copy()
        result /= other
        return result

    def __sub__(self, other):
        result = self.copy()
        result -= other
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other).__imul__(-1)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __iadd__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            for index in product(*map(range, self._shape)):
                self[index] += other
        elif not isinstance(other, SparseData):
            raise TypeError('Unsupported operand type')
        elif self._shape != other.shape:
            raise ValueError('Operands have inconsistent shapes')
        else:
            for index, value in other:
                self[index] += value
        return self

    def __imul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            for index in self._data.keys():
                self[index] *= other
        elif isinstance(other, np.ndarray):
            if self.shape != other.shape:
                raise ValueError("Inconsistent shape")
            for index, value in self:
                self[index] *= other[index]
        elif not isinstance(other, SparseData):
            raise TypeError('Unsupported operand type')
        elif self._shape != other.shape:
            raise ValueError('These data belong to different meshes.')
        else:
            for index, value in other:
                self[index] *= value
            del_indices = set(self._data.keys()).difference(set(other._data.keys()))
            for index in del_indices:
                self._data.pop(index)
        return self

    def __isub__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            for index in product(*map(range, self._shape)):
                self[index] -= other
        elif not isinstance(other, SparseData):
            raise TypeError('Unsupported operand type')
        elif self._shape != other.shape:
            raise ValueError('These data belong to different meshes.')
        else:
            for index, value in other:
                self[index] -= value
        return self

    def __itruediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            for index, value in self:
                self[index] = value / other
        elif not isinstance(other, SparseData):
            raise TypeError('Unsupported operand type')
        elif self._shape != other.shape:
            raise ValueError('These data belong to different meshes.')
        else:
            for index, value in other:
                self[index] /= value
        return self


class FMesh:
    """Fmesh tally object.
    
    Parameters
    ----------
    name : int
        Tally name.
    particle : str
        Particle type (neutron, photon, electron).
    ebins : array_like[float]
        Bin boundaries for energies.
    xbins, ybins, zbins : array_like[float]
        Bin boundaries in X, Y and Z directions respectively for rectangular mesh.
    rbins, zbins, tbins : array_like[float]
        Bin boundaries in R, Z and Theta directions for cylindrical mesh.
    origin : array_like[float]
        Bottom center of the cylinder (For cylindrical mesh only).
    axis : array_like[float]
        Axis of the cylinder (for cylindrical mesh only).
    vec : array_like[float]
        Vector defining along with axis the plane for theta=0.
    data : arraylike[float]
        Fmesh data - quantity estimation averaged over voxel volume. It has
        shape (Ne-1)x(Nx-1)x(Ny-1)x(Nz-1), where Ne, Nx, Ny and Nx - the number
        of corresponding bin boundaries.
    error : arraylike[float]
        Fmesh relative error. Shape - see data.
    transform : Transformation
        Transformation to be applied to the spatial mesh.
    histories : int
        The number of histories run to obtain meshtal data.
    modifier : None
        Data transformation.
        
    Methods
    -------
    get_slice()
        Gets specific slice of data.
    get_spectrum(point)
        Gets energy spectrum at the specified point.
    get_spectrum_by_index(index)
        Gets energy spectrum at the specified mesh index.
    mean_flux()
        Gets average flux for every energy bin.
    """
    def __init__(self, name, particle, data, error, ebins=None, xbins=None, ybins=None, zbins=None, rbins=None,
                 tbins=None, transform=None, modifier=None, origin=None, axis=None, vec=None, histories=None):
        self._data = np.array(data)
        self._error = np.array(error)
        self._name = name
        self._histories = histories
        self._particle = particle
        if ebins is not None:
            self._ebins = np.array(ebins)
        else:
            self._ebins = np.array([0, 1.e+36])
        self._modifier = modifier
        if rbins is None and tbins is None:
            self._mesh = RectMesh(xbins, ybins, zbins, transform=transform)
        elif xbins is None and ybins is None:
            self._mesh = CylMesh(origin, axis, vec, rbins, zbins, tbins)
        if self._data.shape[1:] != self._mesh.shape:
            raise ValueError("Incorrect data shape")
        elif self._error.shape[1:] != self._mesh.shape:
            raise ValueError("Incorrect error shape")

    @property
    def mesh(self):
        return self._mesh

    @property
    def particle(self):
        return self._particle

    @property
    def histories(self):
        """The number of histories in the run."""
        return self._histories

    def mean_flux(self):
        """Gets average flux.

        Returns
        -------
        ebins : np.array[float]
            Energy bin boundaries.
        flux : np.array[float]
            Average flux in each energy bin.
        """
        return self._ebins.copy(), np.mean(self._data, axis=(1, 2, 3))

    def get_spectrum(self, point):
        """Gets energy spectrum at the specified point.
        
        Parameters
        ----------
        point : arraylike[float]
            Point energy spectrum must be get at.
            
        Returns
        -------
        energies: ndarray[float]
            Energy bins for the spectrum at the point - group boundaries.
        flux : ndarray[float]
            Group flux at the point.
        err : ndarray[float]
            Relative errors for flux components.
        """
        index = self._mesh.voxel_index(point)
        if index is None:
            raise ValueError("Point {0} lies outside of the mesh.".format(point))
        return self.get_spectrum_by_index(index)

    def get_spectrum_by_index(self, index):
        """Gets energy spectrum in the specified voxel.

        Parameters
        ----------
        index : tuple[int]
            Indices of spatial mesh bins.

        Returns
        -------
        energies: ndarray[float]
            Energy bins for the spectrum at the point - group boundaries.
        flux : ndarray[float]
            Group flux at the point.
        err : ndarray[float]
            Relative errors for flux components.
        """
        i, j, k = index
        flux = self._data[:, i, j, k]
        err = self._error[:, i, j, k]
        return self._ebins, flux, err

    def get_slice(self, E='total', X=None, Y=None, Z=None, R=None, T=None):
        """Gets data in the specified slice. Only one spatial letter is allowed.

        Parameters
        ----------
        E : str or float
            Energy value of interest. It specifies energy bin. If 'total' - then data
            is summed across energy axis.
        X, Y, Z : float
            Spatial point which belongs to the slice plane. Other two dimensions are free.

        Returns
        -------
        x, y : ndarray[float]
            Centers of spatial bins in free directions.
        data : ndarray[float]
            Data
        err : ndarray[float]
            Relative errors for data.
        """
        if isinstance(self._mesh, RectMesh):
            axis, index, x, y = self._mesh.slice_axis_index(X=X, Y=Y, Z=Z)
        else:
            axis, index, x, y = self._mesh.slice_axis_index(R=R, Z=Z, T=T)

        data = self._data.take(index, axis=axis + 1)   # +1 because the first axis is for energy.
        err = self._error.take(index, axis=axis + 1)

        if E == 'total':
            abs_err = (data * err) ** 2
            abs_tot_err = np.sqrt(np.sum(abs_err, axis=0))
            data = np.sum(data, axis=0)
            err = np.nan_to_num(abs_tot_err / data)
        else:
            if E <= self._ebins[0] or E > self._ebins[-1]:
                raise ValueError("Specified energy lies outside of energy bins.")
            i = np.searchsorted(self._ebins, E) - 1
            data = data.take(i, axis=0)
            err = err.take(i, axis=0)
        return x, y, data, err

