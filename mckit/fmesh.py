# -*- coding: utf-8 -*-

from itertools import product

import numpy as np

from .constants import GLOBAL_BOX, EX, EY, EZ
from .transformation import Transformation

from .geometry import Box


class Box(_Box):
    def __init__(self, center, xdim, ydim, zdim, ex=EX, ey=EY, ez=EZ):
        _Box.__init__(self, center, ex, ey, ez, xdim, ydim, zdim)

# class Box(_Box):
#     """Box object.
#
#     Parameters
#     ----------
#     base : array_like[float]
#         Base point of the box. It is one of its vertices. It has shape 3.
#     ex, ey, ez : array_like[float]
#         Basis vectors that give directions of box's edges. They must be
#         orthogonal. For now it is user's responsibility to ensure the
#         orthogonality. The length of basis vectors denote corresponding
#         dimensions of the box.
#     resolved : dict
#         Dictionary of surface test_box results, if results are +1 or -1.
#
#     Methods
#     -------
#     bounds()
#         Gets bounds in global coordinate system.
#     corners()
#         Gets coordinates of all corners in global CS.
#     generate_random_points(n)
#         Generates n random points inside the box.
#     get_random_points()
#         Gets already generated random points.
#     get_outer_boxes(global_box)
#         Gets a list of outer boxes.
#     volume()
#         Gets volume of the box.
#     split(dim, ratio)
#         Splits the box into two ones along dim direction.
#     test_point(p)
#         Tests whether point lies inside the box.
#     f_ieqcons(x, *arg)
#         Gets constraint function of the Box.
#     fprime_ieqcons(x, *arg)
#         Gets derivatives of constraint function of the Box.
#     test_surface(surf)
#         Checks the sense of surface surf with respect to the box.
#     """
#     def __init__(self, base, ex, ey, ez, resolved={}, unresolved=set(), node_cache={}):
#         self.base = np.array(base)
#         self.ex = np.array(ex)
#         self.ey = np.array(ey)
#         self.ez = np.array(ez)
#         self._tr = Transformation(translation=self.base,
#                                   rotation=np.concatenate((ex, ey, ez)))
#         self.scale = np.array([np.linalg.norm(self.ex),
#                                np.linalg.norm(self.ey),
#                                np.linalg.norm(self.ez)])
#         self._points = None
#         self._resolved = resolved.copy()
#         self._unresolved = set()
#         for s in unresolved:
#             sign = s.test_box(self)
#             if sign == 0:
#                 self._unresolved.add(s)
#             else:
#                 self._resolved[s] = sign
#         self._node_cache = node_cache.copy()
#         self._reduce_flag = False
#
#     def reset_cache(self):
#         self._resolved = {}
#         self._unresolved = set()
#         self._node_cache = {}
#
#     def has_last_unresolved(self):
#         return len(self._unresolved) == 1
#
#     def get_outer_boxes(self, global_box=GLOBAL_BOX):
#         """Gets a list of outer boxes.
#
#         Parameters
#         ----------
#         global_box : Box
#             Global box which limits the space under consideration.
#
#         Returns
#         -------
#         boxes : list[Box]
#             A list of outer boxes.
#         """
#         boxes = []
#         for i in range(3):
#             ratio = (self.base[i] - global_box.base[i]) / global_box.scale[i]
#             box1, box2 = global_box.split(dim=i, ratio=ratio)
#             boxes.append(box1)
#             global_box = box2
#         div_pt = self.base + self.ex + self.ey + self.ez
#         for i in range(3):
#             ratio = div_pt[i] / global_box.scale[i]
#             box1, box2 = global_box.split(dim=i, ratio=ratio)
#             boxes.append(box2)
#             global_box = box1
#         return boxes
#
#     def test_surface(self, surface):
#         """Checks whether the surface crosses the box.
#
#         Box defines a rectangular cuboid. This method checks if the surface
#         crosses the box, i.e. there is two points belonging to this box which
#         have different sense with respect to this surface.
#
#         Parameters
#         ----------
#         surface : Surface
#             Describes the surface to be checked.
#
#         Returns
#         -------
#         result : int
#             Test result. It equals one of the following values:
#             +1 if every point inside the box has positive sense.
#              0 if there are both points with positive and negative sense inside
#                the box
#             -1 if every point inside the box has negative sense.
#         """
#         if surface in self._resolved.keys():
#             return self._resolved[surface]
#         elif surface in self._unresolved:
#             return 0
#
#     def test_node(self, node, collect_stat=False):
#         if node._id in self._node_cache.keys():
#             return self._node_cache[node._id]
#         sign = node.test_box(self, collect_stat=collect_stat)
#         if sign != 0 and not self._reduce_flag:
#             self._node_cache[node._id] = sign
#         return sign
#
#     def test_point(self, p):
#         """Checks if point(s) p lies inside the box.
#
#         Parameters
#         ----------
#         p : array_like[float]
#             Coordinates of point(s) to be checked. If it is the only one point,
#             then p.shape=(3,). If it is an array of points, then
#             p.shape=(num_points, 3).
#
#         Returns
#         -------
#         result : bool or numpy.ndarray[bool]
#             If the point lies inside the box, then True value is returned.
#             If the point lies outside of the box False is returned.
#             Individual point - single value, array of points - array of
#             bools of shape (num_points,) is returned.
#         """
#         p = np.array(p)
#         p1 = p - self.base
#         # ex, ey and ez are not normalized!
#         mat = np.vstack((self.ex, self.ey, self.ez)) / self.scale
#         proj = np.dot(p1, mat.transpose()) / self.scale
#         axis = 1 if len(p.shape) == 2 else 0
#         return np.all(proj >= 0, axis=axis) * np.all(proj <= 1, axis=axis)
#
#     def split(self, dim=None, ratio=0.5):
#         """Splits the box two smaller ones along dim direction.
#
#         Parameters
#         ----------
#         dim : int
#             Dimension along which splitting must take place. 0 - ex, 1 - ey,
#             2 - ez. If not specified, then the box will be split along the
#             longest side.
#         ratio : float
#             The ratio of two new boxes volumes difference. If < 0.5 the first
#             box will be smaller.
#
#         Returns
#         -------
#         box1, box2 : Box
#             Resulting boxes. box1 contains parent box base point.
#         """
#         if dim is None:
#             dim = np.argmax(self.scale)
#         size1 = np.ones((3,))
#         size1[dim] *= ratio
#         size2 = np.ones((3,))
#         size2[dim] *= 1 - ratio
#         offset = np.zeros((3,))
#         offset[dim] = ratio
#         new_base = self.base + offset[0] * self.ex + offset[1] * self.ey + \
#                                offset[2] * self.ez
#         box1 = Box(self.base, size1[0] * self.ex, size1[1] * self.ey,
#                    size1[2] * self.ez, resolved=self._resolved, unresolved=self._unresolved, node_cache=self._node_cache)
#         box2 = Box(new_base, size2[0] * self.ex, size2[1] * self.ey,
#                    size2[2] * self.ez, resolved=self._resolved, unresolved=self._unresolved, node_cache=self._node_cache)
#         return box1, box2
#
#     def corners(self):
#         """Gets coordinates of all corners in global coordinate system.
#
#         Returns
#         -------
#         corners : np.ndarray[8, 3]
#             Coordinates of box's corners. Every row corresponds to one corner.
#         """
#         corners = np.zeros((8, 3))
#         multipliers = product([0, 1], [0, 1], [0, 1])
#         for i, (x, y, z) in enumerate(multipliers):
#             corners[i, :] = self.base + x * self.ex + y * self.ey + z * self.ez
#         return corners
#
#     def bounds(self):
#         """Gets bounds in global coordinate system."""
#         corners = self.corners()
#         bounds = [[lo, hi] for lo, hi in zip(np.amin(corners, axis=0),
#                                              np.amax(corners, axis=0))]
#         return bounds
#
#     def generate_random_points(self, n):
#         """Generates n random points inside the box."""
#         points = np.random.random((n, 3)) * self.scale
#         self._points = self._tr.apply2point(points)
#         return self._points
#
#     def get_random_points(self):
#         """Gets already generated random points."""
#         return self._points
#
#     def volume(self):
#         """Gets volume of the box."""
#         return np.multiply.reduce(self.scale)
#
#     def f_ieqcons(self):
#         """Gets constraint function of the Box.
#
#         Returns
#         -------
#         f(x, *args) : callable
#             A function that denotes inequality constraints corresponding to the
#             Box. This function returns array (6,) where each element >=0 if the
#             point x is inside the Box.
#         """
#         n = np.vstack((self.ex, self.ey, self.ez, -self.ex, -self.ey, -self.ez))
#         p1 = self.base.reshape((1, 3))
#         p2 = (self.base + self.ex + self.ey + self.ez).reshape((1, 3))
#         p0 = np.vstack((np.repeat(p1, 3, axis=0), np.repeat(p2, 3, axis=0)))
#         p = np.sum(np.multiply(n, p0), axis=1)
#         return lambda x, *args: np.dot(n, x) - p
#
#     def fprime_ieqcons(self):
#         """Gets derivatives of constraint function of the Box.
#
#         Returns
#         -------
#         fprime(x, *args) : callable
#             A function that denotes partial derivatives of constraints
#             corresponding to the Box. This function returns matrix (6, 3) -
#             of partial derivatives of the Box's surface equations at point x.
#         """
#         n = np.vstack((self.ex, self.ey, self.ez, -self.ex, -self.ey, -self.ez))
#         return lambda x, *args: n


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
