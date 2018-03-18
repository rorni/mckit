import cython
import numpy as np
cimport numpy as np
cimport cbox
from cpython.mem cimport PyMem_Malloc, PyMem_Free

 
cdef class Box:
    cdef cbox.Box* _cbox

    def __cinit__(self, *args, **kwargs):
        self._cbox = <cbox.Box*> PyMem_Malloc(sizeof(cbox.Box))
        if self._cbox is NULL:
            raise MemoryError()
    
    def __init__(self, center, ex, ey, ez, xdim, ydim, zdim):
        cdef double xd = xdim
        cdef double yd = ydim
        cdef double zd = zdim
        cdef np.ndarray[double, mode='c'] cen = np.array(center, dtype=np.float)
        cdef np.ndarray[double, mode='c'] e_x = np.array(ex, dtype=np.float)
        cdef np.ndarray[double, mode='c'] e_y = np.array(ey, dtype=np.float)
        cdef np.ndarray[double, mode='c'] e_z = np.array(ez, dtype=np.float)
        
        if cen.size != 3 or e_x.size != 3 or e_y.size != 3 or e_z.size != 3:
            raise TypeError()
        
        cbox.box_init(self._cbox, &cen[0], &e_x[0], &e_y[0], &e_z[0], 
                        xdim, ydim, zdim)
            
    def __dealloc__(self):
        cbox.box_dispose(self._cbox)
        PyMem_Free(self._cbox)
        
    def generate_random_points(self, int npts):
        points = np.empty((npts, 3), dtype=np.float64)
        cdef np.ndarray[double, ndim=2, mode='c'] pts = points
        cbox.box_generate_random_points(self._cbox, &pts[0, 0], npts)
        return points
        
    def test_points(self, points):
        cdef np.ndarray[double, ndim=2, mode='c'] pts = np.array(points)
        cdef int npts = points.shape[0]
        if points.shape[1] != 3:
            raise TypeError()
        result = np.empty((npts,), dtype=int)
        cdef np.ndarray[int, mode='c'] r = result
        cbox.box_test_points(self._cbox, &pts[0, 0], npts, &r[0])
        return result
        
    def split(self, int dir=-1, ratio=0.5):
        void = [0, 0, 0]
        box1 = Box(void, void, void, void, 0, 0, 0)
        box2 = Box(void, void, void, void, 0, 0, 0)
        
        cbox.box_split(self._cbox, <cbox.Box*> box1._cbox, 
                                   <cbox.Box*> box2._cbox, dir, <double> ratio)
        return box1, box2
        
    def ieqcons(self, x, args):
        pass
 