import numpy as np
cimport numpy as np
cimport csurface
cimport cbox
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef class Surface:
    cdef csurface.Surface * _csurf
    
    def __init__(self):
        pass

    def __dealloc__(self):
        csurface.surface_dispose(self._csurf)
        PyMem_Free(self._csurf)
        
    def test_points(self, points):
        p = np.array(points, dtype=np.float64)
        l = len(p.shape)
        if l == 1:
            p = p.reshape((1, 3))
        cdef np.ndarray[double, ndim=2, mode='c'] pts = p
        cdef int npts = p.shape[0]
        if p.shape[1] != 3:
            raise ValueError("Incorrect shape of points")
            
        result = np.empty((npts,), dtype=int)
        cdef np.ndarray[int, mode='c'] r = result
        csurface.surface_test_points(self._csurf, &pts[0, 0], npts, &r[0])
        if l == 1:
            return result[0]
        return result
        
    def test_box(self, Box box):
        return csurface.surface_test_box(self._csurf, box._cbox)
        
cdef class Plane(Surface):
    
    def __cinit__(self, *args, **kwargs):
        self._csurf = <csurface.Surface*> PyMem_Malloc(sizeof(csurface.Plane))
        if self._csurf is NULL:
            raise MemoryError()
            
    def __init__(self, name, modifier, norm, offset):
        cdef int cname = name
        cdef int cmod = modifier
        cdef np.ndarray[double, mode='c'] cnorm = np.array(norm, dtype=np.float)
        cdef double coffset = offset
        
        if cnorm.size != 3:
            raise ValueError("Incorrect vector size")
            
        csurface.plane_init(self._csurf, cname, cmod, &cnorm[0], coffset)
    
