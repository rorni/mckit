import numpy as np
cimport numpy as np
cimport cgeometry
cimport csurface
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef class Node:
    cdef cgeometry.Node * _node

    def __cinit__(self, opc, *args, **kwargs):
        cdef int copc
        if opc == 'I' or opc == 0:
            copc = 0
        elif opc == 'C' or opc == 1:
            copc = 1
        elif opc == 'V' or opc == 2:
            copc = 2
        elif opc == 'U' or opc == 3:
            copc = 3
        elif opc == 'E' or opc == 4:
            copc = 4
        elif opc == 'S' or opc == 5:
            copc = 5

        cdef int narg = len(args)
        cdef void ** cargs = <void**> PyMem_Malloc(narg * sizeof(void*))
        cdef csurface.Surface surf
        if copc % 3 == 1:
            surf = <csurface.Surface> args[0]
            cargs[0] = <void*> (surf._csurf)
        elif copc % 3 == 0:
            for i in range(narg):
                cargs[i] = args[i]._node

        self._node = cgeometry.node_create(copc, narg, cargs)
        PyMem_Free(cargs)

    def __dealloc__(self):
        cgeometry.node_free(self._node)

    def test_box(self, Box box, collect=False):
        cdef char ccol = collect
        return cgeometry.node_test_box(self._node, box._cbox, ccol)

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
        cgeometry.node_test_points(self._node, &pts[0, 0], npts, &r[0])
        if l == 1:
            return result[0]
        return result

    def is_empty(self):
        return cgeometry.is_empty(self._node) == 1

    def is_universe(self):
        return cgeometry.is_universe(self._node) == 1

    def complexity(self):
        return cgeometry.node_complexity(self._node)

