#include "common_.h"
#define  NO_IMPORT_ARRAY
#define  PY_ARRAY_UNIQUE_SYMBOL GEOMETRYMODULE_ARRAY_API
#include "numpy/arrayobject.h"

#include "../src/common.h"

int
convert_to_dbl_vec(PyObject * obj, PyObject ** addr)
{
    PyObject * arr = PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr == NULL) return 0;

    if (PyArray_SIZE((PyArrayObject *) arr) != NDIM) {
        PyErr_SetString(PyExc_ValueError, "Vector of length 3 is expected");
        Py_DECREF(arr);
    }
    *addr = arr;
    return 1;
}

int
convert_to_dbl_vec_array(PyObject * obj, PyObject ** addr)
{
    PyObject * arr = PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr == NULL) return 0;

    int n = PyArray_NDIM(arr);
    if (n == 0 || n > 2) {
        PyErr_SetString(PyExc_ValueError, "Vector or matrix are expected");
        goto error;
    }
    npy_intp size, last_dim;
    size = PyArray_SIZE(arr);
    last_dim = PyArray_DIM(arr, n - 1);
    if (last_dim != NDIM) {
        PyErr_SetString(PyExc_ValueError, "Shape (n, 3) is expected");
        goto error;
    }
    *addr = arr;
    return 1;
  error:
    Py_DECREF(arr);
    return 0;
}
