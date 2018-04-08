#include "common_.h"

int
convert_to_dbl_vec(PyObject * obj, PyObject ** addr)
{
    *addr = PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (*addr == NULL) return 0;

    if (PyArray_SIZE((PyArrayObject *) *addr) != NDIM) {
        PyErr_SetString(PyExc_ValueError, "Vector of length 3 is expected");
        Py_DECREF(*addr);
    }
    return 1;
}

int
convert_to_dbl_vec_array(PyObject * obj, PyObject ** addr)
{
    *addr = PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (*addr == NULL) return 0;

    PyArrayObject * arr = (PyArrayObject *) *addr;
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
    return 1;
  error:
    Py_DECREF(*addr);
    return 0;
}
