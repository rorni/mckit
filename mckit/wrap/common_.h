#ifndef WRAP_COMMON_H
#define WRAP_COMMON_H

#include <Python.h>
#include "numpy/arrayobject.h"

#include "../src/common.h"

int
convert_to_dbl_vec(PyObject * obj, PyObject ** addr);

int
convert_to_dbl_vec_array(PyObject * obj, PyObject ** addr);

#endif //WRAP_COMMON_H
