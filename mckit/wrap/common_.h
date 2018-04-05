#ifndef WRAP_COMMON_H
#define WRAP_COMMON_H

#include <Python.h>
#include "numpy/arrayobject.h"

#include "../src/common.h"

static int
convert_to_dbl_vec(PyObject * obj, PyObject ** addr);

static int
convert_to_dbl_vec_array(PyObject * obj, PyObject ** addr);

#endif //WRAP_COMMON_H
