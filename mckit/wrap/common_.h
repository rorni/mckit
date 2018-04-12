#ifndef WRAP_COMMON_H
#define WRAP_COMMON_H

#include <Python.h>
#include <stddef.h>

#define parent_pyobject(type, field, pointer) ((PyObject *) ((char *) (pointer) - offsetof(type, field)))

int
convert_to_dbl_vec(PyObject * obj, PyObject ** addr);

int
convert_to_dbl_vec_array(PyObject * obj, PyObject ** addr);

#endif //WRAP_COMMON_H
