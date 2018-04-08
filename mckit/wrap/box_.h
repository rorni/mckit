#ifndef WRAP_BOX_H
#define WRAP_BOX_H

#include <Python.h>
#include "../src/box.h"

extern PyTypeObject BoxType;

typedef struct {
    PyObject ob_base;
    Box box;
} BoxObject;

#endif //WRAP_BOX_H
