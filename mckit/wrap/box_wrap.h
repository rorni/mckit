#include <Python.h>
#include <structmember.h>
#include "../src/box.h"


typedef struct {
    PyObject_HEAD
    Box * box;
} BoxObject;

static void boxobj_dealloc(BoxObject * self);

static PyObject * 
boxobj_new(PyTypeObject * type, PyObject * args, PyObject * kwds);

static int
boxobj_init(BoxObject * self, PyObject * args, PyObject * kwds);

