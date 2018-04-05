#ifndef __BOX_WRAP_H
#define __BOX_WRAP_H

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

static PyObject *
boxobj_copy(BoxObject * self);

static PyObject *
boxobj_generate_random_points(BoxObject * self, PyObject * npts);

static PyObject *
boxobj_test_points(BoxObject * self, PyObject * points);

static PyObject *
boxobj_split(BoxObject * self, PyObject * args, PyObject * kwds);

static PyObject *
boxobj_get_volume(BoxObject * self, void * closure);

#endif

