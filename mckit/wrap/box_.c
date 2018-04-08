#include "box_.h"
#include "common_.h"
#include <string.h>

#define  NO_IMPORT_ARRAY
#define  PY_ARRAY_UNIQUE_SYMBOL GEOMETRYMODULE_ARRAY_API
#include "numpy/arrayobject.h"


static void boxobj_dealloc(BoxObject * self);

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

static PyMethodDef boxobj_methods[] = {
        {"copy", (PyCFunction) boxobj_copy, METH_NOARGS, "Makes a copy of the box."},
        {"generate_random_points", (PyCFunction) boxobj_generate_random_points, METH_O,
                                                         "Generate N random points inside the box."},
        {"test_points", (PyCFunction) boxobj_test_points, METH_O, "Tests points if they are inside the box."},
        {"split", (PyCFunctionWithKeywords) boxobj_split, METH_VARARGS | METH_KEYWORDS,
                "Splits the box into two smaller."},
        {NULL}
};

PyTypeObject BoxType = {
        PyObject_HEAD_INIT(NULL)
        .tp_name = "geometry.Box",
        .tp_basicsize = sizeof(BoxObject),
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "Box objects",
        .tp_new = PyType_GenericNew,
        .tp_dealloc = (destructor) boxobj_dealloc,
        .tp_init = (initproc) boxobj_init,
        .tp_methods = boxobj_methods,
};

static void boxobj_dealloc(BoxObject * self)
{
    box_dispose(&self->box);
    Py_TYPE(self)->tp_free((PyObject*) self);
}

static double EX[] = {1, 0, 0};
static double EY[] = {0, 1, 0};
static double EZ[] = {0, 0, 1};

static int
boxobj_init(BoxObject * self, PyObject * args, PyObject * kwds)
{
    PyObject *pycent, *pyex = NULL, *pyey = NULL, *pyez = NULL;
    double xdim, ydim, zdim;
    double *cent, *ex, *ey, *ez;

    char * kwlist[] = {"", "", "", "", "ex", "ey", "ez", NULL};
    if (! PyArg_ParseTupleAndKeywords(args, kwds, "O&ddd|O&O&O&", kwlist,
                           convert_to_dbl_vec, &pycent,
                           &xdim ,&ydim, &zdim,
                           convert_to_dbl_vec, &pyex,
                           convert_to_dbl_vec, &pyey,
                           convert_to_dbl_vec, &pyez))
        return -1;

    cent = (double *) PyArray_DATA(pycent);
    if (pyex == NULL) ex = EX; else ex = (double *) PyArray_DATA(pyex);
    if (pyey == NULL) ey = EY; else ey = (double *) PyArray_DATA(pyey);
    if (pyez == NULL) ez = EZ; else ez = (double *) PyArray_DATA(pyez);
    box_dispose(&self->box);
    box_init(&self->box, cent, ex, ey, ez, xdim, ydim, zdim);
    
    Py_DECREF(cent);
    Py_DECREF(ex);
    Py_DECREF(ey);
    Py_DECREF(ez);
    
    return 0;
}

static PyObject *
boxobj_copy(BoxObject * self)
{
    BoxObject * box = (BoxObject *) PyType_GenericNew(&BoxType, NULL, NULL);
    if (box == NULL) return NULL;
    double *c = self->box.center;
    double *ex = self->box.ex, *ey = self->box.ey, *ez = self->box.ez;
    double xdim = self->box.dims[0], ydim = self->box.dims[1], zdim = self->box.dims[2];
    box_init(&box->box, c, ex, ey, ez, xdim, ydim, zdim);
    return (PyObject *) box;
}

static PyObject *
boxobj_generate_random_points(BoxObject * self, PyObject * npts)
{
    if (! PyLong_CheckExact(npts)) {
        PyErr_SetString(PyExc_ValueError, "Integer value is expected");
        return NULL;
    }
    size_t n = PyLong_AsLong(npts);

    npy_intp dims[] = {n, NDIM};
    PyObject * points = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
    if (points == NULL) return NULL;
    
    int status = box_generate_random_points(&self->box, \
                                (double *) PyArray_DATA(points), n);
    if (status == BOX_FAILURE) {
        PyErr_SetString(PyExc_MemoryError, "Could not generate points.");
        Py_DECREF(points);
        points = NULL;
    }
    return points;
}

static PyObject *
boxobj_test_points(BoxObject * self, PyObject * points)
{
    PyObject * pts;
    if (! convert_to_dbl_vec_array(points, &pts)) return NULL;

    npy_intp size = PyArray_SIZE((PyArrayObject *) pts);
    size_t npts = size > NDIM ? PyArray_DIM((PyArrayObject *) pts, 0) : 1;
    npy_intp dims[] = {npts};
    PyObject * result = PyArray_EMPTY(1, dims, NPY_INT, 0);
    if (result == NULL) {
        Py_DECREF(pts);
        return NULL;
    }
    
    box_test_points(&self->box, (double *) PyArray_DATA(pts), npts, \
                   (int *) PyArray_DATA(result));
    Py_DECREF(pts);
    return result;
}

static PyObject *
boxobj_split(BoxObject * self, PyObject * args, PyObject * kwds)
{
    char * dir = "auto";
    double ratio = 0.5;
    int direct;
    static char * kwlist[] = {"dir", "ratio", NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|$sd", &dir, &ratio)) return NULL;

    if (dir == "auto") direct = BOX_SPLIT_AUTODIR;
    else if (strcmp(dir, "x")) direct = BOX_SPLIT_X;
    else if (strcmp(dir, "y")) direct = BOX_SPLIT_Y;
    else if (strcmp(dir, "z")) direct = BOX_SPLIT_Z;
    else {
        PyErr_SetString(PyExc_ValueError, "Unknown splitting direction.");
        return NULL;
    }

    if (ratio <= 0 || ratio >= 1) {
        PyErr_SetString(PyExc_ValueError, "Split ratio is out of range (0, 1).");
        return NULL;
    }

    BoxObject * box1 = (BoxObject *) PyType_GenericNew(&BoxType, NULL, NULL);
    BoxObject * box2 = (BoxObject *) PyType_GenericNew(&BoxType, NULL, NULL);
    int status = box_split(&self->box, &box1->box, &box2->box, direct, ratio);

    if (status == BOX_FAILURE) {
        PyErr_SetString(PyExc_MemoryError, "Could not create new boxes.");
        Py_XDECREF(box1);
        Py_XDECREF(box2);
        return NULL;
    }
    return Py_BuildValue("(OO)", box1, box2);
}

