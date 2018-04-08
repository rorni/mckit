#include <Python.h>
#include <string.h>
#include <structmember.h>
#define  NO_IMPORT_ARRAY
#define  PY_ARRAY_UNIQUE_SYMBOL GEOMETRYMODULE_ARRAY_API
#include "numpy/arrayobject.h"

#include "common_.h"
#include "box_.h"
#include "surface_.h"

#include "../src/surface.h"


typedef struct {
    PyObject ob_base;
    Surface surf;
} SurfaceObject;

typedef struct {
    PyObject ob_base;
    Plane surf;
} PlaneObject;

typedef struct {
    PyObject ob_base;
    Sphere surf;
} SphereObject;

typedef struct {
    PyObject ob_base;
    Cylinder surf;
} CylinderObject;

typedef struct {
    PyObject ob_base;
    Cone surf;
} ConeObject;

typedef struct {
    PyObject ob_base;
    Torus surf;
} TorusObject;

typedef struct {
    PyObject ob_base;
    GQuadratic surf;
} GQuadraticObject;

static PyObject *
surfobj_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyErr_SetString(PyExc_TypeError, "Can't instantiate abstract class Surface");
    return NULL;
}

static PyObject *
surfobj_test_points(SurfaceObject * self, PyObject * points)
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

    surface_test_points(&self->surf, (double *) PyArray_DATA(pts), npts, \
                   (int *) PyArray_DATA(result));
    Py_DECREF(pts);
    return result;
}

static PyObject *
surfobj_test_box(SurfaceObject * self, PyObject * box)
{
    if (! PyObject_TypeCheck(box, &BoxType)) {
        PyErr_SetString(PyExc_ValueError, "Box instance is expected");
        return NULL;
    }

    int result = surface_test_box(&self->surf, &((BoxObject *) box)->box);

    return Py_BuildValue("i", result);
}

static PyMethodDef surfobj_methods[] = {
        {"test_box", (PyCFunction) surfobj_test_box, METH_O, "Tests where the box is located with respect to the surface."},
        {"test_points", (PyCFunction) surfobj_test_points, METH_O, "Tests senses of the points with respect to the surface."},
        {NULL}
};

static void
torusobj_dealloc(TorusObject * self)
{
    torus_dispose(&self->surf);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
parse_surface_args(PyObject * kwds, int * name, char * modifier)
{
    *name = 0;
    char * modstr = NULL;
    char * kwlist[] = {"name", "modifier", NULL};
    PyObject * args = Py_BuildValue("()");
    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|$is", kwlist, name, &modstr)) {
        Py_DECREF(args);
        return 0;
    }
    Py_DECREF(args);

    if (modstr == NULL) *modifier = ORDINARY;
    if (strcmp(modstr, "*") == 0) *modifier = REFLECTIVE;
    else if (strcmp(modstr, "+") == 0) *modifier = WHITE;
    else {
        PyErr_SetString(PyExc_ValueError, "Unknown modifier");
        return 0;
    }

    if (*name < 0) {
        PyErr_SetString(PyExc_ValueError, "Name must be non-negative");
        return 0;
    }
    return 1;
}

static int
planeobj_init(PlaneObject * self, PyObject * args, PyObject * kwds)
{
    PyObject * norm;
    double offset;
    int name;
    char modifier;
    if (! PyArg_ParseTuple(args, "O&d", convert_to_dbl_vec, &norm, &offset)) return -1;
    if (! parse_surface_args(kwds, &name, &modifier)) return -1;

    plane_init(&self->surf, name, modifier, (double *) PyArray_DATA(norm), offset);
    Py_DECREF(norm);
    return 0;
}

static int
sphereobj_init(SphereObject * self, PyObject * args, PyObject * kwds)
{
    PyObject * center;
    double radius;
    int name;
    char modifier;
    if (! PyArg_ParseTuple(args, "O&d", convert_to_dbl_vec, &center, &radius)) return -1;
    if (! parse_surface_args(kwds, &name, &modifier)) return -1;

    sphere_init(&self->surf, name, modifier, (double *) PyArray_DATA(center), radius);
    Py_DECREF(center);
    return 0;
}

static int
cylinderobj_init(CylinderObject * self, PyObject * args, PyObject * kwds)
{
    PyObject * point, *axis;
    double radius;
    int name;
    char modifier;
    if (! PyArg_ParseTuple(args, "O&O&d", convert_to_dbl_vec, &point, convert_to_dbl_vec, &axis, &radius)) return -1;
    if (! parse_surface_args(kwds, &name, &modifier)) return -1;

    cylinder_init(&self->surf, name, modifier, (double *) PyArray_DATA(point), (double *) PyArray_DATA(axis), radius);
    Py_DECREF(point);
    Py_DECREF(axis);
    return 0;
}

static int
coneobj_init(ConeObject * self, PyObject * args, PyObject * kwds)
{
    PyObject *apex, *axis;
    double ta;
    int name;
    char modifier;
    if (! PyArg_ParseTuple(args, "O&O&d", convert_to_dbl_vec, &apex, convert_to_dbl_vec, &axis, &ta)) return -1;
    if (! parse_surface_args(kwds, &name, &modifier)) return -1;

    cone_init(&self->surf, name, modifier, (double *) PyArray_DATA(apex), (double *) PyArray_DATA(axis), ta);
    Py_DECREF(apex);
    Py_DECREF(axis);
    return 0;
}

static int
torusobj_init(TorusObject * self, PyObject * args, PyObject * kwds)
{
    PyObject *center, *axis;
    double r, a, b;
    int name;
    char modifier;
    if (! PyArg_ParseTuple(args, "O&O&d", convert_to_dbl_vec, &center, convert_to_dbl_vec, &axis, &r, &a, &b)) return -1;
    if (! parse_surface_args(kwds, &name, &modifier)) return -1;

    int status = torus_init(&self->surf, name, modifier,
                            (double *) PyArray_DATA(center), (double *) PyArray_DATA(axis), r, a, b);
    if (status == SURFACE_FAILURE) {
        PyErr_SetString(PyExc_MemoryError, "Can't allocate memory for torus content.");
        Py_DECREF(center);
        Py_DECREF(axis);
        return -1;
    }
    Py_DECREF(center);
    Py_DECREF(axis);
    return 0;
}

static int
gqobj_init(GQuadraticObject * self, PyObject * args, PyObject * kwds)
{
    PyObject *m, *v;
    double k;
    int name;
    char modifier;
    if (! PyArg_ParseTuple(args, "O&O&d", convert_to_dbl_vec_array, &m, convert_to_dbl_vec, &v, &k)) return -1;
    if (! parse_surface_args(kwds, &name, &modifier)) return -1;

    gq_init(&self->surf, name, modifier, (double *) PyArray_DATA(m), (double *) PyArray_DATA(v), k);

    Py_DECREF(m);
    Py_DECREF(v);
    return 0;
}

PyTypeObject SurfaceType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "geometry.Surface",
        .tp_basicsize = sizeof(SurfaceObject),
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "Surface base class",
        .tp_new = surfobj_new,
        .tp_methods = surfobj_methods,
};

PyTypeObject PlaneType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_base = &SurfaceType,
        .tp_name = "geometry.Plane",
        .tp_basicsize = sizeof(PlaneObject),
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "Plane class",
        .tp_new = PyType_GenericNew,
        .tp_init = (initproc) planeobj_init,
};

PyTypeObject SphereType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_base = &SurfaceType,
        .tp_name = "geometry.Sphere",
        .tp_basicsize = sizeof(SphereObject),
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "Sphere class",
        .tp_new = PyType_GenericNew,
        .tp_init = (initproc) sphereobj_init,
};

PyTypeObject CylinderType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_base = &SurfaceType,
        .tp_name = "geometry.Cylinder",
        .tp_basicsize = sizeof(CylinderObject),
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "Cylinder class",
        .tp_new = PyType_GenericNew,
        .tp_init = (initproc) cylinderobj_init,
};

PyTypeObject ConeType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_base = &SurfaceType,
        .tp_name = "geometry.Cone",
        .tp_basicsize = sizeof(ConeObject),
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "Cone class",
        .tp_new = PyType_GenericNew,
        .tp_init = (initproc) coneobj_init,
};

PyTypeObject TorusType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_base = &SurfaceType,
        .tp_name = "geometry.Torus",
        .tp_basicsize = sizeof(TorusObject),
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "Torus class",
        .tp_new = PyType_GenericNew,
        .tp_dealloc = (destructor) torusobj_dealloc,
        .tp_init = (initproc) torusobj_init,
};

PyTypeObject GQuadraticType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_base = &SurfaceType,
        .tp_name = "geometry.GQuadratic",
        .tp_basicsize = sizeof(GQuadraticObject),
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "GQuadratic class",
        .tp_new = PyType_GenericNew,
        .tp_init = (initproc) gqobj_init,
};
