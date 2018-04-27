#include <Python.h>
#include <structmember.h>
#include <string.h>

#include "numpy/arrayobject.h"

#include "box.h"
#include "surface.h"
#include "shape.h"

#include "box_doc.h"
#include "surf_doc.h"

// ===================================================================================================== //

#define parent_pyobject(type, field, pointer) ((PyObject *) ((char *) (pointer) - offsetof(type, field)))

static int
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

static int
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

// ========================================================================================== //
// =============================== Module constants ========================================= //
// ========================================================================================== //

static PyObject * module_dict;

#define GET_NAME(name) (PyDict_GetItemString(module_dict, name))
#define MAX_DIM 5000    // Size of global box in cm.
#define MIN_VOLUME 0.001 // Min volume size.

#define ORIGIN "ORIGIN"
#define EX "EX"
#define EY "EY"
#define EZ "EZ"
#define GLOBAL_BOX "GLOBAL_BOX"
#define MIN_VOLUME_NAME "MIN_VOLUME"

// ========================================================================================== //
// ===============================  Box wrappers ============================================ //
// ========================================================================================== //

typedef struct {
    PyObject ob_base;
    Box box;
} BoxObject;

static void       boxobj_dealloc(BoxObject * self);
static int        boxobj_init(BoxObject * self, PyObject * args, PyObject * kwds);
static PyObject * boxobj_copy(BoxObject * self);
static PyObject * boxobj_generate_random_points(BoxObject * self, PyObject * npts);
static PyObject * boxobj_test_points(BoxObject * self, PyObject * points);
static PyObject * boxobj_split(BoxObject * self, PyObject * args, PyObject * kwds);
static PyObject * boxobj_getcorners(BoxObject * self, void * closure);
static PyObject * boxobj_getvolume(BoxObject * self, void * closure);
static PyObject * boxobj_getbounds(BoxObject * self, void * closure);
static PyObject * boxobj_getcenter(BoxObject * self, void * closure);

static PyGetSetDef boxobj_getsetters[] = {
        {"corners", (getter) boxobj_getcorners, NULL, "Box's corners", NULL},
        {"volume",  (getter) boxobj_getvolume,  NULL, "Box's volume",  NULL},
        {"bounds",  (getter) boxobj_getbounds,  NULL, "Box's bounds",  NULL},
        {"center",  (getter) boxobj_getcenter,  NULL, "Box's center",  NULL},
        {NULL}
};

static PyMethodDef boxobj_methods[] = {
        {"copy", (PyCFunction) boxobj_copy, METH_NOARGS, BOX_COPY_DOC},
        {"generate_random_points", (PyCFunction) boxobj_generate_random_points, METH_O, BOX_GRP_DOC},
        {"test_points", (PyCFunction) boxobj_test_points, METH_O, BOX_TEST_POINTS_DOC},
        {"split", (PyCFunctionWithKeywords) boxobj_split, METH_VARARGS | METH_KEYWORDS, BOX_SPLIT_DOC},
        {NULL}
};

static PyTypeObject BoxType = {
        PyObject_HEAD_INIT(NULL)
        .tp_name = "geometry.Box",
        .tp_basicsize = sizeof(BoxObject),
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = BOX_DOC,
        .tp_new = PyType_GenericNew,
        .tp_dealloc = (destructor) boxobj_dealloc,
        .tp_init = (initproc) boxobj_init,
        .tp_methods = boxobj_methods,
        .tp_getset = boxobj_getsetters,
};

static void boxobj_dealloc(BoxObject * self)
{
    box_dispose(&self->box);
    Py_TYPE(self)->tp_free((PyObject*) self);
}

static int
boxobj_init(BoxObject * self, PyObject * args, PyObject * kwds)
{
    PyObject *cent, *ex = NULL, *ey = NULL, *ez = NULL;
    double xdim, ydim, zdim;

    char *kwlist[] = {"", "", "", "", "ex", "ey", "ez", NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "O&ddd|O&O&O&", kwlist,
                           convert_to_dbl_vec, &cent, &xdim ,&ydim, &zdim,
                           convert_to_dbl_vec, &ex, convert_to_dbl_vec, &ey, convert_to_dbl_vec, &ez))
        return -1;

    if (ex == NULL) {
        ex = GET_NAME(EX);
        Py_INCREF(ex);
    }
    if (ey == NULL) {
        ey = GET_NAME(EY);
        Py_INCREF(ey);
    }
    if (ez == NULL) {
        ez = GET_NAME(EZ);
        Py_INCREF(ez);
    }

    box_dispose(&self->box);
    box_init(&self->box, (double *) PyArray_DATA(cent),
             (double *) PyArray_DATA(ex), (double *) PyArray_DATA(ey), (double *) PyArray_DATA(ez),
             xdim, ydim, zdim);

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
    box_copy(&box->box, &self->box);
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

    int status = box_generate_random_points(&self->box, n, \
                                (double *) PyArray_DATA(points));
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

    box_test_points(&self->box, npts, (double *) PyArray_DATA(pts), \
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

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|$sd", kwlist, &dir, &ratio)) return NULL;

    if (strcmp(dir, "auto") == 0) direct = BOX_SPLIT_AUTODIR;
    else if (strcmp(dir, "x") == 0) direct = BOX_SPLIT_X;
    else if (strcmp(dir, "y") == 0) direct = BOX_SPLIT_Y;
    else if (strcmp(dir, "z") == 0) direct = BOX_SPLIT_Z;
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

static PyObject * boxobj_getcorners(BoxObject * self, void * closure)
{
    npy_intp dims[] = {NCOR, NDIM};
    PyObject * corners = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
    int i;
    double * data = (double *) PyArray_DATA(corners);
    for (i = 0; i < NCOR * NDIM; ++i) {
        data[i] = self->box.corners[i];
    }
    return corners;
}


static PyObject * boxobj_getvolume(BoxObject * self, void * closure)
{
    return Py_BuildValue("d", self->box.volume);
}

static PyObject * boxobj_getbounds(BoxObject * self, void * closure)
{
    npy_intp dims[] = {NDIM, 2};
    PyObject * bounds = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
    int i;
    double * data = (double *) PyArray_DATA(bounds);
    for (i = 0; i < NDIM; ++i) {
        data[2 * i] = self->box.lb[i];
        data[2 * i + 1] = self->box.ub[i];
    }
    return bounds;
}

static PyObject * boxobj_getcenter(BoxObject * self, void * closure)
{
    npy_intp dims[] = {NDIM};
    PyObject * center = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
    int i;
    double * data = (double *) PyArray_DATA(center);
    for (i = 0; i < NDIM; ++i) data[i] = self->box.center[i];
    return center;
}


// ========================================================================================== //
// ========================== Surface wrappers ============================================== //
// ========================================================================================== //

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
    PyObject * result = PyArray_EMPTY(1, dims, NPY_BYTE, 0);
    if (result == NULL) {
        Py_DECREF(pts);
        return NULL;
    }

    surface_test_points(&self->surf, npts, (double *) PyArray_DATA(pts), \
                                           (char *) PyArray_DATA(result));
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
        {"test_box", (PyCFunction) surfobj_test_box, METH_O, SURF_TEST_BOX_DOC},
        {"test_points", (PyCFunction) surfobj_test_points, METH_O, SURF_TEST_POINTS_DOC},
        {NULL}
};

static int
planeobj_init(PlaneObject * self, PyObject * args, PyObject * kwds)
{
    PyObject * norm;
    double offset;
    if (! PyArg_ParseTuple(args, "O&d", convert_to_dbl_vec, &norm, &offset)) return -1;

    plane_init(&self->surf, (double *) PyArray_DATA(norm), offset);
    Py_DECREF(norm);
    return 0;
}

static int
sphereobj_init(SphereObject * self, PyObject * args, PyObject * kwds)
{
    PyObject * center;
    double radius;
    if (! PyArg_ParseTuple(args, "O&d", convert_to_dbl_vec, &center, &radius)) return -1;

    sphere_init(&self->surf, (double *) PyArray_DATA(center), radius);
    Py_DECREF(center);
    return 0;
}

static int
cylinderobj_init(CylinderObject * self, PyObject * args, PyObject * kwds)
{
    PyObject * point, *axis;
    double radius;
    if (! PyArg_ParseTuple(args, "O&O&d", convert_to_dbl_vec, &point, convert_to_dbl_vec, &axis, &radius)) return -1;

    cylinder_init(&self->surf, (double *) PyArray_DATA(point), (double *) PyArray_DATA(axis), radius);
    Py_DECREF(point);
    Py_DECREF(axis);
    return 0;
}

static int
coneobj_init(ConeObject * self, PyObject * args, PyObject * kwds)
{
    PyObject *apex, *axis;
    double ta;
    if (! PyArg_ParseTuple(args, "O&O&d", convert_to_dbl_vec, &apex, convert_to_dbl_vec, &axis, &ta)) return -1;

    cone_init(&self->surf, (double *) PyArray_DATA(apex), (double *) PyArray_DATA(axis), ta * ta);
    Py_DECREF(apex);
    Py_DECREF(axis);
    return 0;
}

static int
torusobj_init(TorusObject * self, PyObject * args, PyObject * kwds)
{
    PyObject *center, *axis;
    double r, a, b;
    if (! PyArg_ParseTuple(args, "O&O&ddd", convert_to_dbl_vec, &center, convert_to_dbl_vec, &axis, &r, &a, &b)) return -1;

    int status = torus_init(&self->surf,
                            (double *) PyArray_DATA(center), (double *) PyArray_DATA(axis), r, a, b);
    Py_DECREF(center);
    Py_DECREF(axis);
    return 0;
}

static int
gqobj_init(GQuadraticObject * self, PyObject * args, PyObject * kwds)
{
    PyObject *m, *v;
    double k;
    if (! PyArg_ParseTuple(args, "O&O&d", convert_to_dbl_vec_array, &m, convert_to_dbl_vec, &v, &k)) return -1;

    gq_init(&self->surf, (double *) PyArray_DATA(m), (double *) PyArray_DATA(v), k);

    Py_DECREF(m);
    Py_DECREF(v);
    return 0;
}

static PyTypeObject SurfaceType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "geometry.Surface",
        .tp_basicsize = sizeof(SurfaceObject),
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "Surface base class",
        .tp_new = surfobj_new,
        .tp_methods = surfobj_methods,
};

static PyObject *
planeobj_getnorm(PlaneObject * self, void * closure)
{
    npy_intp dims[] = {NDIM};
    PyObject * norm = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
    double * data = (double *) PyArray_DATA(norm);
    for (int i = 0; i < NDIM; ++i) data[i] = self->surf.norm[i];
    return norm;
}

static PyObject *
planeobj_getoffset(PlaneObject * self, void * closure)
{
    return Py_BuildValue("d", self->surf.offset);
}

static PyGetSetDef planeobj_getset[] = {
        {"_v", (getter) planeobj_getnorm, NULL, "Plane's normal", NULL},
        {"_k", (getter) planeobj_getoffset, NULL, "Plane's offset", NULL},
        {NULL}
};

static PyTypeObject PlaneType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_base = &SurfaceType,
        .tp_name = "geometry.Plane",
        .tp_basicsize = sizeof(PlaneObject),
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "Plane class",
        .tp_new = PyType_GenericNew,
        .tp_init = (initproc) planeobj_init,
        .tp_getset = planeobj_getset,
};

static PyObject *
sphereobj_getcenter(SphereObject * self, void * closure)
{
    npy_intp dims[] = {NDIM};
    PyObject * center = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
    double * data = (double *) PyArray_DATA(center);
    for (int i = 0; i < NDIM; ++i) data[i] = self->surf.center[i];
    return center;
}

static PyObject *
sphereobj_getradius(SphereObject * self, void * closure)
{
    return Py_BuildValue("d", self->surf.radius);
}

static PyGetSetDef sphereobj_getset[] = {
        {"_center", (getter) sphereobj_getcenter, NULL, "Sphere's center", NULL},
        {"_radius", (getter) sphereobj_getradius, NULL, "Sphere's radius", NULL},
        {NULL}
};

static PyTypeObject SphereType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_base = &SurfaceType,
        .tp_name = "geometry.Sphere",
        .tp_basicsize = sizeof(SphereObject),
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "Sphere class",
        .tp_new = PyType_GenericNew,
        .tp_init = (initproc) sphereobj_init,
        .tp_getset = sphereobj_getset,
};

static PyObject *
cylinderobj_getpt(CylinderObject * self, void * closure)
{
    npy_intp dims[] = {NDIM};
    PyObject * pt = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
    double * data = (double *) PyArray_DATA(pt);
    for (int i = 0; i < NDIM; ++i) data[i] = self->surf.point[i];
    return pt;
}

static PyObject *
cylinderobj_getaxis(CylinderObject * self, void * closure)
{
    npy_intp dims[] = {NDIM};
    PyObject * axis = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
    double * data = (double *) PyArray_DATA(axis);
    for (int i = 0; i < NDIM; ++i) data[i] = self->surf.axis[i];
    return axis;
}

static PyObject *
cylinderobj_getradius(CylinderObject * self, void * closure)
{
    return Py_BuildValue("d", self->surf.radius);
}

static PyGetSetDef cylinderobj_getset[] = {
        {"_pt", (getter) cylinderobj_getpt, NULL, "Cylinder's axis point", NULL},
        {"_axis", (getter) cylinderobj_getaxis, NULL, "Cylinder's axis", NULL},
        {"_radius", (getter) cylinderobj_getradius, NULL, "Cylinder's radius", NULL},
        {NULL}
};

static PyTypeObject CylinderType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_base = &SurfaceType,
        .tp_name = "geometry.Cylinder",
        .tp_basicsize = sizeof(CylinderObject),
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "Cylinder class",
        .tp_new = PyType_GenericNew,
        .tp_init = (initproc) cylinderobj_init,
        .tp_getset = cylinderobj_getset,
};

static PyObject *
coneobj_getapex(ConeObject * self, void * closure)
{
    npy_intp dims[] = {NDIM};
    PyObject * apex = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
    double * data = (double *) PyArray_DATA(apex);
    for (int i = 0; i < NDIM; ++i) data[i] = self->surf.apex[i];
    return apex;
}

static PyObject *
coneobj_getaxis(ConeObject * self, void * closure)
{
    npy_intp dims[] = {NDIM};
    PyObject * axis = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
    double * data = (double *) PyArray_DATA(axis);
    for (int i = 0; i < NDIM; ++i) data[i] = self->surf.axis[i];
    return axis;
}

static PyObject *
coneobj_getta(ConeObject * self, void * closure)
{
    return Py_BuildValue("d", self->surf.ta);
}

static PyGetSetDef coneobj_getset[] = {
        {"_apex", (getter) coneobj_getapex, NULL, "Cone's apex", NULL},
        {"_axis", (getter) coneobj_getaxis, NULL, "Cone's axis", NULL},
        {"_t2", (getter) coneobj_getta, NULL, "Cone's angle tangent", NULL},
        {NULL}
};

static PyTypeObject ConeType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_base = &SurfaceType,
        .tp_name = "geometry.Cone",
        .tp_basicsize = sizeof(ConeObject),
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "Cone class",
        .tp_new = PyType_GenericNew,
        .tp_init = (initproc) coneobj_init,
        .tp_getset = coneobj_getset,
};

static PyObject *
torusobj_getcenter(TorusObject * self, void * closure)
{
    npy_intp dims[] = {NDIM};
    PyObject * center = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
    double * data = (double *) PyArray_DATA(center);
    for (int i = 0; i < NDIM; ++i) data[i] = self->surf.center[i];
    return center;
}

static PyObject *
torusobj_getaxis(TorusObject * self, void * closure)
{
    npy_intp dims[] = {NDIM};
    PyObject * axis = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
    double * data = (double *) PyArray_DATA(axis);
    for (int i = 0; i < NDIM; ++i) data[i] = self->surf.axis[i];
    return axis;
}

static PyGetSetDef torusobj_getset[] = {
        {"_center", (getter) torusobj_getcenter, NULL, "Torus's center", NULL},
        {"_axis", (getter) torusobj_getaxis, NULL, "Torus's axis", NULL},
        {NULL}
};

static PyMemberDef torusobj_members[] = {
        {"_R", T_DOUBLE, offsetof(TorusObject, surf) + offsetof(Torus, radius), READONLY, "Torus's major radius."},
        {"_a", T_DOUBLE, offsetof(TorusObject, surf) + offsetof(Torus, a), READONLY, "Torus's minor radius parallel to axis"},
        {"_b", T_DOUBLE, offsetof(TorusObject, surf) + offsetof(Torus, b), READONLY, "Torus's minor radius perpendicular to axis"},
        {NULL}
};

static PyTypeObject TorusType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_base = &SurfaceType,
        .tp_name = "geometry.Torus",
        .tp_basicsize = sizeof(TorusObject),
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "Torus class",
        .tp_new = PyType_GenericNew,
        .tp_init = (initproc) torusobj_init,
        .tp_getset = torusobj_getset,
        .tp_members = torusobj_members,
};

static PyObject *
gqobj_get_m(GQuadraticObject * self, void * closure)
{
    npy_intp dims[] = {NDIM, NDIM};
    PyObject * m = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
    double * data = (double *) PyArray_DATA(m);
    for (int i = 0; i < NDIM * NDIM; ++i) data[i] = self->surf.m[i];
    return m;
}

static PyObject *
gqobj_get_v(GQuadraticObject * self, void * closure)
{
    npy_intp dims[] = {NDIM};
    PyObject * v = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
    double * data = (double *) PyArray_DATA(v);
    for (int i = 0; i < NDIM; ++i) data[i] = self->surf.v[i];
    return v;
}

static PyObject *
gqobj_get_k(GQuadraticObject * self, void * closure)
{
    return Py_BuildValue("d", self->surf.k);
}

static PyGetSetDef gqobj_getset[] = {
        {"_m", (getter) gqobj_get_m, NULL, "GQuadratic's matrix.", NULL},
        {"_v", (getter) gqobj_get_v, NULL, "GQuadratic's vector.", NULL},
        {"_k", (getter) gqobj_get_k, NULL, "GQuadratic's free term", NULL},
        {NULL}
};

static PyTypeObject GQuadraticType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_base = &SurfaceType,
        .tp_name = "geometry.GQuadratic",
        .tp_basicsize = sizeof(GQuadraticObject),
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "GQuadratic class",
        .tp_new = PyType_GenericNew,
        .tp_init = (initproc) gqobj_init,
        .tp_getset = gqobj_getset,
};


// ========================================================================================== //
// ============================= Shape wrappers ============================================= //
// ========================================================================================== //

typedef struct {
    PyObject ob_base;
    Shape shape;
} ShapeObject;

static int        shapeobj_init(ShapeObject * self, PyObject * args, PyObject * kwds);
static PyObject * shapeobj_test_box(ShapeObject * self, PyObject * args, PyObject * kwds);
static PyObject * shapeobj_ultimate_test_box(ShapeObject * self, PyObject * args, PyObject * kwds);
static PyObject * shapeobj_test_points(ShapeObject * self, PyObject * points);
static PyObject * shapeobj_bounding_box(ShapeObject * self, PyObject * args);
static PyObject * shapeobj_volume(ShapeObject * self, PyObject * args, PyObject * kwds);
static PyObject * shapeobj_collect_statistics(ShapeObject * self, PyObject * args);
static PyObject * shapeobj_get_stat_table(ShapeObject * self);
static void       shapeobj_dealloc(ShapeObject * self);

static char * opcodes[] = {"I", "C", "E", "U", "S", "R"};

static PyObject *
shapeobj_getopc(ShapeObject * self, void * closure)
{
    return Py_BuildValue("s", opcodes[self->shape.opc]);
}

static PyObject *
shapeobj_getinvopc(ShapeObject * self, void * closure)
{
    return Py_BuildValue("s", opcodes[invert_opc(self->shape.opc)]);
}

static PyObject *
shapeobj_getargs(ShapeObject * self, void * closure)
{
    PyObject * args = PyTuple_New(self->shape.alen);
    if (args == NULL) return NULL;
    if (self->shape.opc == COMPLEMENT || self->shape.opc == IDENTITY) {
        PyObject * pysurf = parent_pyobject(SurfaceObject, surf, self->shape.args.surface);
        PyTuple_SET_ITEM(args, 0, pysurf);
        Py_INCREF(pysurf);
    } else if (self->shape.opc == UNION || self->shape.opc == INTERSECTION) {
        PyObject * pyshape;
        for (int i = 0; i < self->shape.alen; ++i) {
            pyshape = parent_pyobject(ShapeObject, shape, self->shape.args.shapes[i]);
            PyTuple_SET_ITEM(args, i, pyshape);
            Py_INCREF(pyshape);
        }
    }
    return args;
}

static PyGetSetDef shapeobj_getset[] = {
    {"opc", (getter) shapeobj_getopc, NULL, "Operation code of shape.", NULL},
    {"invert_opc", (getter) shapeobj_getinvopc, NULL, "Inverted operation code of shape.", NULL},
    {"args", (getter) shapeobj_getargs, NULL, "Arguments of shape.", NULL},
    {NULL}
};


static PyMethodDef shapeobj_methods[] = {
        {"test_box", (PyCFunctionWithKeywords) shapeobj_test_box, METH_VARARGS | METH_KEYWORDS, "Tests where the box is located with respect to the surface."},
        {"ultimate_test_box", (PyCFunctionWithKeywords) shapeobj_ultimate_test_box, METH_VARARGS | METH_KEYWORDS, ""},
        {"volume", (PyCFunctionWithKeywords) shapeobj_volume, METH_VARARGS | METH_KEYWORDS, ""},
        {"bounding_box", (PyCFunction) shapeobj_bounding_box, METH_VARARGS, ""},
        {"collect_statistics", (PyCFunction) shapeobj_collect_statistics, METH_VARARGS, ""},
        {"get_stat_table", (PyCFunction) shapeobj_get_stat_table, METH_NOARGS, ""},
        {"test_points", (PyCFunction) shapeobj_test_points, METH_O, "Tests senses of the points with respect to the surface."},
        {NULL}
};


static PyTypeObject ShapeType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "geometry.Shape",
        .tp_basicsize = sizeof(ShapeObject),
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "Shape class",
        .tp_new = PyType_GenericNew,
        .tp_init = (initproc) shapeobj_init,
        .tp_dealloc = (destructor) shapeobj_dealloc,
        .tp_methods = shapeobj_methods,
        .tp_getset = shapeobj_getset,
};

static int
shapeobj_init(ShapeObject * self, PyObject * args, PyObject * kwds)
{
    size_t arglen = PyTuple_Size(args);
    if (arglen < 1) {
        PyErr_SetString(PyExc_TypeError, "Operation identifier is expected.");
        return -1;
    }
    PyObject * pyopc = PyTuple_GetItem(args, 0);
    if (! PyUnicode_Check(pyopc)) {
        PyErr_SetString(PyExc_TypeError, "String object is expected.");
        return -1;
    }
    char * opcstr = PyUnicode_AS_DATA(pyopc);

    char opc;
    if      (strcmp(opcstr, opcodes[INTERSECTION]) == 0) opc = INTERSECTION;
    else if (strcmp(opcstr, opcodes[COMPLEMENT])   == 0) opc = COMPLEMENT;
    else if (strcmp(opcstr, opcodes[UNION])        == 0) opc = UNION;
    else if (strcmp(opcstr, opcodes[EMPTY])        == 0) opc = EMPTY;
    else if (strcmp(opcstr, opcodes[UNIVERSE])     == 0) opc = UNIVERSE;
    else if (strcmp(opcstr, opcodes[IDENTITY])     == 0) opc = IDENTITY;
    else {
        PyErr_SetString(PyExc_ValueError, "Unknown operation");
        return -1;
    }

    int status;
    if (opc == IDENTITY || opc == COMPLEMENT) {
        PyObject * surf = PyTuple_GetItem(args, 1);
        if (surf == NULL || ! PyObject_TypeCheck(surf, &SurfaceType)) {
            PyErr_SetString(PyExc_TypeError, "Surface instance is expected...");
            return -1;
        }
        Py_INCREF(surf);
        status = shape_init(&self->shape, opc, 1, &((SurfaceObject *) surf)->surf);
    } else if (opc == UNIVERSE || opc == EMPTY) {
        status = shape_init(&self->shape, opc, 0, NULL);
    } else {
        size_t i, j, alen = arglen - 1;
        if (alen <= 1) {
            PyErr_SetString(PyExc_ValueError, "More than one shape object is expected");
            return -1;
        }
        PyObject * item;
        Shape ** operands = (Shape **) malloc(alen * sizeof(Shape *));
        for (i = 0; i < alen; ++i) {
            item = PySequence_GetItem(args, i + 1);
            if (PyObject_TypeCheck(item, &ShapeType)) {
                operands[i] = (Shape *) &((ShapeObject *) item)->shape;
                Py_INCREF(item);
            } else {
                PyErr_SetString(PyExc_TypeError, "Shape instance is expected");
                free(operands);
                return -1;
            }
        }
        status = shape_init(&self->shape, opc, alen, operands);
        free(operands);
    }
    if (status != SHAPE_SUCCESS) return -1;
    return 0;
}

static void shapeobj_dealloc(ShapeObject * self)
{
    if (self->shape.opc == COMPLEMENT || self->shape.opc == IDENTITY) {
        PyObject * pysurf = parent_pyobject(SurfaceObject, surf, self->shape.args.surface);
        Py_DECREF(pysurf);
    } else if (self->shape.opc == UNION || self->shape.opc == INTERSECTION) {
        PyObject * pyshape;
        for (int i = 0; i < self->shape.alen; ++i) {
            pyshape = parent_pyobject(ShapeObject, shape, self->shape.args.shapes[i]);
            Py_DECREF(pyshape);
        }
    }
    shape_dealloc(&self->shape);
    Py_TYPE(self)->tp_free((PyObject*) self);
}

static PyObject *
shapeobj_test_box(ShapeObject * self, PyObject * args, PyObject * kwds)
{
    PyObject * box = NULL;
    char collect = 0;
    static char * kwlist[] = {"box", "collect", NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "Ob", kwlist, &box, &collect)) return NULL;

    if (box == NULL) box = GET_NAME(GLOBAL_BOX);

    if (! PyObject_TypeCheck(box, &BoxType)) {
        PyErr_SetString(PyExc_TypeError, "Box instance is expected...");
        return NULL;
    }

    int result = shape_test_box(&self->shape, &((BoxObject *) box)->box, collect);
    return Py_BuildValue("i", result);
}

static PyObject *
shapeobj_ultimate_test_box(ShapeObject * self, PyObject * args, PyObject * kwds)
{
    PyObject * box = NULL;
    char collect = 0;
    double min_vol = MIN_VOLUME;

    static char * kwlist[] = {"box", "min_volume", "collect", NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|Odb", kwlist, &box, &min_vol, &collect)) return NULL;

    if (box == NULL) box = GET_NAME(GLOBAL_BOX);

    if (! PyObject_TypeCheck(box, &BoxType)) {
        PyErr_SetString(PyExc_ValueError, "Box instance is expected");
        return NULL;
    }

    int result = shape_ultimate_test_box(&self->shape, &((BoxObject *) box)->box, min_vol, collect);
    return Py_BuildValue("i", result);
}

static PyObject *
shapeobj_test_points(ShapeObject * self, PyObject * points)
{
    PyObject * pts;
    if (! convert_to_dbl_vec_array(points, &pts)) return NULL;

    npy_intp size = PyArray_SIZE((PyArrayObject *) pts);
    size_t npts = size > NDIM ? PyArray_DIM((PyArrayObject *) pts, 0) : 1;
    npy_intp dims[] = {npts};
    PyObject * result = PyArray_EMPTY(1, dims, NPY_BYTE, 0);
    if (result == NULL) {
        Py_DECREF(pts);
        return NULL;
    }

    shape_test_points(&self->shape, npts, (double *) PyArray_DATA(pts), \
                                          (char *) PyArray_DATA(result));
    Py_DECREF(pts);
    return result;
}

static PyObject *
shapeobj_bounding_box(ShapeObject * self, PyObject * args)
{
    PyObject * start_box;
    double tol;
    if (! PyArg_ParseTuple(args, "Od", &start_box, &tol)) return NULL;

    if (! PyObject_TypeCheck(start_box, &BoxType)) {
        PyErr_SetString(PyExc_ValueError, "Box instance is expected");
        return NULL;
    }

    BoxObject * box = (BoxObject *) boxobj_copy((BoxObject *) start_box);
    if (box == NULL) return NULL;

    int status = shape_bounding_box(&self->shape, &box->box, tol);

    if (status == SHAPE_SUCCESS) return (PyObject *) box;
    else {
        Py_DECREF(box);
        // TODO: Probably some exception should be raised.
        return NULL;
    }
}

static PyObject *
shapeobj_volume(ShapeObject * self, PyObject * args, PyObject * kwds)
{
    PyObject * box = NULL;
    double min_vol = MIN_VOLUME;

    static char * kwlist[] = {"box", "min_volume", NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|Od", kwlist, &box, &min_vol)) return NULL;

    if (box == NULL) box = GET_NAME(GLOBAL_BOX);

    if (! PyObject_TypeCheck(box, &BoxType)) {
        PyErr_SetString(PyExc_ValueError, "Box instance is expected");
        return NULL;
    }

    double vol = shape_volume(&self->shape, &((BoxObject *) box)->box, min_vol);
    return Py_BuildValue("d", vol);
}

static PyObject *
shapeobj_collect_statistics(ShapeObject * self, PyObject * args)
{
    PyObject * box;
    double min_vol;
    if (! PyArg_ParseTuple(args, "Od", &box, &min_vol)) return NULL;

    if (! PyObject_TypeCheck(box, &BoxType)) {
        PyErr_SetString(PyExc_ValueError, "Box instance is expected");
        return NULL;
    }

    shape_collect_statistics(&self->shape, &((BoxObject *) box)->box, min_vol);
    Py_RETURN_NONE;
}

static PyObject *
shapeobj_get_stat_table(ShapeObject * self)
{
    size_t nrows = 0, ncols = 0;
    char * table_data = shape_get_stat_table(&self->shape, &nrows, &ncols);
    npy_intp dims[] = {nrows, ncols};
    PyObject * table = PyArray_SimpleNewFromData(2, dims, NPY_BYTE, table_data);
    return table;
}

// ========================================================================================== //
// =================================== Module =============================================== //
// ========================================================================================== //

static PyModuleDef geometry_module = {
        PyModuleDef_HEAD_INIT,
        "geometry",
        "Geometry native objects.",
        -1,
        NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_geometry(void)
{
    PyObject* m;

    if (PyType_Ready(&BoxType) < 0) return NULL;

    if (PyType_Ready(&SurfaceType) < 0) return NULL;
    if (PyType_Ready(&PlaneType) < 0) return NULL;
    if (PyType_Ready(&SphereType) < 0) return NULL;
    if (PyType_Ready(&CylinderType) < 0) return NULL;
    if (PyType_Ready(&ConeType) < 0) return NULL;
    if (PyType_Ready(&TorusType) < 0) return NULL;
    if (PyType_Ready(&GQuadraticType) < 0) return NULL;

    if (PyType_Ready(&ShapeType) < 0) return NULL;

    m = PyModule_Create(&geometry_module);
    if (m == NULL)
        return NULL;
    import_array();

    Py_INCREF(&BoxType);

    Py_INCREF(&SphereType);
    Py_INCREF(&PlaneType);
    Py_INCREF(&SphereType);
    Py_INCREF(&CylinderType);
    Py_INCREF(&ConeType);
    Py_INCREF(&TorusType);
    Py_INCREF(&GQuadraticType);

    Py_INCREF(&ShapeType);


    PyModule_AddObject(m, "Box", (PyObject *) &BoxType);

    PyModule_AddObject(m, "Surface",    (PyObject *) &SurfaceType);
    PyModule_AddObject(m, "Plane",      (PyObject *) &PlaneType);
    PyModule_AddObject(m, "Sphere",     (PyObject *) &SphereType);
    PyModule_AddObject(m, "Cylinder",   (PyObject *) &CylinderType);
    PyModule_AddObject(m, "Cone",       (PyObject *) &ConeType);
    PyModule_AddObject(m, "Torus",      (PyObject *) &TorusType);
    PyModule_AddObject(m, "GQuadratic", (PyObject *) &GQuadraticType);

    PyModule_AddObject(m, "Shape", (PyObject *) &ShapeType);

    // Create Module constants

    PyObject *ex, *ey, *ez, *origin;
    BoxObject * global_box;
    npy_intp dims[] = {NDIM};

    origin = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    ex = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    ey = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    ez = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);

    *((double *) PyArray_DATA(ex) + 0) = 1.0;
    *((double *) PyArray_DATA(ey) + 1) = 1.0;
    *((double *) PyArray_DATA(ez) + 2) = 1.0;

    global_box = (BoxObject *) PyType_GenericNew(&BoxType, NULL, NULL);
    box_init(&global_box->box, (double *) PyArray_DATA(origin), (double *) PyArray_DATA(ex),
                               (double *) PyArray_DATA(ey),     (double *) PyArray_DATA(ez),
                               MAX_DIM, MAX_DIM, MAX_DIM);

    PyModule_AddObject(m, ORIGIN, (PyObject *) origin);
    PyModule_AddObject(m, EX, (PyObject *) ex);
    PyModule_AddObject(m, EY, (PyObject *) ey);
    PyModule_AddObject(m, EZ, (PyObject *) ez);
    PyModule_AddObject(m, GLOBAL_BOX, (PyObject *) global_box);
    PyModule_AddObject(m, MIN_VOLUME_NAME, Py_BuildValue("d", MIN_VOLUME));

    module_dict = PyModule_GetDict(m);

    return m;
}

