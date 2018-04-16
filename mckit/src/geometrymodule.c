#include <Python.h>
#include <structmember.h>
#include <string.h>

#include "numpy/arrayobject.h"

#include "box.h"
#include "surface.h"
#include "shape.h"

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

static PyMethodDef boxobj_methods[] = {
        {"copy", (PyCFunction) boxobj_copy, METH_NOARGS, "Makes a copy of the box."},
        {"generate_random_points", (PyCFunction) boxobj_generate_random_points, METH_O,
                                                         "Generate N random points inside the box."},
        {"test_points", (PyCFunction) boxobj_test_points, METH_O, "Tests points if they are inside the box."},
        {"split", (PyCFunctionWithKeywords) boxobj_split, METH_VARARGS | METH_KEYWORDS,
                "Splits the box into two smaller."},
        {NULL}
};

static PyTypeObject BoxType = {
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

static int
boxobj_init(BoxObject * self, PyObject * args, PyObject * kwds)
{
    PyObject *cent, *ex, *ey, *ez;
    double xdim, ydim, zdim;

    if (! PyArg_ParseTuple(args, "O&O&O&O&ddd", convert_to_dbl_vec, &cent,
                           convert_to_dbl_vec, &ex, convert_to_dbl_vec, &ey, convert_to_dbl_vec, &ez,
                           &xdim ,&ydim, &zdim))
        return -1;

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
        {"test_box", (PyCFunction) surfobj_test_box, METH_O, "Tests where the box is located with respect to the surface."},
        {"test_points", (PyCFunction) surfobj_test_points, METH_O, "Tests senses of the points with respect to the surface."},
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

    cone_init(&self->surf, (double *) PyArray_DATA(apex), (double *) PyArray_DATA(axis), ta);
    Py_DECREF(apex);
    Py_DECREF(axis);
    return 0;
}

static int
torusobj_init(TorusObject * self, PyObject * args, PyObject * kwds)
{
    PyObject *center, *axis;
    double r, a, b;
    if (! PyArg_ParseTuple(args, "O&O&d", convert_to_dbl_vec, &center, convert_to_dbl_vec, &axis, &r, &a, &b)) return -1;

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

static PyTypeObject PlaneType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_base = &SurfaceType,
        .tp_name = "geometry.Plane",
        .tp_basicsize = sizeof(PlaneObject),
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "Plane class",
        .tp_new = PyType_GenericNew,
        .tp_init = (initproc) planeobj_init,
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
};


// ========================================================================================== //
// ============================= Shape wrappers ============================================= //
// ========================================================================================== //

typedef struct {
    PyObject ob_base;
    Shape shape;
} ShapeObject;

static int shapeobj_init(ShapeObject * self, PyObject * args, PyObject * kwds);
static PyObject * shapeobj_test_box(ShapeObject * self, PyObject * args);
static PyObject * shapeobj_ultimate_test_box(ShapeObject * self, PyObject * args);
static PyObject * shapeobj_test_points(ShapeObject * self, PyObject * points);
static PyObject * shapeobj_bounding_box(ShapeObject * self, PyObject * args);
static PyObject * shapeobj_volume(ShapeObject * self, PyObject * args);
static PyObject * shapeobj_collect_statistics(ShapeObject * self, PyObject * args);
static PyObject * shapeobj_get_stat_table(ShapeObject * self);



static PyMethodDef shapeobj_methods[] = {
        {"test_box", (PyCFunction) surfobj_test_box, METH_VARARGS, "Tests where the box is located with respect to the surface."},
        {"ultimate_test_box", (PyCFunction) shapeobj_ultimate_test_box, METH_VARARGS, ""},
        {"volume", (PyCFunction) shapeobj_volume, METH_VARARGS, ""},
        {"bounding_box", (PyCFunction) shapeobj_bounding_box, METH_VARARGS, ""},
        {"collect_statistics", (PyCFunction) shapeobj_collect_statistics, METH_VARARGS, ""},
        {"get_stat_table", (PyCFunction) shapeobj_get_stat_table, METH_NOARGS, ""},
        {"test_points", (PyCFunction) shapeobj_test_points, METH_O, "Tests senses of the points with respect to the surface."},
        {NULL}
};

static PyTypeObject ShapeType = {
        PyVarObject_HEAD_INIT(NULL, 0)
                .tp_name = "geometry.Shape",
        .tp_basicsize = sizeof(Shape),
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "Shape class",
        .tp_new = PyType_GenericNew,
        .tp_init = (initproc) shapeobj_init,
        .tp_methods = shapeobj_methods,
};

static int
shapeobj_init(ShapeObject * self, PyObject * args, PyObject * kwds)
{
    char * opcstr;
    PyObject * arglist;
    if (! PyArg_ParseTuple(args, "sO", &opcstr, &arglist)) return -1;

    if (! PySequence_Check(arglist)) {
        PyErr_SetString(PyExc_TypeError, "Sequence instance is expected");
        return -1;
    }

    char opc;
    if (strcmp(opcstr, "I") == 0) opc = INTERSECTION;
    else if (strcmp(opcstr, "C") == 0) opc = COMPLEMENT;
    else if (strcmp(opcstr, "U") == 0) opc = UNION;
    else if (strcmp(opcstr, "empty") == 0) opc = EMPTY;
    else if (strcmp(opcstr, "universe") == 0) opc = UNIVERSE;
    else if (strcmp(opcstr, "identity") == 0) opc = IDENTITY;
    else {
        PyErr_SetString(PyExc_ValueError, "Unknown operation");
        free(opcstr);
        return -1;
    }

    size_t i, alen = PySequence_Size(arglist);
    if (alen < 0) return -1;

    void ** arguments = NULL;

    if (alen > 0) {
        arguments = malloc(alen * sizeof(void *));
        PyObject *item;
        for (i = 0; i < alen; ++i) {
            item = PySequence_Fast_GET_ITEM(arglist, i);
            if (PyObject_TypeCheck(item, &SurfaceType)) {
                *(arguments + i) = &((SurfaceObject *) item)->surf;
            } else if (PyObject_TypeCheck(item, &ShapeType)) {
                *(arguments + i) = &((ShapeObject *) item)->shape;
            } else {
                PyErr_SetString(PyExc_TypeError, "Shape instance is expected");
                free(arguments);
                return -1;
            }
        }
    }
    int status = shape_init(&self->shape, opc, alen, arguments);
    free(arguments);
    if (status != SHAPE_SUCCESS) return -1;
    return 0;
}

static PyObject *
shapeobj_test_box(ShapeObject * self, PyObject * args)
{
    PyObject * box;
    char collect;
    if (! PyArg_ParseTuple(args, "Oc", &box, &collect)) return NULL;

    if (! PyObject_TypeCheck(box, &BoxType)) {
        PyErr_SetString(PyExc_ValueError, "Box instance is expected");
        return NULL;
    }

    int result = shape_test_box(&self->shape, &((BoxObject *) box)->box, collect);
    return Py_BuildValue("i", result);
}

static PyObject *
shapeobj_ultimate_test_box(ShapeObject * self, PyObject * args)
{
    PyObject * box;
    char collect;
    double min_vol;
    if (! PyArg_ParseTuple(args, "Odc", &box, &min_vol, &collect)) return NULL;

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
shapeobj_volume(ShapeObject * self, PyObject * args)
{
    PyObject * box;
    double min_vol;
    if (! PyArg_ParseTuple(args, "Od", &box, &min_vol)) return NULL;

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
}

static PyObject *
shapeobj_get_stat_table(ShapeObject * self)
{
    size_t nrows, ncols;
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

    PyModule_AddObject(m, "Surface", (PyObject *) &SurfaceType);
    PyModule_AddObject(m, "Plane", (PyObject *) &PlaneType);
    PyModule_AddObject(m, "Sphere", (PyObject *) &SphereType);
    PyModule_AddObject(m, "Cylinder", (PyObject *) &CylinderType);
    PyModule_AddObject(m, "Cone", (PyObject *) &ConeType);
    PyModule_AddObject(m, "Torus", (PyObject *) &TorusType);
    PyModule_AddObject(m, "GQuadratic", (PyObject *) &GQuadraticType);

    PyModule_AddObject(m, "Shape", (PyObject *) &ShapeType);

    return m;
}

