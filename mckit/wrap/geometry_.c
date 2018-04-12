//
// Created by Roma on 08.04.2018.
//
#include <Python.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include "geometry_.h"
#include "surface_.h"
#include "common_.h"
#include "box_.h"
#include "../src/geometry.h"
#include "../src/rbtree.h"
#include "../src/surface.h"

typedef struct {
    PyObject ob_base;
    Node * node;
} NodeObject;

static int
get_operation_code(PyObject * kwds)
{
    PyObject * args = Py_BuildValue("()");
    char * kwlist[] = {"opc", NULL};
    char * opc = NULL;
    if (! PyArg_ParseTupleAndKeywords(args, kwds, "z", kwlist, &opc)) {
        Py_DECREF(args);
        free(opc);
        return -1;
    }
    Py_DECREF(args);

    int opcode;
    if (opc == NULL) opcode = IDENTITY;
    else if (strcmp(opc, "C") == 0 || strcmp(opc, "complement") == 0) opcode = COMPLEMENT;
    else if (strcmp(opc, "I") == 0 || strcmp(opc, "intersection") == 0) opcode = INTERSECTION;
    else if (strcmp(opc, "U") == 0 || strcmp(opc, "union") == 0) opcode = UNION;
    else if (strcmp(opc, "E") == 0 || strcmp(opc, "empty") == 0) opcode = EMPTY;
    else if (strcmp(opc, "R") == 0 || strcmp(opc, "universe") == 0) opcode = UNIVERSE;
    else {
        PyErr_SetString(PyExc_ValueError, "Unknown operation code");
        free(opc);
        return -1;
    }
    free(opc);
    return opcode;
}

/* Keyword argument: opc - operation code. It is string.
 * args - list of arguments - SurfaceObject or NodeObjects.
 *
 * Possible variants:
 *
 * 1. opc == C (complement) or None + SurfaceObject (len == 1)
 * 2. opc == C (complement) or None + NodeObject    (len == 1)
 * 3. opc == I (intersection) or U (union) + array of NodeObjects
 * 4. opc == E (empty) + (len == 0) - empty set
 * 5. opc == R (universe) + (len == 0) - whole space.
 */
static int
nodeobj_init(NodeObject * self, PyObject * args, PyObject * kwds)
{
    size_t len = PyTuple_Size(args);
    void * arguments = NULL;

    int opc = get_operation_code(kwds);
    if (opc < 0) return -1;

    if (len == 0) {
        if (opc != EMPTY && opc != UNIVERSE) {
            PyErr_SetString(PyExc_ValueError, "Arguments are expected");
            return -1;
        }
    } else if (len == 1) {
        if (opc == EMPTY || opc == UNIVERSE) {
            PyErr_SetString(PyExc_ValueError, "Wrong operation type");
            return -1;
        }
        PyObject * arg = PyTuple_GetItem(args, 0);
        if (PyObject_TypeCheck(arg, &SurfaceType)) arguments = ((SurfaceObject *) arg)->surf;
        else if (PyObject_TypeCheck(arg, &NodeType)) arguments = ((NodeObject *) arg)->node;
        else {
            PyErr_SetString(PyExc_TypeError, "Node or Surface instance is expected");
            return -1;
        }
    } else {
        if (opc != INTERSECTION && opc != UNION) {
            PyErr_SetString(PyExc_ValueError, "Wrong operation type");
            return -1;
        }
        arguments = malloc(len * sizeof(Node *));
        size_t i;
        PyObject * arg;
        for (i = 0; i < len; ++i) {
            arg = PyTuple_GetItem(args, i);
            if (PyObject_TypeCheck(arg, &NodeType)) arguments[i] = ((NodeObject *) arg)->node;
            else {
                PyErr_SetString(PyExc_TypeError, "Node instance is expected");
                free(arguments);
                return -1;
            }
        }
    }

    self->node = node_create(opc, len, arguments);
    free(arguments);
    return 0;
}

static void
nodeobj_dealloc(NodeObject * self)
{
    node_free(self->node);
    Py_TYPE(self)->tp_free((PyObject*) self);
}

static PyObject *
nodeobj_test_box(NodeObject * self, PyObject * args, PyObject * kwds)
{
    PyObject * pybox;
    char collect = 0;
    char * kwlist[] = {"", "collect", NULL};
    if (! PyArg_ParseTuple(args, kwds, "O|i", kwlist, &pybox, &collect)) return NULL;
    if (! PyObject_TypeCheck(pybox, &BoxType)) {
        PyErr_SetString(PyExc_ValueError, "Box instance is expected");
        return NULL;
    }
    BoxObject * box = (BoxObject *) pybox;
    int result = node_test_box(self->node, &box->box, collect);
    return Py_Build_Value("i", result);
}

static PyObject *
nodeobj_test_points(NodeObject * self, PyObject * points)
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

    node_test_points(self->node, (double *) PyArray_DATA(pts), npts, \
                   (int *) PyArray_DATA(result));
    Py_DECREF(pts);
    return result;
}

static PyObject *
nodeobj_complexity(NodeObject * self)
{
    int result = node_complexity(self->node);
    return Py_BuildValue("i", result);
}

static PyObject *
nodeobj_get_surfaces(NodeObject * self)
{
    RBTree * rbt = rbtree_create(surface_compare);
    node_get_surfaces(self->node, rbt);
    PyObject * set = PySet_New(NULL);
    if (set != NULL) {
        Surface * arr = rbtree_to_array(rbt);
        int i, status;
        PyObject * pysurf;
        for (i = 0; i < rbt->len; ++i) {
            pysurf = parent_pyobject(SurfaceObject, surf, arr + i);
            status = PySet_Add(set, pysurf);
        }
        free(arr);
    }
    rbtree_free(rbt);
    return set;
}

static PyObject *
nodeobj_bounding_box(NodeObject * self, PyObject * args, PyObject * kwds)
{
    PyObject * pybox;
    double tol;
    if (! PyArg_ParseTuple(args, "Od", &pybox, tol)) return NULL;
    if (! PyObject_TypeCheck(pybox, &BoxType)) {
        PyErr_SetString(PyExc_ValueError, "Box instance is expected");
        return NULL;
    }
    BoxObject * box = (BoxObject *) PyType_GenericNew(&BoxType, NULL, NULL);
    if (box == NULL) return NULL;
    box_copy(&((BoxObject *) pybox)->box, &box->box);
    int status = node_bounding_box(self->node, &box->box, tol);
    if (status != NODE_SUCCESS) {
        Py_DECREF(box);
        box = NULL;
    }
    return (PyObject *) box;
}

static PyObject *
nodeobj_volume(NodeObject * self, PyObject * args, PyObject * kwds)
{
    PyObject * pybox;
    double min_vol;
    if (! PyArg_ParseTuple(args, "Od", &pybox, min_vol)) return NULL;
    if (! PyObject_TypeCheck(pybox, &BoxType)) {
        PyErr_SetString(PyExc_ValueError, "Box instance is expected");
        return NULL;
    }
    double vol = node_volume(self->node, &((BoxObject *) pybox)->box, min_vol);
    return Py_BuildValue("d", vol);
}

static PyObject *
nodeobj_collect_statistics(NodeObject * self, PyObject * args, PyObject * kwds)
{
    PyObject * pybox;
    double min_vol;
    if (! PyArg_ParseTuple(args, "Od", &pybox, min_vol)) return NULL;
    if (! PyObject_TypeCheck(pybox, &BoxType)) {
        PyErr_SetString(PyExc_ValueError, "Box instance is expected");
        return NULL;
    }
    node_collect_stat(self->node, &((BoxObject *) pybox)->box, min_vol);
    Py_RETURN_NONE;
}

static PyObject *
nodeobj_get_simplest(NodeObject * self, PyObject * args, PyObject * kwds)
{
    // void node_get_simplest(Node * node);
    PyErr_SetString(PyExc_NotImplementedError, "Not Implemented");
    return NULL;
}

static PyObject *
nodeobj_complement(NodeObject * self)
{
    NodeObject * result = PyType_GenericNew(NodeType, NULL, NULL);
    if (result != NULL)
        result->node = node_complement(self->node);
    return (PyObject *) result;
}

static PyObject *
nodeobj_intersection(NodeObject * self, PyObject * other)
{
    if (! PyObject_TypeCheck(ohter, &NodeType)) {
        PyErr_SetString(PyExc_TypeError, "Node instance is expected.");
        return NULL;
    }
    NodeObject * result = PyType_GenericNew(NodeType, NULL, NULL);
    if (result != NULL) result->node = node_intersection(self->node, ((NodeObject *) other)->node);
    return (PyObject *) result;
}

static PyObject *
nodeobj_union(NodeObject * self, PyObject * other)
{
    if (! PyObject_TypeCheck(ohter, &NodeType)) {
        PyErr_SetString(PyExc_TypeError, "Node instance is expected.");
        return NULL;
    }
    NodeObject * result = PyType_GenericNew(NodeType, NULL, NULL);
    if (result != NULL) result->node = node_union(self->node, ((NodeObject *) other)->node);
    return (PyObject *) result;
}

static PyMethodDef nodeobj_methods[] = {
        {"union", (PyCFunction) nodeobj_union, METH_O, "Makes an union of two geometries."},
        {"intersection", (PyCFunction) nodeobj_intersection, METH_O, "Makes an intersection of two geometries."},
        {"complement", (PyCFunction) nodeobj_complement, METH_NOARGS, "Gets a geometry complement."},
        {"test_box", (PyCFunctionWithKeywords) boxobj_split, METH_VARARGS | METH_KEYWORDS, "Tests box."},
        {"test_points", (PyCFunction) nodeobj_test_points, METH_O, "Tests points' senses."},
        {"complexity", (PyCFunction) nodeobj_complexity, METH_NOARGS, "Complexity of the node."},
        {"get_surfaces", (PyCFunction) nodeobj_get_surfaces, METH_NOARGS, "Gets a set of unique surfaces."},
        {"bounding_box", (PyCFunction) nodeobj_bounding_box, METH_VARARGS, "Gets bounding box of the geometry."},
        {"volume", (PyCFunction) nodeobj_volume, METH_VARARGS, "Gets volume of the cell."},
        {"collect_statistics", (PyCFunction) nodeobj_collect_statistics, METH_VARARGS, "Collects statistics."},
        {"get_simplest", (PyCFunction) nodeobj_get_simplest, METH_VARARGS, "Gets the simplest form of the geometry"},
        {NULL}
};

PyTypeObject NodeType = {
        PyObject_HEAD_INIT(NULL)
        .tp_name = "geometry.Node",
        .tp_basicsize = sizeof(NodeObject),
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "Node class",
        .tp_new = PyType_GenericNew,
        .tp_dealloc = (destructor) nodeobj_dealloc,
        .tp_init = (initproc) nodeobj_init,
        .tp_methods = nodeobj_methods,
};
