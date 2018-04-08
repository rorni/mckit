//
// Created by Roma on 08.04.2018.
//
#include <Python.h>
#include <stdlib.h>
#include <string.h>
#include "geometry_.h"
#include "surface_.h"
#include "box_.h"
#include "../src/geometry.h"

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
