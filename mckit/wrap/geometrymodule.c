#include <Python.h>
#include <structmember.h>

extern PyTypeObject BoxType;

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

    if (PyType_Ready(&BoxType) < 0)
        return NULL;

    m = PyModule_Create(&geometry_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&BoxType);
    PyModule_AddObject(m, "Box", (PyObject *) &BoxType);
    return m;
}

