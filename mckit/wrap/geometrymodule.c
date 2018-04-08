#include <Python.h>
#include <structmember.h>

#include "box_.h"
#include "surface_.h"

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
    
    if (PyType_Ready(&SurfaceType) < 0) return NULL;
    if (PyType_Ready(&PlaneType) < 0) return NULL;
    if (PyType_Ready(&SphereType) < 0) return NULL;
    if (PyType_Ready(&CylinderType) < 0) return NULL;
    if (PyType_Ready(&ConeType) < 0) return NULL;
    if (PyType_Ready(&TorusType) < 0) return NULL;
    if (PyType_Ready(&GQuadraticType) < 0) return NULL;

    m = PyModule_Create(&geometry_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&BoxType);

    Py_INCREF(&SphereType);
    Py_INCREF(&PlaneType);
    Py_INCREF(&SphereType);
    Py_INCREF(&CylinderType);
    Py_INCREF(&ConeType);
    Py_INCREF(&TorusType);
    Py_INCREF(&GQuadraticType);

    PyModule_AddObject(m, "Box", (PyObject *) &BoxType);

    PyModule_AddObject(m, "Surface", (PyObject *) &SurfaceType);
    PyModule_AddObject(m, "Plane", (PyObject *) &PlaneType);
    PyModule_AddObject(m, "Sphere", (PyObject *) &SphereType);
    PyModule_AddObject(m, "Cylinder", (PyObject *) &CylinderType);
    PyModule_AddObject(m, "Cone", (PyObject *) &ConeType);
    PyModule_AddObject(m, "Torus", (PyObject *) &TorusType);
    PyModule_AddObject(m, "GQuadratic", (PyObject *) &GQuadraticType);

    return m;
}

