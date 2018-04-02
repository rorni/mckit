#include "box_wrap.h"
#include "numpy/arrayobject.h"

static void boxobj_dealloc(BoxObject * self)
{
    box_dispose(self->box);
    free(self->box);
    Py_TYPE(self)->tp_free((PyObject*) self);
}

static PyObject * 
boxobj_new(PyTypeObject * type, PyObject * args, PyObject * kwds)
{
    BoxObject * self;
    
    self = (BoxObject *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->box = (Box *) malloc(sizeof(Box));
        if (self->box == NULL) {
            PyDECREF(self);
            return NULL;
        }
    }
    
    return (PyObject *) self;
}

static PyArrayObject *
convert_2C_double_array(PyObject * obj, size_t nrows)
{
    PyArrayObject * arr = PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr == NULL) return NULL;
    
    int nd = PyArray_NDIM(arr);
    int rows = (nd == 2) ? PyArray_DIM(arr, 0) : 1;
    int cols = nd > 0 ? PyArray_DIM(arr, -1) : 0;
    if (nd == 0 || nd > 2 || nrows > 0 && rows != nrows || ncols != NDIM) {
        PyErr_SetString(PyExc_ValueError, "Wrong dimensions");
        PyDECREF(arr);
        return NULL;
    }
    return arr;    
}

static int
boxobj_init(BoxObject * self, PyObject * args, PyObject * kwds)
{
    PyObject *cent, *ex, *ey, *ez;
    double xdim, ydim, zdim;
    
    if (! PyArg_ParseTuple(args, "OdddOOO",
                        &cent, &xdim ,&ydim, &zdim, &ex, &ey, &ez))
        return -1;
      
    cent = convert_2C_double_array(cent, 1);
    if (cent == NULL) return -1;
    
    ex = convert_2C_double_array(ex, 1);
    if (ex == NULL) goto error1;       
                                     
    ey = convert_2C_double_array(ey, 1);
    if (ey == NULL) goto error2;       
                                     
    ez = convert_2C_double_array(ez, 1);
    if (ez == NULL) goto error3;
    
    box_dispose(&self->box);
    box_init(&self->box, (double *) PyArray_DATA(cent), 
                         (double *) PyArray_DATA(ex), 
                         (double *) PyArray_DATA(ey), 
                         (double *) PyArray_DATA(ez), 
             xdim, ydim, zdim);
    
    PyDECREF(cent);
    PyDECREF(ex);
    PyDECREF(ey);
    PyDECREF(ez);
    
    return 0;
    
  error3:
    PyDECREF(ey);
  error2:
    PyDECREF(ex);
  error1:
    PyDECREF(cent);
    return -1;
}

static PyObject *
boxobj_copy(BoxObject * self)
{
    PyObject * box = boxobj_new(BoxType, NULL, NULL);
    if (box == NULL) return NULL;
    double *c = self->box->center;
    double *ex = self->box->ex, *ey = self->box->ey, *ez = self->box->ez;
    double xdim = self->box->dims[0], ydim = self->box->dims[1], zdim = self->box->dims[2];
    box_init(box->box, c, ex, ey, ez, xdim, ydim, zdim);
    return box;
}

static PyObject *
boxobj_generate_random_points(BoxObject * self, PyObject * args)
{
    size_t npts;
    if (! PyArg_ParseTuple(args, "I", &npts)) return -1;
    int dims[] = {npts, NDIM};
    PyArrayObject * points = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
    if (points == NULL) return NULL;
    
    int status = box_generate_random_points(self->box, \
                                (double *) PyArray_DATA(points), npts);
    if (status == BOX_FAILURE) {
        PyErr_SetString(PyExc_MemoryError, "Could not generate points.");
        PyDECREF(points);
        points = NULL;
    }
    return points;
}

static PyObject *
boxobj_test_points(BoxObject * self, PyObject * args)
{
    PyObject * points;
    if (! PyArg_ParseTuple(args, "O", &points)) return -1;
    points = convert_2C_double_array(points, 0);
    if (points == NULL) return NULL;
    
    size_t npts = PyArray_DIM(points, 0);
    int dims[] = {npts};
    PyArrayObject * result = PyArray_EMPTY(1, dims, NPY_INT, 0);
    if (result == NULL) {
        PyDECREF(points);
        return NULL;
    }
    
    box_test_points(self->box, (double *) PyArray_DATA(points), npts, \
                   (int *) PyArray_DATA(result));
    return result;
}

static PyObject *
boxobj_split(BoxObject * self, PyObject * args)
{
    
}
