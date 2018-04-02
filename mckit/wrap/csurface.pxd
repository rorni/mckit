from cbox cimport Box
cdef extern from "../src/surface.h":

    ctypedef struct Surface:
        pass
        
    ctypedef struct Plane:
        pass

    ctypedef struct Sphere:
        pass

    ctypedef struct Cylinder:
        pass

    ctypedef struct Cone:
        pass

    ctypedef struct Torus:
        pass

    ctypedef struct GQuadratic:
        pass
       
    int plane_init(Surface * surf, unsigned int name, \
        int modifier, const double * norm, double offset)
    
    int sphere_init(Surface * surf, unsigned int name, int modifier, \
        const double * center, double radius)
    
    int cylinder_init(Surface * surf, unsigned int name, int modifier, \
        const double * point, const double * axis, double radius)

    int cone_init(Surface * surf, unsigned int name, int modifier, \
        const double * apex, const double * axis, double ta)
    
    int torus_init(Surface * surf, unsigned int name, int modifier, \
        const double * center, const double * axis, double radius, \
        double a, double b)

    int gq_init(Surface * surf, unsigned int name, int modifier,
        const double * m, const double * v, double k)
    
    void surface_dispose(Surface * surf)
    
    void surface_test_points(const Surface * surf, const double * points, \
        int npts, int * result)
    
    int surface_test_box(const Surface * surf, const Box * box)
