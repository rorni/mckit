#ifndef __SURFACE
#define __SURFACE

typedef struct Surface Surface;
typedef struct Plane Plane;

struct Plane {
    double v[NDIM];
    double k;
};

struct Sphere {
    double center[NDIM];
    double radius;
};

struct Cylinder {
    double point[NDIM];
    double axis[NDIM];
    double radius;
};

struct Cone {
    double vertex[NDIM];
    double axis[NDIM];
    double t2;
};

struct Surface {
    
};

#endif