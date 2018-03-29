#ifndef __SURFACE_H
#define __SURFACE_H

#include <stddef.h>
#include <stdint.h>
#include "common.h"
#include "box.h"

#define SURFACE_SUCCESS  0
#define SURFACE_FAILURE -1

typedef struct Surface      Surface;
typedef struct Plane        Plane;
typedef struct Sphere       Sphere;
typedef struct Cylinder     Cylinder;
typedef struct Cone         Cone;
typedef struct Torus        Torus;
typedef struct GQuadratic   GQuadratic;

typedef struct Modifier Modifier;
typedef struct SurfType SurfType;

enum SurfType {PLANE=1, SPHERE, CYLINDER, CONE, TORUS, GQUADRATIC};
enum Modifier {ORDINARY, REFLECTIVE, WHITE};

struct Surface {
    unsigned int name;
    Modifier modifier;
    SurfType type;
    uint64_t hash;
    const Box * last_box;
    int last_box_result;
};

struct Plane {
    Surface base;
    double norm[NDIM];
    double offset;
};

struct Sphere {
    Surface base;
    double center[NDIM];
    double radius;
};

struct Cylinder {
    Surface base;
    double point[NDIM];
    double axis[NDIM];
    double radius;
};

struct Cone {
    Surface base;
    double apex[NDIM];
    double axis[NDIM];
    double ta;
};

struct Torus {
    Surface base;
    double center[NDIM];
    double axis[NDIM];
    double radius;
    double a;
    double b;
    double * specpts;
};

struct GQuadratic {
    Surface base;
    double m[NDIM * NDIM];
    double v[NDIM];
    double k;
};

// Methods //

int plane_init(
    Plane * surf,
    unsigned int name,
    Modifier modifier,
    const double * norm,
    double offset
);

int sphere_init(
    Sphere * surf,
    unsigned int name,
    Modifier modifier,
    const double * center,
    double radius
);

int cylinder_init(
    Cylinder * surf,
    unsigned int name,
    Modifier modifier,
    const double * point,
    const double * axis,
    double radius
);

int cone_init(
    Cone * surf,
    unsigned int name,
    Modifier modifier,
    const double * apex,
    const double * axis,
    double ta
);

int torus_init(
    Torus * surf,
    unsigned int name,
    Modifier modifier,
    const double * center,
    const double * axis,
    double radius,
    double a,
    double b
);

int gq_init(
    GQuadratic * surf,
    unsigned int name,
    Modifier modifier,
    const double * m,
    const double * v,
    double k
);

void surface_dispose(Surface * surf);

void torus_dispose(Torus * surf);

void surface_test_points(
    const Surface * surf, 
    const double * points, 
    int npts,
    int * result
);

int surface_test_box(
    const Surface * surf, 
    const Box * box
);

#endif