#ifndef __SURFACE_H
#define __SURFACE_H

#include "common.h"
#include "box.h"

#define SURFACE_SUCCESS  0
#define SURFACE_FAILURE -1

typedef struct Surface Surface;

enum SurfType {PLANE=1, SPHERE, CYLINDER, CONE, TORUS, GQUADRATIC};
enum Modifier {ORDINARY, REFLECTIVE, WHITE};

struct Surface {
    unsigned int name;
    enum SurfType type;
    enum Modifier modifier;
    void * data;
};

int plane_init(
    Surface * surf,
    unsigned int name,
    enum Modifier modifier,
    const double * norm,
    double offset
);

int sphere_init(
    Surface * surf,
    unsigned int name,
    enum Modifier modifier,
    const double * center,
    double radius
);

int cylinder_init(
    Surface * surf,
    unsigned int name,
    enum Modifier modifier,
    const double * point,
    const double * axis,
    double radius
);

int cone_init(
    Surface * surf,
    unsigned int name,
    enum Modifier modifier,
    const double * apex,
    const double * axis,
    double ta
);

int torus_init(
    Surface * surf,
    unsigned int name,
    enum Modifier modifier,
    const double * center,
    const double * axis,
    double radius,
    double a,
    double b
);

int gq_init(
    Surface * surf,
    unsigned int name,
    enum Modifier modifier,
    const double * m,
    const double * v,
    double k
);

void surface_dispose(Surface * surf);

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