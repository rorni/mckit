#ifndef __BOX
#define __BOX

#define NDIM 3

typedef struct Box Box;

struct Box {
    double origin[NDIM];
    double ex[NDIM];
    double ey[NDIM];
    double ez[NDIM];
    double dims[NDIM];
    double lb[NDIM];
    double ub[NDIM];
    double corners[8][NDIM];
    double * points;
    double volume;
    int npts;
};

void box_create(Box * box, const double *origin, *ex, *ey, *ez);
void box_generate_random_points(Box * box, int npts);
int * box_test_points(const Box * box, const double * points, int npts);
void box_split(const Box * box, Box *box1, *box2, int dir, double ratio);

#endif