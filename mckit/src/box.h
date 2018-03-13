#ifndef __BOX
#define __BOX

#define NDIM 3
#define NCOR 8

#define BOX_SUCCESS 0
#define BOX_FAILURE 1

#define BOX_SPLIT_X 0
#define BOX_SPLIT_Y 1
#define BOX_SPLIT_Z 2
#define BOX_SPLIT_AUTODIR -1


typedef struct Box Box;

struct Box {
    double center[NDIM];    // center of the box
    double ex[NDIM];        // 
    double ey[NDIM];        // basis vectors. Shows directions of box's edges
    double ez[NDIM];        //
    double dims[NDIM];      // Dimensions of the box.
    double lb[NDIM];        // lower bounds
    double ub[NDIM];        // upper bounds
    double corners[NCOR][NDIM];  // corners
    double volume;
    void * rng;             // random generator
};

int box_create(             // creates or initializes new box
    Box * box,
    const double * center, 
    const double * ex, 
    const double * ey, 
    const double * ez,
    double xdim, 
    double ydim, 
    double zdim
);

void box_destroy(Box * box);

void box_generate_random_points(
    Box * box, 
    double * points,
    int npts
);

void box_test_points(
    const Box * box, 
    const double * points, 
    int npts,
    int * result
);

int box_split(
    const Box * box, 
    Box * box1, 
    Box * box2, 
    int dir, 
    double ratio
);

void box_ieqcons(
    unsigned int m,
    double * result,
    unsigned int n,
    const double * x,
    double * grad,
    void * f_data
);

#endif