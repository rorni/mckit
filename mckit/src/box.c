#include <stdlib.h>
#include "gsl/gsl_cblas.h"
#include "box.h"

static double perm[NCOR][NDIM] = {
    {-1, -1, -1},
    {-1, -1,  1},
    {-1,  1, -1},
    {-1,  1,  1},
    { 1, -1, -1},
    { 1, -1,  1},
    { 1,  1, -1},
    { 1,  1,  1}
};


int box_init(
    Box * box,    
    const double * center, 
    const double * ex, 
    const double * ey, 
    const double * ez,
    double xdim, 
    double ydim, 
    double zdim
) 
{
    if (!box || !center || !ex || !ey || !ez) {
        return BOX_FAILURE;
    }
    
    int i, j;
    for (i = 0; i < NDIM; ++i) box->center[i] = center[i];
    
    box->dims[0] = xdim;
    box->dims[1] = ydim;
    box->dims[2] = zdim;
    
    box->volume = xdim * ydim * zdim;
    
    // basis vectors.
    for (i = 0; i < NDIM; ++i) {
        box->ex[i] = ex[i];
        box->ey[i] = ey[i];
        box->ez[i] = ez[i];
    }
    
    // Finding coordinates of box's corners
    for (i = 0; i < NCOR; ++i) {
        cblas_dcopy(NDIM, box->center, 1, box->corners[i], 1);
        cblas_daxpy(NDIM, 0.5 * perm[i][0], box->ex, 1, box->corners[i], 1);
        cblas_daxpy(NDIM, 0.5 * perm[i][1], box->ey, 1, box->corners[i], 1);
        cblas_daxpy(NDIM, 0.5 * perm[i][2], box->ez, 1, box->corners[i], 1);
    }
    
    // Finding lower and upper bounds
    cblas_dcopy(NDIM, box->corners[i], 1, box->lb, 1);
    cblas_dcopy(NDIM, box->corners[i], 1, box->ub, 1);
    for (int i = 1; i < NCOR; ++i) {
        for (int j = 0; j < NDIM; ++j) {
            if (box->corners[i][j] < box->lb[j]) box->lb[j] = box->corners[i][j];
            if (box->corners[i][j] > box->ub[j]) box->ub[j] = box->corners[i][j];
        }
    }
    
    box->rng = NULL;
        
    return BOX_SUCCESS;
}


void box_dispose(Box * box) {
    if (box != NULL && box->rng != NULL) gsl_rng_free((gsl_rng*) box->rng);
}


int box_generate_random_points(
    Box * box, 
    double * points,
    int npts
) 
{
    if (box->rng == NULL) box->rng = gsl_rng_alloc(gsl_rng_taus);
    if (box->rng == NULL) return BOX_FAILURE;    
    int i;
    double x, y, z;
    
    gsl_rng_set(rng, 0);
    
    for (i = 0; i < npts; ++i) {
        x = (gsl_rng_uniform(rng) - 0.5) * box->dims[0];
        y = (gsl_rng_uniform(rng) - 0.5) * box->dims[1];
        z = (gsl_rng_uniform(rng) - 0.5) * box->dims[2];
        
        cblas_dcopy(NDIM, box->center, 1, points + i * NDIM, 1);
        cblas_daxpy(NDIM, x, box->ex, 1, points + i * NDIM, 1);
        cblas_daxpy(NDIM, y, box->ey, 1, points + i * NDIM, 1);
        cblas_daxpy(NDIM, z, box->ez, 1, points + i * NDIM, 1);
    }
    return BOX_SUCCESS;
}


void box_test_points(
    const Box * box, 
    const double * points, 
    int npts,
    int * result
)
{
    double delta[NDIM];
    double x, y, z;
    int i;
    
    for (i = 0; i < npts; ++i) {
        cblas_dcopy(NDIM, points + i * NDIM, 1, delta, 1);
        cblas_daxpy(NDIM, -1, box->center, 1, delta, 1);
        x = cblas_ddot(NDIM, delta, 1, box->ex, 1);
        y = cblas_ddot(NDIM, delta, 1, box->ey, 1);
        z = cblas_ddot(NDIM, delta, 1, box->ez, 1);
        if (x > -0.5 && x < 0.5 && y > -0.5 && y < 0.5 && z > -0.5 && z < 0.5) {
            result[i] = 1;
        } else {
            result[i] = 0;
        }
    }
}


int box_split(
    const Box * box, 
    Box * box1, 
    Box * box2, 
    int dir, 
    double ratio
)
{
    // Find splitting direction
    if (dir == BOX_SPLIT_AUTODIR) dir = cblas_idamax(NDIM, box->dims, 1);
    
    double center1[NDIM], center2[NDIM], dims1[NDIM], dims2[NDIM];
    const double* basis[NDIM] = {box->ex, box->ey, box->ez};
    
    // find new dimensions
    cblas_dcopy(NDIM, box->dims, 1, dims1, 1);
    cblas_dcopy(NDIM, box->dims, 1, dims2, 1);
    dims1[dir] *= ratio;
    dims2[dir] *= 1 - ratio;
    
    // find new centers.
    cblas_dcopy(NDIM, box->center, 1, center1, 1);
    cblas_dcopy(NDIM, box->center, 1, center2, 1);
    
    cblas_daxpy(NDIM, -0.5 * dims2[dir], basis[dir], 1, center1, 1);
    cblas_daxpy(NDIM,  0.5 * dims1[dir], basis[dir], 1, center2, 1);
    
    // create new boxes.
    int status;
    status = box_create(box1, center1, box->ex, box->ey, box->ez, 
                        dims1[0], dims1[1], dims1[2]);
    if (status == BOX_FAILURE) return BOX_FAILURE;

    status = box_create(box2, center2, box->ex, box->ey, box->ez, 
                        dims2[0], dims2[1], dims2[2]);
    if (status == BOX_FAILURE) return BOX_FAILURE;
    
    return BOX_SUCCESS;
}


void box_ieqcons(
    unsigned int m,
    double * result,
    unsigned int n,
    const double * x,
    double * grad,
    void * f_data
)
{
    Box * box = (Box*) f_data;
    
    const double * basis[NDIM] = {box->ex, box->ey, box->ez};
    double  point[NDIM];
    int i, j, mult;
    
    for (i = 0; i < 6; ++i) {
        cblas_dcopy(NDIM, box->center, 1, point, 1);
        mult = 2 * (i % 2) - 1;
        j = i % 3;
        cblas_daxpy(NDIM, mult * box->dims[j], basis[j], 1, point, 1);
        result[i] = mult * (cblas_ddot(NDIM, basis[j], 1, x, 1) -
                            cblas_ddot(NDIM, basis[j], 1, point, 1));
        
        if (grad != NULL) {
            cblas_dcopy(NDIM, basis[j], 1, grad + i * NDIM, 1);
            cblas_dscal(NDIM, mult, grad + i * NDIM, 1);
        }
    }  
}