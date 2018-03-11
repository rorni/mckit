#include <cmath>
#include "box.hpp"
#include "gsl/gsl_cblas.h"
#include "surface.hpp"

static double perm[Box::CNUM][DIM] = {
    {-1, -1, -1},
    {-1, -1,  1},
    {-1,  1, -1},
    {-1,  1,  1},
    { 1, -1, -1},
    { 1, -1,  1},
    { 1,  1, -1},
    { 1,  1,  1}
};

static double EX[DIM] = {1, 0, 0};
static double EY[DIM] = {0, 1, 0};
static double EZ[DIM] = {0, 0, 1};

Box :: Box(const double *center, double xdim, double ydim, double zdim, 
           const double *ex, const double *ey, const double *ez) 
{
    for (int i = 0; i < DIM; ++i) {
        this->center[i] = center[i];
        this->ex[i] = ex[i];
        this->ey[i] = ey[i];
        this->ez[i] = ez[i];
    }
    dims[0] = xdim;
    dims[1] = ydim;
    dims[2] = zdim;
    
    for (int i = 0; i < Box :: CNUM; ++i) {
        cblas_dcopy(DIM, this->center, 1, corners[i], 1);
        cblas_daxpy(DIM, 0.5 * perm[i][0], this->ex, 1, corners[i], 1);
        cblas_daxpy(DIM, 0.5 * perm[i][1], this->ey, 1, corners[i], 1);
        cblas_daxpy(DIM, 0.5 * perm[i][2], this->ez, 1, corners[i], 1);
    }
    
    cblas_dcopy(DIM, corners[i], 1, lb.data(), 1);
    cblas_dcopy(DIM, corners[i], 1, ub.data(), 1);
    for (int i = 1; i < Box :: CNUM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            if (corners[i][j] < lb[j]) lb[j] = corners[i][j];
            if (corners[i][j] > ub[j]) ub[j] = corners[i][j];
        }
    }
    
    this->vol = xdim * ydim * zdim;
}


Box :: Box(const double *center, double xdim, double ydim, double zdim):
       Box(center, xdim, ydim, zdim, EX, EY, EZ) {}
       
       
const vector<double> & Box :: lower_bounds(void) {
    return this->lb;
}

const vector<double> & Box :: upper_bounds(void) {
    return this->ub;
}

const double * Box :: get_corners(void) {
    return this->corners;
}

double Box :: volume(void) {
    return this->vol;
}

void Box :: test_points(const double* points, int npt, int * result) {
    // TODO
}

double Box :: ieqcons(const vector<double> &x, vector<double> &grad, void *data) {
    return 0;
}
        
const double* Box :: get_random_points(int& npt) {
    npt = nrp;
    return this->rpts;
}

void Box :: generate_random_points(int npt) {
    
}