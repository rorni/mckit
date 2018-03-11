#ifndef __BOX
#define __BOX

#include <map>
#include <vector>
#include "surface.hpp"

using namespace std

class Box {
    public:
        Box(const double *center, double xdim, double ydim, double zdim);
        Box(const double *center, double xdim, double ydim, double zdim, 
            const double *ex, const double *ey, const double *ez);
        const vector<double> & lower_bounds(void);
        const vector<double> & upper_bounds(void);
        const double * get_corners(void);
        double volume(void);
        void split(Box& box1, Box& box2, int dim=-1, double ratio=0.5);
        void test_points(const double * points, int npt, int * result);
        double ieqcons(const vector<double> &x, vector<double> &grad, void *data);
        const double* get_random_points(int& npt);
        void generate_random_points(int npt);
    private:
        vector<double> lb = vector<double>(DIM);
        vector<double> ub = vector<double>(DIM);
        double corners[CNUM][DIM];
        double center[DIM];
        double dims[DIM];
        double ex[DIM], ey[DIM], ez[DIM];
        double vol;
        double * rpts {nullptr};
        int nrp;
        static const int CNUM {8}
};

#endif