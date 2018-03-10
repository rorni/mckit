#include <map>
#include <vector>
#include "surface.hpp"

#ifndef __BOX
#define __BOX

class Box {
    public:
        Box(const double *origin, *ex, *ey, *ez);
        const vector<double> & lower_bounds(void);
        const vector<double> & upper_bounds(void);
        const vector<double*> & get_corners(void);
        double volume(void);
        void split(Box& box1, Box& box2, int dim=-1, double ratio=0.5);
        int* test_points(double* points, int npt);
        double f_ieqcons(double* x);
        double* fprime_ieqcons(double* x);
        double* get_random_points(int& npt=1);
    private:
        vector<double> lb = vector<double>(DIM);
        vector<double> ub = vector<double>(DIM);
        vector<double*> corners = vector<double>(8);
        void generate_random_points(int npt);
        double origin[DIM];
        double dims[DIM];
        double ex[DIM], ex[DIM], ex[DIM];
};

#endif