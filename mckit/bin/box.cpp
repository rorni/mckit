#include <cmath>
#include "box.hpp"
#include "surface.hpp"

Box :: Box(const double* base, double xlen, ylen, zlen): 
xlen{xlen}, ylen{ylen}, zlen{zlen} {
    this->base = new double[DIM];
    for (int i = 0; i < DIM; ++i) this->base[i] = base[i];
    
    lb[0] = fmin(base[0], base[0] + xlen);
    lb[1] = fmin(base[1], base[1] + ylen);
    lb[2] = fmin(base[2], base[2] + zlen);

    ub[0] = fmax(base[0], base[0] + xlen);
    ub[1] = fmax(base[1], base[1] + ylen);
    ub[2] = fmax(base[2], base[2] + zlen);
    
    for (int i = 0; i < 8; ++i) corners[i] = new double[DIM];
}