#include <cmath>
#include "surface.hpp"

Surface :: Surface (int name): name{name} { 
    ++number; 
    id = number;
}

uint64_t Surface :: hash() { return id; }

bool Surface :: eq(const Surface& other) { return id == other.id; }

Surface :: number = 0;

int * Surface :: test_points(const double* points, int npt) {
    int * result = new int[npt];
    int * x;
    for (int i = 0; i < npt; ++i) {
        x = points + DIM * i;
        result[i] = signbit(func(x));
    }
    return result;
}

int Surface :: test_box(const Box& box) {
    int result;
    // insert code here
    return result;
}


Plane :: Plane(const double* normal, double offset, int name): Surface(name) {
    for (int i = 0; i < DIM; ++i) v[i] = normal[i];
    k = offset;
}

int Plane :: test_box(const Box& box) {
    int result {-1};
    // insert code here.
    return result;
}

double Plane :: func(const double * x, int sign=1) {
    int result {k};
    for (int i = 0; i < DIM; ++i ) result += x[i] * v[i];
    return sign * result;
}

double * Plane :: grad(const double * x, int sign=1) {
    double * result = new double[DIM];
    for (int i = 0; i < DIM; ++i) result[i] = v[i];
    return result;
}

bool Plane :: equals(const Surface& other, const Box& box, double tol=1.e-10) {
    bool result {false};
    // insert code here
    return result;
}

double * Plane :: projection(const double* points, int npt) {
    return nullptr;
}

