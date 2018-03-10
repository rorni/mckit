#include <cstdint>
#include "box.hpp"

#ifndef __SURFACE
#define __SURFACE

#define DIM 3

class Surface {
    public:
        Surface (int name);
        uint64_t hash();
        bool eq(const Surface& other);
        virtual bool equals(const Surface& other, const Box& box, double tol=1.e-10) =0;
        virtual double  func(const double* x, int sign=1) =0;
        virtual double* grad(const double* x, int sign=1) =0;
        virtual double* projection(const double* points, int npt) =0;
        virtual int  test_box(const Box& box);
        int* test_points(const double* points, int npt);
    private:
        int name;
        static unsigned long number;
        uint64_t id;
};


class Plane: public Surface {
    public:
        Plane(const double* normal, double offset, int name);
        bool equals(const Surface& other, const Box& box, double tol);
        double  func(const double* x, int sign);
        double* grad(const double* x, int sign);
        double* projection(const double* points, int npt);
        int  test_box(const Box& box);    
    private:
        double v[DIM];
        double k;
};

#endif