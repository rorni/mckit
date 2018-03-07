#include <iostream>
#include "surface.hpp"

int main() {
    Plane* p;
    double n[] {1, 0, 0};
    p = new Plane(&n[0], 2, 5);
    std::cout << p->test_points(&n[0], 1);
    return 0;
}