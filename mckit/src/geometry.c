#include "geometry.h"

char geom_complement(char * arg) {
    return -1 * (*arg);
}

char geom_intersection(char * args, size_t n) {
    size_t i;
    char result = +1;
    for (i = 0; i < n; ++i) {
        if (*(args + i) == 0) result = 0
        else if (*(args + i) == -1) {
            result = -1;
            break;
        }
    }
    return result;
}

char geom_union(char * args, size_t n) {
    size_t i;
    char result = -1;
    for (i = 0; i < n; ++i) {
        if (*(args + i) == 0) result = 0
        else if (*(args + i) == +1) {
            result = +1;
            break;
        }
    }
    return result;
}