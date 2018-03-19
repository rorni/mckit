#include <stdlib.h>
#include "geometry.h"
#include "surface.h"

int node_init(Node * node, enum Operation opc, size_t n, void * args)
{
    if (opc == COMPLEMENT || opc == IDENTITY) {
        if (n > 1) return NODE_FAILURE;
        node->args = args;
        node->n = 1;
    } else {
        node->n = n;
        node->args = args;
    }
    node->opc = opc;
    return NODE_SUCCESS;
}

int node_copy(const Node * src, Node * dst)
{
    
}

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