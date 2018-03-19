#ifndef __GEOMETRY_H
#define __GEOMETRY_H

#include <stddef.h>
#include "box.h"

typedef struct Node Node;

enum Operation {INTERSECTION=0, COMPLEMENT, UNION, IDENTITY};

struct Node {
    Operation opc;
    size_t alen;
    Node * args;
    // other fields.
};

int node_init(Node * node, enum Operation opc, size_t n, const Node * args);

void node_dispose(Node * node);

int node_test_box(Node * node, const Box * box, char collect);

int node_test_points(
    const Node * node, 
    const double * points, 
    size_t npts, 
    int * result
);

int node_bounding_box(const Node * node, Box * box, double tol

char geom_complement(char * arg);
char geom_intersection(char * args, size_t n);
char geom_union(char * args, size_t n);

#endif