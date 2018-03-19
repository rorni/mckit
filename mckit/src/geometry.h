#ifndef __GEOMETRY_H
#define __GEOMETRY_H

#include <stddef.h>
#include "box.h"

#define NODE_SUCCESS  0
#define NODE_FAILURE -1

typedef struct Node Node;

enum Operation {INTERSECTION=0, COMPLEMENT, UNION, IDENTITY};

struct Node {
    Operation opc;
    size_t alen;
    void * args;
    // other fields.
};

int node_init(Node * node, enum Operation opc, size_t n, const void * args);

void node_dispose(Node * node);

int node_test_box(Node * node, const Box * box, char collect);

int node_test_points(
    const Node * node, 
    const double * points, 
    size_t npts, 
    int * result
);

int node_bounding_box(const Node * node, Box * box, double tol);

double node_volume(const Node * node, const Box * box, double min_vol);

void node_clean(Node * node);

int node_complexity(const Node * node);

void node_get_surfaces(const Node * node);

void node_collect_stat(Node * node, const Box * box, double min_vol);

void node_get_simplest(Node * node);

int node_copy(const Node * src, Node * dst);

enum Operation node_invert_opc(const Node * node);


char geom_complement(char * arg);
char geom_intersection(char * args, size_t n);
char geom_union(char * args, size_t n);

#endif