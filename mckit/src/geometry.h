#ifndef __GEOMETRY_H
#define __GEOMETRY_H

#include <stddef.h>
#include "box.h"
#include "rbtree.h"

#define NODE_SUCCESS    0
#define NODE_FAILURE   -1
#define NODE_NO_MEMORY -2
#define NODE_WRONG_ARGLENGTH -3

#define BOX_INSIDE_NODE        +1
#define BOX_CAN_INTERSECT_NODE  0
#define BOX_OUTSIDE_NODE       -1
#define COLLECT_STAT    1

typedef struct Node Node;
typedef enum Operation Operation;

enum Operation {INTERSECTION=0, COMPLEMENT, EMPTY, UNION, IDENTITY, UNIVERSE};

struct Node {
    Operation opc;
    void * args;
    size_t alen;
    Set stats;
    uint64_t hash;
};

// Initializes Node structure. Returns status: NODE_SUCCESS | NODE_FAILURE
// 
int node_init(
    Node * node,       // Node to be initialized.
    Operation opc,     // Operation code.
    size_t alen,       // Length of argument array.
    const void * args  // Argument array.
);

// Frees memory allocated for Node struct members.
//
void node_dispose(Node * node);

Node * node_copy(const Node * src);

// Tests box location with respect to the node. 
// Returns BOX_INSIDE_NODE | BOX_CAN_INTERSECT_NODE | BOX_OUTSIDE_NODE
//
int node_test_box(
    Node * node,      // Node to test.
    const Box * box,  // Box to test.
    char collect      // Collect statistics about results.
);

int node_test_points(
    const Node * node, 
    const double * points, 
    size_t npts, 
    int * result
);

int is_empty(Node * node);
int is_universe(Node * node);
int node_complexity(const Node * node);
void node_get_surfaces(const Node * node, Set * surfs);

int node_bounding_box(const Node * node, Box * box, double tol);
double node_volume(const Node * node, const Box * box, double min_vol);

int node_compare(const Node * a, const Node * b);

void node_collect_stat(Node * node, const Box * box, double min_vol);
void node_get_simplest(Node * node);

Node * node_complement(const Node * src);
Node * node_intersection(const Node * a, const Node * b);
Node * node_union(const Node * a, const Node * b);

char geom_complement(char * arg);
char geom_intersection(char * args, size_t n);
char geom_union(char * args, size_t n);

#endif