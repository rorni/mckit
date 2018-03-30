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
    RBTree * stats;
    uint64_t hash;
    size_t ref_count;
    uint64_t last_box;
    int last_box_result;
};

// Creates new node or returns pointer to the existing such node.
// 
Node * node_create(
    Operation opc,     // Operation code.
    size_t alen,       // Length of argument array.
    const void * args  // Argument array.
);

// Frees memory allocated for Node.
//
void node_free(Node * node);

// Tests box location with respect to the node. 
// Returns BOX_INSIDE_NODE | BOX_CAN_INTERSECT_NODE | BOX_OUTSIDE_NODE
//
int node_test_box(
    Node * node,      // Node to test.
    const Box * box,  // Box to test.
    char collect      // Collect statistics about results.
);

// Tests whether points belong to this node.
// Retruns status - NODE_SUCCESS | NODE_NO_MEMORY
//
int node_test_points(
    const Node * node,      // test node
    const double * points,  // array of points - NDIM * npts
    size_t npts,            // the number of points
    int * result            // Result - +1 if point belongs to node, -1 
                            // otherwise. It must have length npts.
);

int is_empty(Node * node);
int is_universe(Node * node);
int node_complexity(const Node * node);
void node_get_surfaces(const Node * node, RBTree * surfs);

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