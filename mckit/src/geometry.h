#ifndef __GEOMETRY_H
#define __GEOMETRY_H

#include <stddef.h>
#include "box.h"
#include "container.h"

#define NODE_SUCCESS    0
#define NODE_FAILURE   -1
#define NODE_NO_MEMORY -2
#define NODE_WRONG_ARGLENGTH -3

typedef struct Node Node;
typedef enum Operation Operation;

enum Operation {INTERSECTION=0, COMPLEMENT, EMPTY, UNION, IDENTITY, UNIVERSE};

struct Node {
    Operation opc;
    const void * args;
    Map * stats;
    uint64_t hash;
};

Node * node_create(Operation opc, size_t n, const void * args);

void node_destroy(Node * node);

Node * node_copy(const Node * src);

int node_test_box(Node * node, const Box * box, char collect);

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