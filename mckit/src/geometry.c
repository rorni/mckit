#include <stdlib.h>
#include "geometry.h"
#include "surface.h"

static uint64_t node_hash(Node * node);

int node_init(Node * node, enum Operation opc, size_t n, void * args)
{
    set_init(&node->args, 
    if (opc == COMPLEMENT || opc == IDENTITY) {
        if (n > 1) return NODE_FAILURE;
        
    } else {
        
    }
    node->opc = opc;
    node->hash = node_hash(node);
    return NODE_SUCCESS;
}

int node_compare(const Node * a, const Node * b)
{
    if (a->hash < b->hash) return  1;
    if (a->hash > b->hash) return -1;
    // hash collision?
    if (a->opc < b->opc) return  1;
    if (a->opc > b->opc) return -1;
    // compare arguments
    if (a->args->len < b->args->len) return  1;
    if (a->args->len > b->args->len) return -1;
    
    set_iter(a->args, -1);
    set_iter(b->args, -1);
    Node *aa, *bb;
    int c;
    while ((aa = set_next(a->args)) && (bb = set_next(b->args))) {
        c = node_compare(aa, bb);
        if (c != 0) break;
    }
    return c;
}

int node_copy(const Node * src, Node * dst)
{
    int status;
    if (src->opc == COMPLEMENT || src->opc == IDENTITY) {
        status = node_init(dst, src->opc, 1, set_at_index(src->args, 0));
    } else {
        size_t i = 0, n = src->args->len;
        Node *args[n], *a;
        set_iter(src->args, -1);
        while (a = set_next(src->args)) {
            args[i] = (Node *) malloc(sizeof(Node));
            if (args[i] == NULL) return NODE_NO_MEMORY;
            status = node_copy(a, args[i]);
            if (status != NODE_SUCCESS) return status;
            ++i;
        }
        status = node_init(dst, src->opc, n, args);
    }
    return status;
}

// Auxiliary functions

static uint64_t node_hash(Node * node) 
{
    uint64_t result = 0;
    if (node->opc == UNION || node->opc == IDENTITY) {
        Surface * sur = set_at_index(node->args, 0);
        result = sur->hash;
    } else {
        set_iter(node->args, -1);
        Node *a;
        while (a = set_next(node->args)) result ^= a->hash;
    }
    if (node->opc == UNION || node->opc == COMPLEMENT) result = ~result;
    return result;
}

// Operation functions

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