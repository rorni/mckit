#include <stdlib.h>
#include "geometry.h"
#include "surface.h"

#define is_final(opc) (opc == COMPLEMENT || opc == IDENTITY)
#define is_void(opc)  (opc == EMPTY || opc == UNIVERSE)
#define is_composite(opc) (opc == UNION || opc == INTERSECTION)
#define invert_opc(opc) ((opc + 3) % 6)

#define NODE_STANDARD  0
#define NODE_EMPTY    -1
#define NODE_UNIVERSE  1

typedef struct StatUnit StatUnit;

struct StatUnit {
    char * arr;
    size_t len;
}

static stat_compare(const StatUnit * a, const StatUnit * b) {
    int i, n = a->len;
    for (i = 0; i < n; ++i) {
        if (a[i] < b[i]) return 1;
        else if (a[i] > b[i]) return -1;
    }
    return 0;
}

static uint64_t node_hash(Node * node);
static Node ** extend_args(Operation opc, const void * args, size_t * n);

Node * node_create(enum Operation opc, size_t n, const void * args)
{
    Node * node = malloc(sizeof(Node));
    if (node == NULL) return NULL;
    
    node->opc = opc;
    if (is_final(opc)) node->args = args;
    else if (is_composite(opc)) {
        node->args = set_create(node_compare);
        args = extend_args(opc, args, &n);
        Node * a;
        int i;
        for (i = 0; i < n; ++i) {
            if (args[i]->opc == EMPTY && opc == INTERSECTION || \
                args[i]->opc == UNIVERSE && opc == UNION) {
                
                node->opc = args[i]->opc;
                break;
            } else if (args[i]->opc == EMPTY && opc == UNION || \
                       args[i]->opc == UNIVERSE && opc == INTERSECTION) {
                continue;
            }
            a = node_complement(args[i]);
            if (set_contains(node->args, a) {
                if (opc == INTERSECTION) {
                    node->opc = EMPTY;
                    node_destroy(a);
                    break;
                } else if (opc == UNION) {
                    node_destroy(node_pop(node->args, a));
                }
            } else if (!set_contains(node->args, args[i]) {
                set_add(node->args, node_copy(args[i]);
            }
            node_destroy(a);
        }
        free(args);
    }
    if (is_void(node->opc) && node->args != NULL) {
        set_free(node->args, 1);
    } 
    if (is_composite(node->opc)) {
        if (node->args->len == 0) {
            set_free(node->args, 1);
            node->opc = UNIVERSE;
        } else if (node->args->len == 1) {
            Node * a = set_at_index(node->args, 0);
            set_free(node->args, 1); // REVISE DELETION!!!
            free(node);
            node = a;
        }
    }
    node->hash = node_hash(node);
    node->stats = map_create(stat_compare);
    return node;
}

void node_destroy(Node * node)
{
    if (is_composite(node->opc)) {
        
    }
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

Node * node_copy(const Node * src)
{
    Node * dst = malloc(sizeof(Node));
    if (dst == NULL) return NULL;
    
    dst->opc = src->opc;
    dst->hash = src->hash;
    dst->state = src->state;
    
    if (is_final(src->opc)) dst->args = src->args;
    else {
        dst->args = set_create(node_compare);
        Node * a;
        set_iter(src->args, -1);
        while (a = set_next(src->args)) {
            set_add(dst->args, node_copy(a));
        }
    }
    return dst;
}

int node_test_box(Node * node, const Box * box, char collect)
{
    int result;
    if (is_final(node->opc)) {
        result = surface_test_box(node->args, box);
        if (node->opc == COMPLEMENT) result = geom_complement(result);
    } else if (node->opc == UNIVERSE) {
        result = +1;
    } else if (node->opc == EMPTY) {
        result = -1;
    } else {
        size_t n = node->args->len;
        char * sub = malloc(n * sizeof(char));
        set_iter(node->args, -1);
        for (i = 0; i < n; ++i) {
            sub[i] = node_test_box(set_next(node->args), box, collect);
        }
        if (node->opc == INTERSECTION) {
            result = geom_intersection(sub, n, 1);
        } else {
            result = geom_union(sub, n, 1);
        }
        
        if (collect && result != 0) {
            StatUnit * stat = (StatUnit *) malloc(sizeof(StatUnit));
            stat->arr = sub;
            stat->len = n;
            double * vol = map_get(node->stats, stat);
            if (vol == NULL) {
                vol = (double *) malloc(sizeof(double));
                *vol = box->volume;
                map_add(node->stats, stat, vol);
            } else {
                *vol += box->volume;
                free(sub);
                free(stat);
            }
        } else free(sub);
    }
    return result;
}

int node_test_points(
    const Node * node, 
    const double * points, 
    size_t npts, 
    int * result
)
{
    int i;
    if (is_final(node->opc)) {
        surface_test_points(node->args, points, npts, result);
        if (node->opc == COMPLEMENT)
            for (i = 0; i < npts; ++i) result[i] = geom_complement(result[i]);
    } else if (is_void(node->opc)) {
        char fill = node->opc == UNIVERSE ? 1 : -1;
        for (i = 0; i < npts; ++i) result[i] = fill;
    } else {
        char (*op)(char * arg, size_t n, size_t inc);
        op = node->opc == INTERSECTION ? geom_intersection : geom_union;
        size_t n = node->args->len;
        int * sub = malloc(n * npts * sizeof(int));
        if (sub == NULL) return NODE_NO_MEMORY;
        set_iter(node->args, -1);
        for (i = 0; i < n; ++i) {
            node_test_points(set_next(node->args), points, npts, sub+i*npts);
        }
        for (i = 0; i < npts; ++i) result[i] = (*op)(sub, n * npts, npts);
        free(sub);        
    }
    return NODE_SUCCESS;
}

int node_bounding_box(const Node * node, Box * box, double tol)
{
    double lower, apper, ratio;
    int dim, tl;
    Box box1, box2;
    for (dim = 0; dim < NDIM; ++dim) {
        lower = 0;
        while (box->dims[dim] - lower) > tol {
            ratio = 0.5 * (lower + box->dims[dim]) / box->dims[dim];
            box_split(box, &box1, &box2, dim, ratio);
            tl = node_test_box(node, &box2, 0);
            if (tl == -1) box_copy(&box1, box);
            else lower = box1.dims[dim];
        }
        upper = 0;
        while (box->dims[dim] - upper > tol) {
            ratio = 0.5 * (box->dims[dim] - upper) / box->dims[dim];
            box_split(box, &box1, &box2, dim, ratio);
            tl = node_test_box(node, &box1, 0);
            if (tl == -1) box_copy(&box2, box);
            else upper = box2.dims[dim];
        }
    }
    return NODE_SUCCESS;
}

double node_volume(const Node * node, const Box * box, double min_vol)
{
    int result = node_test_box(node, box, 0);
    if (result == +1) return box->volume;
    if (result == -1) return 0;
    if (box->volume > min_vol) {
        Box box1, box2;
        box_split(box, box1, box2, BOX_SPLIT_AUTODIR, 0.5);
        double vol1 = node_volume(node, box1, min_vol);
        double vol2 = node_volume(node, box2, min_vol);
        return vol1 + vol2;
    } else return 0.5 * box->volume;
}

void node_collect_stat(Node * node, const Box * box, double min_vol)
{
    int result = node_test_box(node, box, 1);
    if (result != 0) return result;
    if (box->volume <= min_volume) return -1;
    Box box1, box2;
    box_split(box, box1, box2, BOX_SPLIT_AUTODIR, 0.5);
    int res[2];
    res[0] = node_collect_stat(node, box1, min_vol);
    res[1] = node_collect_stat(node, box2, min_vol);
    result = geom_union(res, 2, 1);
    return result;
}

void node_get_simplest(Node * node)
{
    
}

Node * node_complement(const Node * src)
{
    Node * dst = malloc(sizeof(Node));
    if (dst == NULL) return NULL;
    dst->opc = invert_opc(src->opc);
    if (is_final(dst->opc)) dst->args = src->args;
    else if (is_composite(dst->opc)) {
        dst->args = set_create(node_compare);
        if (dst->args == NULL) {
            node_destroy(dst);
            return NULL;
        }
        set_iter(src->args, -1);
        Node * a;
        while (a = set_next(src->args)) {
            set_add(dst->args, node_complement(a));
        }
    }
    dst->hash = node_hash(dst);
    return dst;
}

Node * node_intersection(const Node * a, const Node * b)
{
    Node args[2] = {a, b};
    return node_create(INTERSECTION, 2, args);   
}

Node * node_union(const Node * a, const Node * b)
{
    Node args[2] = {a, b};
    return node_create(UNION, 2, args);
}

int is_empty(Node * node) {return node->state == NODE_EMPTY;}

int is_universe(Node * node) {return node->state == NODE_UNIVERSE;}

int node_complexity(const Node * node) 
{
    int result = 0;
    if (is_final(node->opc)) result = 1;
    else {
        set_iter(node->args, -1);
        while (a = set_next(node->args)) result += node_complexity(a);
    }
    return result;
}

void node_get_surfaces(const Node * node, Set * surfs)
{
    if (is_final(node->opc)) set_add(surfs, node->args);
    else {
        Node * a;
        set_iter(node->args, -1);
        while (a = set_next(node->args)) node_get_surfaces(a, surfs);
    }
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

static Node ** extend_args(Operation opc, const void * args, size_t * n)
{
    Node * result[], *a;
    size_t i, nn = 0, j = 0;
    for (i = 0; i < n; ++i) {
        if (args[i]->opc == opc) nn += args[i]->args->len;
        else nn += 1;
    }
    result = malloc(nn * sizeof(Node*));
    j = 0;
    for (i = 0; i < n; ++i) {
        if (args[i]->opc == opc) {
            set_iter(args[i]->args, -1);
            while (a = set_next(args[i]->args)) result[j++] = a;
        } else result[j++] = args[i];
    }
    *n = nn;
    return result;
}

// Operation functions

char geom_complement(char arg) {
    return -1 * arg;
}

char geom_intersection(char * args, size_t n, size_t inc) {
    size_t i;
    char result = +1;
    for (i = 0; i < n; i += inc) {
        if (*(args + i) == 0) result = 0;
        else if (*(args + i) == -1) {
            result = -1;
            break;
        }
    }
    return result;
}

char geom_union(char * args, size_t n, size_t inc) {
    size_t i;
    char result = -1;
    for (i = 0; i < n; i += inc) {
        if (*(args + i) == 0) result = 0;
        else if (*(args + i) == +1) {
            result = +1;
            break;
        }
    }
    return result;
}