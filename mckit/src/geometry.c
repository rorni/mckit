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

static uint64_t node_hash(Node * node);
static Node ** extend_args(int opc, const void * args, size_t * n);
static Node ** clean_args(int opc, const void * args, size_t * n);
static int is_complement(Node * a, Node * b);

static RBTree node_bag = {NULL, 0, (int (*)(const void *, const void *)) node_compare};

struct StatUnit {
    char * arr;
    size_t len;
    double vol;
};

static int stat_compare(const StatUnit * a, const StatUnit * b)
{
    size_t i, n = a->len;
    for (i = 0; i < n; ++i) {
        if (a->arr[i] < b->arr[i]) return 1;
        else if (a->arr[i] > b->arr[i]) return -1;
    }
    return 0;
}

static Node * node_retrieve(Node * node) 
{
    Node * stored = rbtree_get(&node_bag, node);
    if (stored == NULL) {
        rbtree_add(&node_bag, node);
    } else {
        node_free(node);
        node = stored;
        node->ref_count++;
    }
    return node;
}

Node * node_create(int opc, size_t alen, const void * args)
{
    Node * node = malloc(sizeof(Node));
    if (node != NULL) {
        node->opc = opc;
        node->alen = alen;
        
        node->ref_count = 1;
        if (is_final(opc)) node->args = args;
        else if (is_composite(opc)) {
            size_t i;
            
            Node ** cargs = clean_args(opc, args, &alen);
            if (alen == 1) {
                free(node);
                node = cargs[0];
                node->ref_count++;
                return node;
            } else {
                node->args = (void *) malloc(alen * sizeof(Node*));
                for (i = 0; i < alen; ++i) {
                    *(node->args + i) = cargs[i];
                    cargs[i]->ref_count++;
                }
                node->stats = rbtree_create(stat_compare);
            }
        }
        
        node->hash = node_hash(node);
        // Check if a such node already exists
        node = node_retrieve(node);
    }
    return node;
}

void node_free(Node * node) 
{
    node->ref_count--;
    if (node->ref_count == 0) {
        rbtree_pop(&node_bag, node);
                
        if (is_composite(node->opc)) {
            size_t i;
            for (i = 0; i < node->alen; ++i) {
                node_free((Node*) (node->args + i));
            } 
            free(node->args);
            
            StatUnit * a;
            while (a = (StatUnit*) rbtree_pop(node->stats, NULL)) {
                free(a->arr);
                free(a);
            }
            rbtree_free(node->stats);
        }
        
        free(node);
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

int node_test_box(Node * node, const Box * box, char collect)
{
    if (node->last_box != NULL) {
        int bc = box_is_in(box, node->last_box);
        if (bc == 0 || bc > 0 && last_box_result != 0) return last_box_result; 
    }
    
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
    // Cash test result;
    node->last_box = box->subdiv;
    node->last_box_result = result;
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
        while (box->dims[dim] - lower > tol) {
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
    Operation opc = invert_opc(src->opc);
    void * args = NULL;
    if (is_final(opc)) args = src->args;
    else if (is_composite(opc)) {
        args = (void*) malloc(src->alen * sizeof(Node*));
        size_t i;
        for (i = 0; i < src->alen; ++i) {
            *(args + i) = node_complement((Node*) (src->args + i));
        }
    }
    return node_create(opc, src->alen, args);
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

void node_get_surfaces(const Node * node, RBTree * surfs)
{
    if (is_final(node->opc)) rbtree_add(surfs, node->args);
    else if (is_composite(node->opc)) {
        Node * a;
        size_t i;
        for (i = 0; i < node->alen; ++i) {
            a = (Node *) (node->args + i);
            node_get_surfaces(a, surfs);
        }
    }
}

// Auxiliary functions

static uint64_t node_hash(Node * node) 
{
    uint64_t result = 0;
    if (is_final(node->opc)) {
        Surface * sur = (Surface *) node->args;
        result = sur->hash;
    } else if (is_composite(node->opc)) {
        size_t i;
        Node *a;
        for (i = 0; i < node->alen; ++i) {
            a = (Node *) (node->args + i);
            result ^= a->hash;
        }
    }
    if (node->opc == UNION || node->opc == COMPLEMENT || node->opc == UNIVERSE)
        result = ~result;
    return result;
}

static Node ** extend_args(Operation opc, const void * args, size_t * n)
{
    Node * result[], *a;
    size_t i, k, nn = 0, j = 0;
    for (i = 0; i < n; ++i) {
        a = (Node*) args[i];
        if (a->opc == opc) nn += a->alen;
        else nn += 1;
    }
    result = malloc(nn * sizeof(Node*));
    j = 0;
    for (i = 0; i < n; ++i) {
        a = (Node*) args[i];
        if (a->opc == opc) {
            for (k = 0; k < a->alen; ++k) result[j++] = a->args + k;
        } else result[j++] = args[i];
    }
    *n = nn;
    return result;
}

static Node ** clean_args(Operation opc, const void * args, size_t * n)
{
    Node ** ext_args = extend_args(opc, args, n);
    qsort(ext_args, n, sizeof(Node*), node_compare);
    Node ** result = (Node**) malloc(n * sizeof(Node*));
    size_t i, j, l, nn = 0;
    for (i = 0, l = 0; i < n; ++i) {
        if (is_empty(ext_args[i]) && opc == UNION || \
            is_universe(ext_args[i]) && opc == INTERSECTION) continue;
        else if (is_empty(ext_args[i]) && opc == INTERSECTION || \
                 is_universe(ext_args[i]) && opc == UNION) {
            result[0] = ext_args[i];
            nn = 1;
            break;
        } else if (i < n-1 && node_compare(ext_args[i], ext_args[i+1] == 0)) {
            continue;
        }
        for (j = i + 1; j < n; ++j) {
            if (is_complement(ext_args[i], ext_args[j])) {
                result[0] = node_create(EMPTY, 0, NULL);
                result[0]->ref_count--;
                nn = 1;
                break;
            }
        }    
        result[l++] = ext_args[i];
        ++nn;
    }
    n = nn;
    free(ext_args);
    return result;
}

static int is_complement(Node * a, Node * b)
{
    if (~(a->hash) != b->hash) return 0;
    if (a->opc != invert_opc(b->opc)) return 0;
    if (is_composite(a->opc)) {
        if (a->alen != b->alen) return 0;
        size_t i;
        for (i = 0; i < a->alen; ++i) {
            if (!is_complement(a->args + i, b->args + i)) return 0;
        }
    }
    return 1;
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