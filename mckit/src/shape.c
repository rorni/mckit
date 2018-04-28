//
// Created by Roma on 14.04.2018.
//

#include <stdlib.h>
#include "shape.h"
#include "surface.h"

#define is_final(opc) (opc == COMPLEMENT || opc == IDENTITY)
#define is_void(opc)  (opc == EMPTY || opc == UNIVERSE)
#define is_composite(opc) (opc == UNION || opc == INTERSECTION)

#define geom_complement(arg) (-1 * (arg))

static int zero_surfs = 0;

char geom_intersection(char * args, size_t n, size_t inc);
char geom_union(char * args, size_t n, size_t inc);

typedef struct StatUnit StatUnit;

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

// Initializes shape struct
int shape_init(
        Shape * shape,          // Pointer to struct to be initialized
        char opc,               // Operation code
        size_t alen,            // Length of arguments
        const void * args       // Argument array.
)
{
    shape->opc = opc;
    shape->alen = alen;
    shape->stats = rbtree_create(stat_compare);;
    shape->last_box = 0;
    shape->last_box_result = 0;
    if (is_final(opc)) {
        shape->args.surface = (Surface *) args;
    } else if (is_void(opc)) {
        shape->args.surface = NULL;
    } else {
        shape->args.shapes = (Shape **) malloc(alen * sizeof(Shape *));
        if (shape->args.shapes == NULL) return SHAPE_NO_MEMORY;
        size_t i;
        for (i = 0; i < alen; ++i) shape->args.shapes[i] = ((Shape **) args)[i];
    }
    return SHAPE_SUCCESS;
}

void shape_dealloc(Shape * shape)
{
    if (is_composite(shape->opc)) free(shape->args.shapes);
    if (shape->stats != NULL) {
        shape_reset_stat(shape);
        rbtree_free(shape->stats);
    }
}

// Tests box location with respect to the shape.
// Returns BOX_INSIDE_SHAPE | BOX_CAN_INTERSECT_SHAPE | BOX_OUTSIDE_SHAPE
//
int shape_test_box(
        Shape * shape,          // Shape to test.
        const Box * box,        // Box to test.
        char collect            // Collect statistics about results.
)
{
    if (shape->last_box != 0) {
        int bc = box_is_in(box, shape->last_box);
        // if it is the box already tested (bc == 0) then returns cached result;
        // if it is inner box - then returns cached result only if it is not 0. For inner box result may be different.
        if (bc == 0 || bc > 0 && shape->last_box_result != BOX_CAN_INTERSECT_SHAPE)
            return shape->last_box_result;
    }

    int result;
    if (is_final(shape->opc)) {
        char already = (box->subdiv == (shape->args.surface)->last_box);
        result = surface_test_box(shape->args.surface, box);
        if (shape->opc == COMPLEMENT) result = geom_complement(result);
        if (collect > 0 && result == 0 && !already) ++zero_surfs;
        else if (collect < 0 && result == 0) {
            if (zero_surfs == 1) result = 1;
            else result = -1;
        }
    } else if (shape->opc == UNIVERSE) {
        result = BOX_INSIDE_SHAPE;
    } else if (shape->opc == EMPTY) {
        result = BOX_OUTSIDE_SHAPE;
    } else {
        char * sub = malloc(shape->alen * sizeof(char));

        for (int i = 0; i < shape->alen; ++i) {
            sub[i] = shape_test_box((shape->args.shapes)[i], box, collect);
        }

        if (shape->opc == INTERSECTION) {
            result = geom_intersection(sub, shape->alen, 1);
        } else {
            result = geom_union(sub, shape->alen, 1);
        }

        // TODO: Review statistics collection
        if (collect != 0 && result != 0) {
            StatUnit * stat = (StatUnit *) malloc(sizeof(StatUnit));
            stat->arr = sub;
            stat->len = shape->alen;
            stat->vol = box->volume;
            if (rbtree_add(shape->stats, stat) != RBT_OK) {
                free(stat);
                free(sub);
            }
        } else free(sub);
    }
    // Cache test result;
    shape->last_box = box->subdiv;
    shape->last_box_result = result;
    return result;
}

// Tests box location with respect to the shape. It tries to find out
// if the box really intersects the shape with desired accuracy.
// Returns BOX_INSIDE_SHAPE | BOX_CAN_INTERSECT_SHAPE | BOX_OUTSIDE_SHAPE
int shape_ultimate_test_box(
        Shape * shape,          // Pointer to shape
        const Box * box,        // box
        double min_vol,         // minimal volume until which splitting process goes.
        char collect            // Whether to collect statistics about results.
)
{
    if (collect != 0 && (box->volume <= min_vol || zero_surfs == 1)) collect = -1;
    else zero_surfs = 0;
    int result = shape_test_box(shape, box, collect);
    if (result == BOX_CAN_INTERSECT_SHAPE && box->volume > min_vol) {
        Box box1, box2;
        box_split(box, &box1, &box2, BOX_SPLIT_AUTODIR, 0.5);
        int result1 = shape_ultimate_test_box(shape, &box1, min_vol, collect);
        int result2 = shape_ultimate_test_box(shape, &box2, min_vol, collect);
        if (result1 != BOX_CAN_INTERSECT_SHAPE && result2 != BOX_CAN_INTERSECT_SHAPE)
            return result1;     // No matter what value (result1 or result2) is returned because they
                                // will be equal.
    }
    return result;
}

// Tests whether points belong to this shape.
// Returns status - SHAPE_SUCCESS | SHAPE_NO_MEMORY
//
int shape_test_points(
        const Shape * shape,    // test shape
        size_t npts,            // the number of points
        const double * points,  // array of points - NDIM * npts
        char * result           // Result - +1 if point belongs to shape, -1
                                // otherwise. It must have length npts.
)
{
    int i;
    if (is_final(shape->opc)) {
        surface_test_points(shape->args.surface, npts, points, result);
        if (shape->opc == COMPLEMENT)
            for (i = 0; i < npts; ++i) result[i] = geom_complement(result[i]);
    } else if (is_void(shape->opc)) {
        char fill = (shape->opc == UNIVERSE) ? 1 : -1;
        for (i = 0; i < npts; ++i) result[i] = fill;
    } else {
        char (*op)(char * arg, size_t n, size_t inc);
        op = (shape->opc == INTERSECTION) ? geom_intersection : geom_union;

        size_t n = shape->alen;
        char * sub = malloc(n * npts * sizeof(char));
        if (sub == NULL) return SHAPE_NO_MEMORY;

        for (i = 0; i < n; ++i) {
            shape_test_points((shape->args.shapes)[i], npts, points, sub + i * npts);
        }
        for (i = 0; i < npts; ++i) result[i] = op(sub, n * npts, npts);
        free(sub);
    }
    return SHAPE_SUCCESS;
}

// Gets bounding box, that bounds the shape.
int shape_bounding_box(
        const Shape * shape,    // Shape to de bound
        Box * box,              // INOUT: Start box. It is modified to obtain bounding box.
        double tol              // Absolute tolerance. When change of box dimensions become smaller than tol
                                // the process of box reduction finishes.
)
{
    double lower, upper, ratio;
    int dim, tl;
    double min_vol = tol * tol * tol;
    Box box1, box2;
    for (dim = 0; dim < NDIM; ++dim) {
        lower = 0;
        while (box->dims[dim] - lower > tol) {
            ratio = 0.5 * (lower + box->dims[dim]) / box->dims[dim];
            box_split(box, &box1, &box2, dim, ratio);
            tl = shape_ultimate_test_box(shape, &box2, min_vol, 0);
            if (tl == -1) box_copy(box, &box1);
            else lower = box1.dims[dim];
        }
        upper = 0;
        while (box->dims[dim] - upper > tol) {
            ratio = 0.5 * (box->dims[dim] - upper) / box->dims[dim];
            box_split(box, &box1, &box2, dim, ratio);
            tl = shape_ultimate_test_box(shape, &box1, min_vol, 0);
            if (tl == -1) box_copy(box, &box2);
            else upper = box2.dims[dim];
        }
    }
    return SHAPE_SUCCESS;
}

// Gets volume of the shape
double shape_volume(
        const Shape * shape,    // Shape
        const Box * box,        // Box from which the process of volume finding starts
        double min_vol          // Minimum volume - when volume of the box become smaller than min_vol the process
                                // of box splitting finishes.
)
{
    int result = shape_test_box(shape, box, 0);
    if (result == BOX_INSIDE_SHAPE) return box->volume;   // Box totally belongs to the shape
    if (result == BOX_OUTSIDE_SHAPE) return 0;             // Box don't belong to the shape
    if (box->volume > min_vol) {            // Shape intersects the box
        Box box1, box2;
        box_split(box, &box1, &box2, BOX_SPLIT_AUTODIR, 0.5);
        double vol1 = shape_volume(shape, &box1, min_vol);
        double vol2 = shape_volume(shape, &box2, min_vol);
        return vol1 + vol2;
    } else {                        // Minimum volume has been reached, but shape still intersects box
        return 0.5 * box->volume;   // This is statistical decision. On average a half of the box belongs to the shape.
    }
}

// Resets collected statistics or initializes statistics storage
void shape_reset_stat(Shape * shape)
{
    StatUnit * s;
    while ((s = rbtree_pop(shape->stats, NULL)) != NULL) {
        free(s->arr);
        free(s);
    }
    shape->last_box = 0;
    if (is_composite(shape->opc)) {
        for (int i = 0; i < shape->alen; ++i) {
            shape_reset_stat(shape->args.shapes[i]);
        }
    }
}

// Collects statistics about shape.
void shape_collect_statistics(
        Shape * shape,          // Shape
        const Box * box,        // Global box, where statistics is collected
        double min_vol          // minimal volume, when splitting process stops.
)
{
    zero_surfs = 0;
    shape_reset_stat(shape);
    shape_ultimate_test_box(shape, box, min_vol, 1);
}

// Gets statistics table
char * shape_get_stat_table(
        Shape * shape,          // Shape
        size_t * nrows,         // number of rows
        size_t * ncols          // number of columns
)
{
    *nrows = shape->stats->len;
    *ncols = shape->alen;
    char * table = malloc(*ncols * *nrows * sizeof(char));
    StatUnit ** statarr = rbtree_to_array(shape->stats);
    size_t i, j;
    for (i = 0; i < *nrows; ++i)
        for (j = 0; j < *ncols; ++j)
            *(table + i * (*ncols) + j) = statarr[i]->arr[j];
    free(statarr);
    return table;
}

// Operation functions

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
