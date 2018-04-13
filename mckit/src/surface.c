#include <stdlib.h>
#include <math.h>
#include "nlopt.h"
#include "mkl.h"
#include "surface.h"

double plane_func(
    unsigned int n,
    const double * x,
    double * grad,
    void * f_data
)
{
    Plane * data = (Plane *) f_data;
    if (grad != NULL) {
        cblas_dcopy(NDIM, data->norm, 1, grad, 1);
    }
    return cblas_ddot(NDIM, x, 1, data->norm, 1) + data->offset;
}

double sphere_func(
    unsigned int n,
    const double * x,
    double * grad,
    void * f_data
)
{
    Sphere * data = (Sphere *) f_data;
    if (grad != NULL) {
        cblas_dcopy(NDIM, x, 1, grad, 1);
        cblas_daxpy(NDIM, -1, data->center, 1, grad, 1);
        cblas_dscal(NDIM, 2, grad, 1);
    }
    double delta[NDIM];
    cblas_dcopy(NDIM, x, 1, delta, 1);
    cblas_daxpy(NDIM, -1, data->center, 1, delta, 1);
    return cblas_ddot(NDIM, delta, 1, delta, 1) - pow(data->radius, 2);
}

double cylinder_func(
    unsigned int n,
    const double * x,
    double * grad,
    void * f_data
)
{
    Cylinder * data = (Cylinder *) f_data;
    double a[NDIM];
    cblas_dcopy(NDIM, x, 1, a, 1);
    cblas_daxpy(NDIM, -1, data->point, 1, a, 1);
    double an = cblas_ddot(NDIM, a, 1, data->axis, 1);
    if (grad != NULL) {
        cblas_dcopy(NDIM, a, 1, grad, 1);
        cblas_daxpy(NDIM, -an, data->axis, 1, grad, 1);
        cblas_dscal(NDIM, 2, grad, 1);
    }
    return cblas_ddot(NDIM, a, 1, a, 1) - pow(an, 2) - pow(data->radius, 2);
}

double cone_func(
    unsigned int n,
    const double * x,
    double * grad,
    void * f_data
)
{
    Cone * data = (Cone *) f_data;
    double a[NDIM];
    cblas_dcopy(NDIM, x, 1, a, 1);
    cblas_daxpy(NDIM, -1, data->apex, 1, a, 1);
    double an = cblas_ddot(NDIM, a, 1, data->axis, 1);
    if (grad != NULL) {
        cblas_dcopy(NDIM, a, 1, grad, 1);
        cblas_daxpy(NDIM, -an * (1 + data->ta), data->axis, 1, grad, 1);
        cblas_dscal(NDIM, 2, grad, 1);
    }
    return cblas_ddot(NDIM, a, 1, a, 1) - pow(an, 2) * (1 + data->ta);
}

double gq_func(
    unsigned int n,
    const double * x,
    double * grad,
    void * f_data
)
{
    GQuadratic * data = (GQuadratic *) f_data;
    if (grad != NULL) {
        cblas_dcopy(NDIM, data->v, 1, grad, 1);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, NDIM, NDIM, 2, data->m, 1, x, 1, 1, grad, 1);       
    }
    double y[NDIM];
    cblas_dcopy(NDIM, data->v, 1, y, 1);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, NDIM, NDIM, 1, data->m, 1, x, 1, 1, grad, 1);           
    return cblas_ddot(NDIM, y, 1, x, 1) + data->k;
}

double torus_func(
    unsigned int n,
    const double * x,
    double * grad,
    void * f_data
)
{
    Torus * data = (Torus *) f_data;
    double p[NDIM];
    cblas_dcopy(NDIM, x, 1, p, 1);
    cblas_daxpy(NDIM, -1, data->center, 1, p, 1);
    double pn = cblas_ddot(NDIM, p, 1, data->axis, 1);
    double pp = cblas_ddot(NDIM, p, 1, p, 1);
    double sq = sqrt(max(pp - pow(pn, 2), 0));
    if (grad != NULL) {
        cblas_dcopy(NDIM, p, 1, grad, 1);
        cblas_daxpy(NDIM, -pn, data->axis, 1, grad, 1);
        cblas_dscal(NDIM, 1 / pow(data->b, 2), grad, 1);
        cblas_daxpy(NDIM, pn / pow(data->a, 2), data->axis, 1, grad, 1);
        cblas_dscal(NDIM, 2, grad, 1);
    }    
    return pow(pn / data->a, 2) + pow((sq - data->radius) / data->b, 2) - 1;
}

double surface_func(
    unsigned int n,
    const double * x,
    double * grad,
    void * f_data
)
{
    Surface * surf = (Surface *) f_data;
    double fval;
    switch (surf->type) {
        case PLANE:
            fval = plane_func(n, x, grad, f_data);
            break;
        case SPHERE:
            fval = sphere_func(n, x, grad, f_data);
            break;
        case CYLINDER:
            fval = cylinder_func(n, x, grad, f_data);
            break;
        case CONE:
            fval = cone_func(n, x, grad, f_data);
            break;
        case TORUS:
            fval = torus_func(n, x, grad, f_data);
            break;
        case GQUADRATIC:
            fval = gq_func(n, x, grad, f_data);
            break;
        default:
            fval = 0;
            break;
    }
    return fval;
}

inline void surface_init(Surface * surf, unsigned int name, int mod)
{
    surf->name = name;
    surf->modifier = mod;
    surf->last_box = 0;
    surf->last_box_result = 0;
    surf->hash = (uint64_t) surf;
}

int plane_init(
    Plane * surf,
    unsigned int name,
    int modifier,
    const double * norm,
    double offset
)
{
    int i;
    surface_init((Surface *) surf, name, modifier);
    surf->base.type = PLANE;
    surf->offset = offset;
    for (i = 0; i < NDIM; ++i) {
        surf->norm[i] = norm[i];
    }
    return SURFACE_SUCCESS;
}

int sphere_init(
    Sphere * surf,
    unsigned int name,
    int modifier,
    const double * center,
    double radius
)
{
    int i;
    surface_init((Surface *) surf, name, modifier);
    surf->base.type = SPHERE;
    surf->radius = radius;
    for (i = 0; i < NDIM; ++i) {
        surf->center[i] = center[i];
    }
    return SURFACE_SUCCESS;
}

int cylinder_init(
    Cylinder * surf,
    unsigned int name,
    int modifier,
    const double * point,
    const double * axis,
    double radius
)
{
    int i;
    surface_init((Surface *) surf, name, modifier);
    surf->base.type = CYLINDER;
    surf->radius = radius;
    for (i = 0; i < NDIM; ++i) {
        surf->point[i] = point[i];
        surf->axis[i] = axis[i];
    }
    return SURFACE_SUCCESS;
}

int cone_init(
    Cone * surf,
    unsigned int name,
    int modifier,
    const double * apex,
    const double * axis,
    double ta
)
{
    int i;
    surface_init((Surface *) surf, name, modifier);
    surf->base.type = CONE;
    surf->ta = ta;
    for (i = 0; i < NDIM; ++i) {
        surf->apex[i] = apex[i];
        surf->axis[i] = axis[i];
    }
    return SURFACE_SUCCESS;
}

int torus_init(
    Torus * surf,
    unsigned int name,
    int modifier,
    const double * center,
    const double * axis,
    double radius,
    double a,
    double b
)
{
    int i;
    surface_init((Surface *) surf, name, modifier);
    surf->base.type = TORUS;
    surf->radius = radius;
    surf->a = a;
    surf->b = b;
    for (i = 0; i < NDIM; ++i) {
        surf->center[i] = center[i];
        surf->axis[i] = axis[i];
    }
    if (surf->b > surf->radius) {
        double offset = a * sqrt(1 - pow(radius / b, 2));
        surf->specpts = (double *) malloc(2 * NDIM * sizeof(double));
        if (surf->specpts == NULL) return SURFACE_FAILURE;
        cblas_dcopy(NDIM, center, 1, surf->specpts, 1);
        cblas_dcopy(NDIM, center, 1, surf->specpts + NDIM, 1);
        cblas_daxpy(NDIM, offset, axis, 1, surf->specpts, 1);
        cblas_daxpy(NDIM, -offset, axis, 1, surf->specpts + NDIM, 1);
    }
    return SURFACE_SUCCESS;
}

int gq_init(
    GQuadratic * surf,
    unsigned int name,
    int modifier,
    const double * m,
    const double * v,
    double k
)
{
    int i, j;
    surface_init((Surface *) surf, name, modifier);
    surf->base.type = GQUADRATIC;
    surf->k = k;
    for (i = 0; i < NDIM; ++i) {
        surf->v[i] = v[i];
        for (j = 0; j < NDIM; ++j) surf->m[i * NDIM + j] = m[i * NDIM + j];
    }
    return SURFACE_SUCCESS;
}

void torus_dispose(Torus * surf)
{
    if (surf->specpts != NULL) free(surf->specpts);    
}

void surface_test_points(
    const Surface * surf, 
    const double * points, 
    size_t npts,
    int * result
)
{
    int i;
    double fval;
    for (i = 0; i < npts; ++i) {
        fval = surface_func(NDIM, points + NDIM * i, NULL, (void*) surf);
        result[i] = (int) copysign(1, fval);
    }
}

int surface_test_box(Surface * surf, const Box * box)
{
    if (surf->last_box != 0) {
        int bc = box_is_in(box, surf->last_box);
        if (bc == 0 || bc > 0 && surf->last_box_result != 0) 
            return surf->last_box_result; 
    }
    
    int corner_tests[NCOR];
    surface_test_points(surf, box->corners, NCOR, corner_tests);
    int mins = 1, maxs = -1, i;

    for (i = 0; i < NCOR; ++i) {
        if (corner_tests[i] < mins) mins = corner_tests[i];
        if (corner_tests[i] > maxs) maxs = corner_tests[i];
    }
    int sign = (int) copysign(1, mins + maxs);
    
    if (sign != 0 && surf->type != PLANE) {
        if (surf->type == TORUS && ((Torus*) surf)->specpts != NULL) {
            int test_res[2];
            box_test_points(box, ((Torus*) surf)->specpts, 2, test_res);
            if (test_res[0] == 1 || test_res[1] == 1) return 0;
        }
        
        double x[NDIM], opt_val;
        nlopt_result opt_result;
        
        nlopt_opt opt;
        opt = nlopt_create(NLOPT_LD_SLSQP, 3);
        nlopt_set_lower_bounds(opt, box->lb);
        nlopt_set_upper_bounds(opt, box->ub);
        
        if (sign > 0) nlopt_set_min_objective(opt, surface_func, surf);
        else          nlopt_set_max_objective(opt, surface_func, surf);
    
        nlopt_add_inequality_mconstraint(opt, 6, box_ieqcons, (void*) box, NULL);
        nlopt_set_stopval(opt, 0);
        nlopt_set_maxeval(opt, 1000);
        
        for (i = 0; i < NCOR; ++i) {
            cblas_dcopy(NDIM, box->corners + i * NDIM, 1, x, 1);
            opt_result = nlopt_optimize(opt, x, &opt_val);
            if (sign * opt_val < 0) return 0;
        }
        nlopt_destroy(opt);
    }
    // Cash test result;
    surf->last_box = box->subdiv;
    surf->last_box_result = sign;
    
    return sign;
}

int surface_compare(const Surface * surf, const Surface * other)
{
    return surf->hash - other->hash;
}

