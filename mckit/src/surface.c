#include <stdlib.h>
#include <math.h>
#include "nlopt.h"
#include "mkl.h"
#include "surface.h"

typedef struct PlaneData PlaneData;
typedef struct SphereData SphereData;
typedef struct CylinderData CylinderData;
typedef struct ConeData ConeData;
typedef struct TorusData TorusData;
typedef struct GQData GQData;

struct PlaneData {
    double norm[NDIM];
    double offset;
};

struct SphereData {
    double center[NDIM];
    double radius;
};

struct CylinderData {
    double point[NDIM];
    double axis[NDIM];
    double radius;
};

struct ConeData {
    double apex[NDIM];
    double axis[NDIM];
    double ta;
};

struct TorusData {
    double center[NDIM];
    double axis[NDIM];
    double radius;
    double a;
    double b;
    double * specpts;
};

struct GQData {
    double m[NDIM * NDIM];
    double v[NDIM];
    double k;
};

double plane_func(
    unsigned int n,
    const double * x,
    double * grad,
    void * f_data
)
{
    PlaneData * data = (PlaneData *) f_data;
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
    SphereData * data = (SphereData *) f_data;
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
    CylinderData * data = (CylinderData *) f_data;
    double a[NDIM];
    cblas_dcopy(NDIM, x, 1, a, 1);
    cblas_daxpy(NDIM, -1, data->point, 1, a, 1);
    double an = cblas_ddot(NDIM, a, 1, data->axis);
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
    ConeData * data = (ConeData *) f_data;
    double a[NDIM];
    cblas_dcopy(NDIM, x, 1, a, 1);
    cblas_daxpy(NDIM, -1, data->apex, 1, a, 1);
    double an = cblas_ddot(NDIM, a, 1, data->axis);
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
    GQData * data = (GQData *) f_data;
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
    TorusData * data = (TorusData *) f_data;
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
        PLANE:
            fval = plane_func(n, x, grad, surf->data);
            break;
        SPHERE:
            fval = sphere_func(n, x, grad, surf->data);
            break;
        CYLINDER:
            fval = cylinder_func(n, x, grad, surf->data);
            break;
        CONE:
            fval = cone_func(n, x, grad, surf->data);
            break;
        TORUS:
            fval = torus_func(n, x, grad, surf->data);
            break;
        GQUADRATIC:
            fval = gq_func(n, x, grad, surf->data);
            break;
    }
    return fval;
}

inline void surface_init(
    Surface * surf, 
    unsigned int name,
    enum Modifier modifier
)
{
    surf->name = name;
    surf->modifier = modifier;
}

int plane_init(
    Surface * surf,
    unsigned int name,
    enum Modifier modifier,
    const double * norm,
    double offset
)
{
    int i;
    surface_init(surf, name, modifier);
    surf->type = PLANE;
    surf->data = malloc(sizeof(PlaneData));
    if (surf->data == NULL) return SURFACE_FAILURE;
    surf->data->offset = offset;
    for (i = 0; i < NDIM; ++i) {
        surf->data->norm[i] = norm[i];
    }
    return SURFACE_SUCCESS;
}

int sphere_init(
    Surface * surf,
    unsigned int name,
    enum Modifier modifier,
    const double * center,
    double radius
)
{
    int i;
    surface_init(surf, name, modifier);
    surf->type = SPHERE;
    surf->data = malloc(sizeof(SphereData)); 
    if (surf->data == NULL) return SURFACE_FAILURE;
    surf->data->radius = radius;
    for (i = 0; i < NDIM; ++i) {
        surf->data->center[i] = center[i];
    }
    return SURFACE_SUCCESS;
}

int cylinder_init(
    Surface * surf,
    unsigned int name,
    enum Modifier modifier,
    const double * point,
    const double * axis,
    double radius
)
{
    int i;
    surface_init(surf, name, modifier);
    surf->type = CYLINDER;
    surf->data = malloc(sizeof(CylinderData));
    if (surf->data == NULL) return SURFACE_FAILURE;
    surf->data->radius = radius;
    for (i = 0; i < NDIM; ++i) {
        surf->data->point[i] = point[i];
        surf->data->axis[i] = axis[i];
    }
    return SURFACE_SUCCESS;
}

int cone_init(
    Surface * surf,
    unsigned int name,
    enum Modifier modifier,
    const double * apex,
    const double * axis,
    double ta;
)
{
    int i;
    surface_init(surf, name, modifier);
    surf->type = CONE;
    surf->data = malloc(sizeof(ConeData));    
    if (surf->data == NULL) return SURFACE_FAILURE;
    surf->data->ta = ta;
    for (i = 0; i < NDIM; ++i) {
        surf->data->apex[i] = apex[i];
        surf->data->axis[i] = axis[i];
    }
    return SURFACE_SUCCESS;
}

int torus_init(
    Surface * surf,
    unsigned int name,
    enum Modifier modifier,
    const double * center,
    const double * axis,
    double radius,
    double a,
    double b
)
{
    int i;
    surface_init(surf, name, modifier);
    surf->type = TORUS;
    surf->data = malloc(sizeof(TorusData));    
    if (surf->data == NULL) return SURFACE_FAILURE;
    surf->data->radius = radius;
    surf->data->a = a;
    surf->data->b = b;
    for (i = 0; i < NDIM; ++i) {
        surf->data->center[i] = center[i];
        surf->data->axis[i] = axis[i];
    }
    if (surf->data->b > surf->data->radius) {
        double offset = a * sqrt(1 - pow(radius / b, 2));
        surf->data->specpts = (double *) malloc(2 * NDIM * sizeof(double));
        if (surf->data->specpts == NULL) return SURFACE_FAILURE;
        cblas_dcopy(NDIM, center, 1, surf->data->specpts, 1);
        cblas_dcopy(NDIM, center, 1, surf->data->specpts + NDIM, 1);
        cblas_daxpy(NDIM, offset, axis, 1, surf->data->specpts, 1);
        cblas_daxpy(NDIM, -offset, axis, 1, surf->data->specpts + NDIM, 1);
    }
    return SURFACE_SUCCESS;
}

int gq_init(
    Surface * surf,
    unsigned int name,
    enum Modifier modifier,
    const double * m,
    const double * v,
    double k
)
{
    int i, j;
    surface_init(surf, name, modifier);
    surf->type = GQUADRATIC;
    surf->data = malloc(sizeof(GQData));    
    if (surf->data == NULL) return SURFACE_FAILURE;
    surf->data->k = k;
    for (i = 0; i < NDIM; ++i) {
        surf->data->v[i] = v[i];
        for (j = 0; j < NDIM; ++j) surf->data->m[i][j] = m[i][j];
    }
    return SURFACE_SUCCESS;
}

void surface_dispose(Surface * surf) {
    if (surf->type == TORUS && surf->data->specpts != NULL) {
        free(surf->data->specpts);
    }
    free(surf->data);
}

void surface_test_points(
    const Surface * surf, 
    const double * points, 
    int npts,
    int * result
)
{
    int i;
    double fval;
    for (i = 0; i < npts; ++i) {
        fval = surface_func(NDIM, points + NDIM * i, NULL, surf);
        result[i] = (int) copysign(1, fval);
    }
}

int surface_test_box(
    const Surface * surf, 
    const Box * box
)
{
    int corner_tests[NCOR];
    surface_test_points(surf, box->corners, NCOR, corner_tests);
    int mins = 1, maxs = -1, i;

    for (i = 0; i < NCOR; ++i) {
        if (corner_tests[i] < mins) mins = corner_tests[i];
        if (corner_tests[i] > maxs) maxs = corner_tests[i];
    }
    int sign = (int) copysign(1, mins + maxs);
    
    if (sign != 0 && surface->type != PLANE) {
        if (spec->type == TORUS && surf->data->specpts != NULL) {
            int test_res[2];
            box_test_points(box, surf->data->specpts, 2, test_res);
            if (test_res[0] == 1 || test_res[1] == 1) return 0;
        }
        
        double x[NDIM], opt_val;
        nlopt_result opt_result;
        
        nlopt_opt opt;
        opt = nlopt_create(NLOPT_LD_SLSQP, 3);
        nlopt_set_lower_bounds(opt, box->lb);
        nlopt_set_upper_bounds(opt, box->ub);
        
        if (sign > 0) nlopt_set_min_objective(opt, surface_func, surf)
        else          nlopt_set_max_objective(opt, surface_func, surf);
    
        nlopt_add_inequality_mconstraint(opt, 6, box_ieqcons, box, NULL);
        nlopt_set_stopval(opt, 0);
        nlopt_set_maxeval(opt, 1000);
        
        for (i = 0; i < NCOR; ++i) {
            cblas_dcopy(NDIM, box->corners[i], 1, x, 1);
            opt_result = nlopt_optimize(opt, x, &opt_val);
            if (sign * opt_val < 0) return 0;
        }
        nlopt_destroy(opt);
    }
    
    return sign;
}

