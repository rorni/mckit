from cbox cimport Box
cdef extern from "../src/geometry.h":
    ctypedef struct Node:
        pass

    ctypedef struct RBTree:
        pass

    Node * node_create(int opc, size_t alen, const void * args);

    void node_free(Node * node);

    int node_test_box(Node * node, const Box * box, char collect);

    int node_test_points(const Node * node, const double * points, size_t npts, int * result);

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
