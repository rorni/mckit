cdef extern from "../src/box.h":
    ctypedef struct Box:
        pass
    
    int box_init(Box * box, const double * center, 
                 const double * ex, 
                 const double * ey, 
                 const double * ez,
                 double xdim, double ydim, double zdim)
                   
    void box_dispose(Box * box)
    
    void box_generate_random_points(Box * box, double * points, int npts)

    void box_test_points(const Box * box, const double * points, 
                         int npts, int * result)

    int box_split(const Box * box, Box * box1, Box * box2, int dir, double ratio)
      
    void box_ieqcons(unsigned int m, double * result,
                     unsigned int n, const double * x,
                     double * grad, void * f_data)