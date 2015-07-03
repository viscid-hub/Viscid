from libc.math cimport fabs

from viscid.cython.cyfield cimport real_t

cdef inline int int_min(int a, int b):
    return b if b < a else a

cdef inline int int_max(int a, int b):
    return b if b > a else a

cdef inline real_t real_min(real_t a, real_t b):
    return b if b < a else a

cdef inline real_t real_max(real_t a, real_t b):
    return b if b > a else a

cdef inline int isclose(real_t a, real_t b):
    # tolerances don't need to be that exact
    cdef real_t atol = 1e-6
    cdef real_t rtol = 1e-5
    return fabs(a - b) <= (atol + rtol * fabs(b))

cdef inline int less_close(real_t a, real_t b):
    """Returns True if a and b are close, or if a <= b"""
    return a <= b or isclose(a, b)

cdef inline int greater_close(real_t a, real_t b):
    """Returns True if a and b are close, or if a >= b"""
    return a >= b or isclose(a, b)

cdef inline int less_not_close(real_t a, real_t b):
    """Returns True if a and b are not close, and if a < b"""
    return a < b and not isclose(a, b)

cdef inline int greater_not_close(real_t a, real_t b):
    """Returns True if a and b are not close, and a > b"""
    return a > b and not isclose(a, b)

##
## EOF
##
