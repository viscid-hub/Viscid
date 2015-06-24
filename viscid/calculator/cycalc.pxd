from cyfield cimport real_t, fld_t
from cyfield cimport CyField, FusedField, make_cyfield

cdef inline int int_min(int a, int b):
    return b if b < a else a

cdef inline int int_max(int a, int b):
    return b if b > a else a

cdef inline real_t real_min(real_t a, real_t b):
    return b if b < a else a

cdef inline real_t real_max(real_t a, real_t b):
    return b if b > a else a

# heavy lifting interpolation functions
cdef real_t _c_interp_trilin(FusedField fld, int m, real_t x[3])
cdef real_t _c_interp_nearest(FusedField fld, int m, real_t x[3])

# finding closest indices
cdef int closest_preceeding_ind(FusedField fld, int d, real_t value)
cdef int closest_ind(FusedField fld, int d, real_t value)
