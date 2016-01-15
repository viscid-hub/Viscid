from viscid.cython.cyfield cimport real_t
from viscid.cython.cyfield cimport FusedField

# heavy lifting interpolation functions
cdef real_t _c_interp_trilin(FusedField fld, int m, real_t x[3],
                             int cached_idx3[3]) nogil
cdef real_t _c_interp_nearest(FusedField fld, int m, real_t x[3],
                              int cached_idx3[3]) nogil

# finding closest indices
# Note: can't export these symbols b/c they're defined as inline, but
#       inlining them gives a 12% performance boost
# cdef inline int closest_preceeding_ind(FusedField fld, int d, real_t value)
# cdef inline int closest_ind(FusedField fld, int d, real_t value)
