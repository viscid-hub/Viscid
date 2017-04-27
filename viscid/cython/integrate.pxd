from viscid.cython.cyfield cimport real_t
from viscid.cython.cyfield cimport FusedField

cdef int _c_euler1(FusedField fld, real_t x[3], real_t *ds,
                   real_t max_error, real_t smallest_ds, real_t largest_ds,
                   real_t vscale[3], int cached_idx3[3]) nogil except -1

cdef int _c_rk2(FusedField fld, real_t x[3], real_t *ds,
                real_t max_error, real_t smallest_ds, real_t largest_ds,
                real_t vscale[3], int cached_idx3[3]) nogil except -1

cdef int _c_rk4(FusedField fld, real_t x[3], real_t *ds,
                real_t max_error, real_t smallest_ds, real_t largest_ds,
                real_t vscale[3], int cached_idx3[3]) nogil except -1


cdef int _c_euler1a(FusedField fld, real_t x[3], real_t *ds,
                    real_t max_error, real_t smallest_ds, real_t largest_ds,
                    real_t vscale[3], int cached_idx3[3]) nogil except -1

cdef int _c_rk12(FusedField fld, real_t x[3], real_t *ds,
                 real_t max_error, real_t smallest_ds, real_t largest_ds,
                 real_t vscale[3], int cached_idx3[3]) nogil except -1

cdef int _c_rk45(FusedField fld, real_t x[3], real_t *ds,
                 real_t max_error, real_t smallest_ds, real_t largest_ds,
                 real_t vscale[3], int cached_idx3[3]) nogil except -1
