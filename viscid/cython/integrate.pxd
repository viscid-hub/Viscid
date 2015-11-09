from viscid.cython.cyfield cimport real_t
from viscid.cython.cyfield cimport FusedField

cdef int _c_euler1(FusedField fld, real_t x[3], real_t *ds,
                   real_t tol_lo, real_t tol_hi,
                   real_t fac_refine, real_t fac_coarsen,
                   real_t smallest_step, real_t largest_step,
                   real_t vscale[3]) nogil except -1

cdef int _c_rk2(FusedField fld, real_t x[3], real_t *ds,
                real_t tol_lo, real_t tol_hi,
                real_t fac_refine, real_t fac_coarsen,
                real_t smallest_step, real_t largest_step,
                real_t vscale[3]) nogil except -1

cdef int _c_rk12(FusedField fld, real_t x[3], real_t *ds,
                 real_t tol_lo, real_t tol_hi,
                 real_t fac_refine, real_t fac_coarsen,
                 real_t smallest_step, real_t largest_step,
                 real_t vscale[3]) nogil except -1

cdef int _c_euler1a(FusedField fld, real_t x[3], real_t *ds,
                    real_t tol_lo, real_t tol_hi,
                    real_t fac_refine, real_t fac_coarsen,
                    real_t smallest_step, real_t largest_step,
                    real_t vscale[3]) nogil except -1
