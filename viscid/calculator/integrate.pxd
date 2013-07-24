from cycalc_util cimport *

cdef int _c_euler1(real_t[:,:,:,:] s, real_t[:] crdz, real_t[:] crdy,
                   real_t[:] crdx, real_t *ds, real_t[:] x,
                   real_t tol_lo, real_t tol_hi, 
                   real_t fac_refine, real_t fac_coarsen,
                   int start_inds[3]) except -1

cdef int _c_rk2(real_t[:,:,:,:] s, real_t[:] crdz, real_t[:] crdy,
                   real_t[:] crdx, real_t *ds, real_t[:] x,
                   real_t tol_lo, real_t tol_hi, 
                   real_t fac_refine, real_t fac_coarsen,
                   int start_inds[3]) except -1

cdef int _c_rk12(real_t[:,:,:,:] s, real_t[:] crdz, real_t[:] crdy,
                   real_t[:] crdx, real_t *ds, real_t[:] x,
                   real_t tol_lo, real_t tol_hi, 
                   real_t fac_refine, real_t fac_coarsen,
                   int start_inds[3]) except -1

cdef int _c_euler1a(real_t[:,:,:,:] s, real_t[:] crdz, real_t[:] crdy,
                    real_t[:] crdx, real_t *ds, real_t[:] x0,
                    real_t tol_lo, real_t tol_hi, 
                    real_t fac_refine, real_t fac_coarsen,
                    int start_inds[3]) except -1
