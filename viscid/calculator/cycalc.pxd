from cycalc_util cimport *

cdef _c_magnitude3d(real_t[:,:,:,:] vect, real_t[:,:,:] mag)

cdef _c_div3d(real_t[:,:,:,:] vect, real_t[:] crdx, real_t[:] crdy,
              real_t[:] crdz, real_t[:,:,:] div_arr)

cdef int _c_closest_ind(real_t[:] crd, real_t point, int *startind) except -1

cdef real_t _c_trilin_interp(real_t[:,:,:] s, real_t[:] crdz,
                             real_t[:] crdy, real_t[:] crdx, real_t[:] x,
                             int start_inds[3])

##
## EOF
##
