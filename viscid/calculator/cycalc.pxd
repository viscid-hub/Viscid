cimport numpy as cnp

from cycalc_util cimport *

cdef inline int _c_closest_ind(real_t[:] crd, real_t point, int *startind) except -1

cdef real_t _c_interp_trilin(real_t[:,:,:,::1] s, cnp.intp_t m, real_t[:] *crds,
                             real_t[:] x, int start_inds[3])

cdef fld_t _c_interp_nearest(fld_t[:,:,:,::1] s, cnp.intp_t m, real_t[:] *crds,
                             real_t[:] x, int start_inds[3],
                             fld_t fill, bint use_fill)

##
## EOF
##
