cimport numpy as cnp

from cycalc_util cimport *

cdef inline int _c_closest_ind(real_t[:] crd, real_t point, int *startind) except -1

cdef real_t _c_trilin_interp(real_t[:,:,:,::1] s, np.intp_t m, real_t[:] *crds,
                             real_t[:] x, int start_inds[3])

##
## EOF
##
