from cycalc_util cimport *

cdef int _c_euler1(real_t[:,:,:,:] s, real_t[:] crdz, real_t[:] crdy,
                   real_t[:] crdx, real_t ds, real_t[:] x) except -1
