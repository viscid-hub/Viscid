# cython: boundscheck=True, wraparound=True
#
# Cython module for euler1 integration, and in the future rk4 and rk45

from __future__ import print_function

###########
# cimports
from libc.math cimport sqrt

from cycalc cimport *

#####################
# now the good stuff

cdef int _c_euler1(real_t[:,:,:,:] s, real_t[:] crdz, real_t[:] crdy,
                   real_t[:] crdx, real_t ds, real_t[:] x) except -1:
    vx = _c_trilin_interp[real_t](s[...,0], crdz, crdy, crdx, x)
    vy = _c_trilin_interp[real_t](s[...,1], crdz, crdy, crdx, x)
    vz = _c_trilin_interp[real_t](s[...,2], crdz, crdy, crdx, x)
    vmag = sqrt(vx**2 + vy**2 + vz**2)
    if vmag == 0.0:
        return 1
    x[0] += ds * vz / vmag
    x[1] += ds * vy / vmag
    x[2] += ds * vx / vmag
    return 0

# cdef real_t _c_rk4(real_t[:,:,:] s, real_t[:] crdz, real_t[:] crdy,
#                       real_t[:] crdx, real_t[:] x):
#     _c_trilin_interp[real_t](s, crdz, crdy, crdx, x)
#     return x[0]

# cdef real_t _c_rk45(real_t[:,:,:] s, real_t[:] crdz, real_t[:] crdy,
#                        real_t[:] crdx, real_t[:] x):
#     return x[0]

##
## EOF
##
