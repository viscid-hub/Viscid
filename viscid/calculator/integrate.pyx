# cython: boundscheck=False, wraparound=False, cdivision=True, profile=False
#
# Cython module for euler1 integration, and in the future rk4 and rk45

from __future__ import print_function
# import logging

###########
# cimports
from cython.operator cimport dereference as deref
from libc.math cimport sqrt, fabs

from cycalc cimport *

cdef extern from "math.h":
    bint isnan(double x)

#####################
# now the good stuff

cdef int _c_euler1(real_t[:,:,:,::1] s, real_t[:] *crds,
                   real_t *ds, real_t[:] x,
                   real_t tol_lo, real_t tol_hi,
                   real_t fac_refine, real_t fac_coarsen,
                   int start_inds[3]):
    cdef real_t vx, vy, vz, vmag
    vx = _c_trilin_interp[real_t](s, 0, crds, x, start_inds)
    vy = _c_trilin_interp[real_t](s, 1, crds, x, start_inds)
    vz = _c_trilin_interp[real_t](s, 2, crds, x, start_inds)
    vmag = sqrt(vx**2 + vy**2 + vz**2)
    if vmag == 0.0 or isnan(vmag):
        # logging.warning("vmag issue at: {0} {1} {2}".format(x[0], x[1], x[2]))
        return 1
    x[0] += deref(ds) * vz / vmag
    x[1] += deref(ds) * vy / vmag
    x[2] += deref(ds) * vx / vmag
    return 0

cdef int _c_rk2(real_t[:,:,:,::1] s, real_t[:] *crds,
                real_t *ds, real_t[:] x0,
                real_t tol_lo, real_t tol_hi,
                real_t fac_refine, real_t fac_coarsen,
                int start_inds[3]):
    cdef real_t[3] x1
    cdef real_t[3] v0
    cdef real_t[3] v1
    cdef real_t vmag0, vmag1
    cdef real_t ds_half = 0.5 * deref(ds)
    v0[2] = _c_trilin_interp[real_t](s, 0, crds, x0, start_inds)
    v0[1] = _c_trilin_interp[real_t](s, 1, crds, x0, start_inds)
    v0[0] = _c_trilin_interp[real_t](s, 2, crds, x0, start_inds)
    vmag0 = sqrt(v0[0]**2 + v0[1]**2 + v0[2]**2)
    # logging.info("x0: {0} | v0 : | vmag0: {1}".format([x0[0], x0[1], x0[2]], vmag0))
    if vmag0 == 0.0 or isnan(vmag0):
        # logging.warning("vmag0 issue at: {0} {1} {2}".format(x0[0], x0[1], x0[2]))
        return 1
    x1[0] = x0[0] + ds_half * v0[0] / vmag0
    x1[1] = x0[1] + ds_half * v0[1] / vmag0
    x1[2] = x0[2] + ds_half * v0[2] / vmag0

    v1[2] = _c_trilin_interp[real_t](s, 0, crds, x1, start_inds)
    v1[1] = _c_trilin_interp[real_t](s, 1, crds, x1, start_inds)
    v1[0] = _c_trilin_interp[real_t](s, 2, crds, x1, start_inds)
    vmag1 = sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    # logging.info("x1: {0} | v0 : | vmag1: {1}".format([x1[0], x1[1], x1[2]], vmag1))
    if vmag1 == 0.0 or isnan(vmag1):
        # logging.warning("vmag1 issue at: {0} {1} {2}".format(x1[0], x1[1], x1[2]))
        return 1
    x0[0] += deref(ds) * v1[0] / vmag1
    x0[1] += deref(ds) * v1[1] / vmag1
    x0[2] += deref(ds) * v1[2] / vmag1
    return 0

cdef int _c_rk12(real_t[:,:,:,::1] s, real_t[:] *crds,
                 real_t *ds, real_t[:] x0,
                 real_t tol_lo, real_t tol_hi,
                 real_t fac_refine, real_t fac_coarsen,
                 int start_inds[3]):
    cdef real_t[3] x1
    cdef real_t[3] v0
    cdef real_t[3] v1
    cdef real_t[3] x_first
    cdef real_t[3] x_second
    cdef real_t vmag0, vmag1
    cdef real_t dist
    cdef real_t ds_half

    while True:
        ds_half = 0.5 * deref(ds)

        # print("A", start_inds[0], start_inds[1], start_inds[2])
        v0[2] = _c_trilin_interp[real_t](s, 0, crds, x0, start_inds)
        v0[1] = _c_trilin_interp[real_t](s, 1, crds, x0, start_inds)
        v0[0] = _c_trilin_interp[real_t](s, 2, crds, x0, start_inds)
        vmag0 = sqrt(v0[0]**2 + v0[1]**2 + v0[2]**2)
        # logging.info("x0: {0} | v0 : | vmag0: {1}".format([x0[0], x0[1], x0[2]], vmag0))
        if vmag0 == 0.0 or isnan(vmag0):
            # logging.warning("vmag0 issue at: {0} {1} {2}".format(x0[0], x0[1], x0[2]))
            return 1

        x_first[0] = x0[0] + deref(ds) * v0[0] / vmag0
        x_first[1] = x0[1] + deref(ds) * v0[1] / vmag0
        x_first[2] = x0[2] + deref(ds) * v0[2] / vmag0

        x1[0] = x0[0] + ds_half * v0[0] / vmag0
        x1[1] = x0[1] + ds_half * v0[1] / vmag0
        x1[2] = x0[2] + ds_half * v0[2] / vmag0

        # print("B", start_inds[0], start_inds[1], start_inds[2])
        v1[2] = _c_trilin_interp[real_t](s, 0, crds, x1, start_inds)
        v1[1] = _c_trilin_interp[real_t](s, 1, crds, x1, start_inds)
        v1[0] = _c_trilin_interp[real_t](s, 2, crds, x1, start_inds)
        vmag1 = sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
        # logging.info("x1: {0} | v0 : | vmag1: {1}".format([x1[0], x1[1], x1[2]], vmag1))
        if vmag1 == 0.0 or isnan(vmag1):
            # logging.warning("vmag1 issue at: {0} {1} {2}".format(x1[0], x1[1], x1[2]))
            return 1
        x_second[0] = x0[0] + deref(ds) * v1[0] / vmag1
        x_second[1] = x0[1] + deref(ds) * v1[1] / vmag1
        x_second[2] = x0[2] + deref(ds) * v1[2] / vmag1

        dist = sqrt((x_second[0] - x_first[0])**2 + \
                    (x_second[1] - x_first[1])**2 + \
                    (x_second[2] - x_first[2])**2)

        if dist > tol_hi * fabs(deref(ds)):
            # logging.debug("Refining ds: {0} -> {1}".format(
            #     deref(ds), fac_refine * deref(ds)))
            ds[0] = fac_refine * deref(ds)
            continue
        elif dist < tol_lo * fabs(deref(ds)):
            # logging.debug("Coarsening ds: {0} -> {1}".format(
            #     deref(ds), fac_coarsen * deref(ds)))
            ds[0] = fac_coarsen * deref(ds)
            break
        else:
            break

    x0[0] = x_second[0]
    x0[1] = x_second[1]
    x0[2] = x_second[2]
    return 0

cdef int _c_euler1a(real_t[:,:,:,::1] s, real_t[:] *crds,
                    real_t *ds, real_t[:] x0,
                    real_t tol_lo, real_t tol_hi,
                    real_t fac_refine, real_t fac_coarsen,
                    int start_inds[3]):
    cdef real_t[3] x1
    cdef real_t[3] x2
    cdef real_t[3] v0
    cdef real_t[3] v1
    cdef real_t vmag0, vmag1, dist

    while True:
        # go forward
        v0[2] = _c_trilin_interp[real_t](s, 0, crds, x0, start_inds)
        v0[1] = _c_trilin_interp[real_t](s, 1, crds, x0, start_inds)
        v0[0] = _c_trilin_interp[real_t](s, 2, crds, x0, start_inds)
        vmag0 = sqrt(v0[0]**2 + v0[1]**2 + v0[2]**2)
        # logging.info("x0: {0} | v0 : | vmag0: {1}".format([x0[0], x0[1], x0[2]], vmag0))
        if vmag0 == 0.0 or isnan(vmag0):
            # logging.warning("vmag0 issue at: {0} {1} {2}".format(x0[0], x0[1], x0[2]))
            return 1

        x1[0] = x0[0] + deref(ds) * v0[0] / vmag0
        x1[1] = x0[1] + deref(ds) * v0[1] / vmag0
        x1[2] = x0[2] + deref(ds) * v0[2] / vmag0

        # now go backward
        v1[2] = _c_trilin_interp[real_t](s, 0, crds, x1, start_inds)
        v1[1] = _c_trilin_interp[real_t](s, 1, crds, x1, start_inds)
        v1[0] = _c_trilin_interp[real_t](s, 2, crds, x1, start_inds)
        vmag1 = sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)

        # logging.info("x1: {0} | v0 : | vmag1: {1}".format([x1[0], x1[1], x1[2]], vmag1))
        if vmag1 == 0.0 or isnan(vmag1):
            # logging.warning("vmag1 issue at: {0} {1} {2}".format(x1[0], x1[1], x1[2]))
            return 1

        x2[0] = x1[0] - deref(ds) * v1[0] / vmag1
        x2[1] = x1[1] - deref(ds) * v1[1] / vmag1
        x2[2] = x1[2] - deref(ds) * v1[2] / vmag1

        dist = sqrt((x2[0] - x0[0])**2 + \
                    (x2[1] - x0[1])**2 + \
                    (x2[2] - x0[2])**2)

        if dist > tol_hi * fabs(deref(ds)):
            # logging.debug("Refining ds: {0} -> {1}".format(
            #     deref(ds), fac_refine * deref(ds)))
            ds[0] = fac_refine * deref(ds)
            continue
        elif dist < tol_lo * fabs(deref(ds)):
            # logging.debug("Coarsening ds: {0} -> {1}".format(
            #     deref(ds), fac_coarsen * deref(ds)))
            ds[0] = fac_coarsen * deref(ds)
            break
        else:
            break

    x0[0] = x1[0]
    x0[1] = x1[1]
    x0[2] = x1[2]
    return 0

##
## EOF
##
