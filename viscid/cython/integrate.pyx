# cython: boundscheck=False, wraparound=False, cdivision=True, profile=False
"""3D Integrators of Vector Fields with 3 Components"""
from __future__ import print_function
from viscid import logger

from cython.operator cimport dereference as deref
from libc.math cimport sqrt, isnan, fabs

from viscid.cython.cyfield cimport real_t
from viscid.cython.cyfield cimport FusedField
from viscid.cython.cycalc cimport _c_interp_trilin
from viscid.cython.misc_inlines cimport real_min, real_max


cdef int _c_euler1(FusedField fld, real_t x[3], real_t *ds,
                   real_t tol_lo, real_t tol_hi,
                   real_t fac_refine, real_t fac_coarsen,
                   real_t smallest_step, real_t largest_step,
                   real_t vscale[3]) nogil except -1:
    """Simplest 1st order euler integration"""
    cdef real_t v[3]
    cdef real_t vmag
    v[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x)
    v[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x)
    v[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x)
    vmag = sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if vmag == 0.0 or isnan(vmag):
        # logger.warn("vmag issue at: {0} {1} {2}, [{3}, {4}, {5}] == |{6}|".format(
        #                 x[0], x[1], x[2], vx, vy, vz, vmag))
        return 1
    x[0] += deref(ds) * v[0] / vmag
    x[1] += deref(ds) * v[1] / vmag
    x[2] += deref(ds) * v[2] / vmag
    return 0

cdef int _c_rk2(FusedField fld, real_t x[3], real_t *ds,
                real_t tol_lo, real_t tol_hi,
                real_t fac_refine, real_t fac_coarsen,
                real_t smallest_step, real_t largest_step,
                real_t vscale[3]) nogil except -1:
    """Runge-Kutta 2nd order integrator"""
    cdef real_t x1[3]
    cdef real_t v0[3]
    cdef real_t v1[3]
    cdef real_t vmag0, vmag1
    cdef real_t ds_half = 0.5 * deref(ds)
    v0[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x)
    v0[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x)
    v0[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x)
    vmag0 = sqrt(v0[0]**2 + v0[1]**2 + v0[2]**2)
    # logger.info("x: {0} | v0 : | vmag0: {1}".format([x[0], x[1], x[2]], vmag0))
    if vmag0 == 0.0 or isnan(vmag0):
        # logger.warn("vmag0 issue at: {0} {1} {2}".format(x[0], x[1], x[2]))
        return 1
    x1[0] = x[0] + ds_half * v0[0] / vmag0
    x1[1] = x[1] + ds_half * v0[1] / vmag0
    x1[2] = x[2] + ds_half * v0[2] / vmag0

    v1[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x1)
    v1[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x1)
    v1[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x1)
    vmag1 = sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    # logger.info("x1: {0} | v0 : | vmag1: {1}".format([x1[0], x1[1], x1[2]], vmag1))
    if vmag1 == 0.0 or isnan(vmag1):
        # logger.warn("vmag1 issue at: {0} {1} {2}".format(x1[0], x1[1], x1[2]))
        return 1
    x[0] += deref(ds) * v1[0] / vmag1
    x[1] += deref(ds) * v1[1] / vmag1
    x[2] += deref(ds) * v1[2] / vmag1
    return 0

cdef int _c_rk12(FusedField fld, real_t x[3], real_t *ds,
                 real_t tol_lo, real_t tol_hi,
                 real_t fac_refine, real_t fac_coarsen,
                 real_t smallest_step, real_t largest_step,
                 real_t vscale[3]) nogil except -1:
    """Runge-Kutta 2nd order integrator with adaptive step based on
    difference from 1st order euler
    """
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
        v0[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x)
        v0[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x)
        v0[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x)
        vmag0 = sqrt(v0[0]**2 + v0[1]**2 + v0[2]**2)
        # logger.info("x: {0} | v0 : | vmag0: {1}".format([x[0], x[1], x[2]], vmag0))
        if vmag0 == 0.0 or isnan(vmag0):
            # logger.warn("vmag0 issue at: {0} {1} {2}".format(x[0], x[1], x[2]))
            return 1

        x_first[0] = x[0] + deref(ds) * v0[0] / vmag0
        x_first[1] = x[1] + deref(ds) * v0[1] / vmag0
        x_first[2] = x[2] + deref(ds) * v0[2] / vmag0

        x1[0] = x[0] + ds_half * v0[0] / vmag0
        x1[1] = x[1] + ds_half * v0[1] / vmag0
        x1[2] = x[2] + ds_half * v0[2] / vmag0

        # print("B", start_inds[0], start_inds[1], start_inds[2])
        v1[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x1)
        v1[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x1)
        v1[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x1)
        vmag1 = sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
        # logger.info("x1: {0} | v0 : | vmag1: {1}".format([x1[0], x1[1], x1[2]], vmag1))
        if vmag1 == 0.0 or isnan(vmag1):
            # logger.warn("vmag1 issue at: {0} {1} {2}".format(x1[0], x1[1], x1[2]))
            return 1
        x_second[0] = x[0] + deref(ds) * v1[0] / vmag1
        x_second[1] = x[1] + deref(ds) * v1[1] / vmag1
        x_second[2] = x[2] + deref(ds) * v1[2] / vmag1

        dist = sqrt((x_second[0] - x_first[0])**2 + \
                    (x_second[1] - x_first[1])**2 + \
                    (x_second[2] - x_first[2])**2)

        if dist > tol_hi * fabs(deref(ds)):
            # logger.debug("Refining ds: {0} -> {1}".format(
            #     deref(ds), fac_refine * deref(ds)))
            if ds[0] <= smallest_step:
                break
            else:
                ds[0] = real_max(fac_refine * deref(ds), smallest_step)
                continue
        elif dist < tol_lo * fabs(deref(ds)):
            # logger.debug("Coarsening ds: {0} -> {1}".format(
            #     deref(ds), fac_coarsen * deref(ds)))
            ds[0] = real_min(fac_coarsen * deref(ds), largest_step)
            break
        else:
            break

    x[0] = x_second[0]
    x[1] = x_second[1]
    x[2] = x_second[2]
    return 0


cdef int _c_euler1a(FusedField fld, real_t x[3], real_t *ds,
                    real_t tol_lo, real_t tol_hi,
                    real_t fac_refine, real_t fac_coarsen,
                    real_t smallest_step, real_t largest_step,
                    real_t vscale[3]) nogil except -1:
    """1st order euler with adaptive step based on going backward to
    see how close we get to our starting point
    """
    cdef real_t[3] x1
    cdef real_t[3] x2
    cdef real_t[3] v0
    cdef real_t[3] v1
    cdef real_t vmag0, vmag1, dist

    while True:
        # go forward
        v0[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x)
        v0[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x)
        v0[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x)
        vmag0 = sqrt(v0[0]**2 + v0[1]**2 + v0[2]**2)
        # logger.info("x0: {0} | v0 : | vmag0: {1}".format([x0[0], x0[1], x0[2]], vmag0))
        if vmag0 == 0.0 or isnan(vmag0):
            # logger.warn("vmag0 issue at: {0} {1} {2}".format(x0[0], x0[1], x0[2]))
            return 1

        x1[0] = x[0] + deref(ds) * v0[0] / vmag0
        x1[1] = x[1] + deref(ds) * v0[1] / vmag0
        x1[2] = x[2] + deref(ds) * v0[2] / vmag0

        # now go backward
        v1[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x1)
        v1[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x1)
        v1[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x1)
        vmag1 = sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)

        # logger.info("x1: {0} | v0 : | vmag1: {1}".format([x1[0], x1[1], x1[2]], vmag1))
        if vmag1 == 0.0 or isnan(vmag1):
            # logger.warn("vmag1 issue at: {0} {1} {2}".format(x1[0], x1[1], x1[2]))
            return 1

        x2[0] = x1[0] - deref(ds) * v1[0] / vmag1
        x2[1] = x1[1] - deref(ds) * v1[1] / vmag1
        x2[2] = x1[2] - deref(ds) * v1[2] / vmag1

        dist = sqrt((x2[0] - x[0])**2 + \
                    (x2[1] - x[1])**2 + \
                    (x2[2] - x[2])**2)

        if dist > tol_hi * fabs(deref(ds)):
            # logger.debug("Refining ds: {0} -> {1}".format(
            #     deref(ds), fac_refine * deref(ds)))
            if ds[0] <= smallest_step:
                break
            else:
                ds[0] = real_max(fac_refine * deref(ds), smallest_step)
                continue
        elif dist < tol_lo * fabs(deref(ds)):
            # logger.debug("Coarsening ds: {0} -> {1}".format(
            #     deref(ds), fac_coarsen * deref(ds)))
            ds[0] = real_min(fac_coarsen * deref(ds), largest_step)
            break
        else:
            break

    x[0] = x1[0]
    x[1] = x1[1]
    x[2] = x1[2]
    return 0

##
## EOF
##
