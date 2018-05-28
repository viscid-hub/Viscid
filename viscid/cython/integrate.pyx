# cython: boundscheck=False, wraparound=False, cdivision=True, profile=False
# cython: emit_code_comments=False

"""3D Integrators of Vector Fields with 3 Components"""

from __future__ import print_function
from viscid import logger

from cython.operator cimport dereference as deref
from libc.math cimport sqrt, isnan, fabs

from viscid.cython.cyfield cimport real_t
from viscid.cython.cyfield cimport FusedField
from viscid.cython.cycalc cimport _c_interp_trilin
from viscid.cython.misc_inlines cimport real_min, real_max


# hack for Anaconda py27 on Windows which uses MSVC2008, which doesn't
# define libc.math.isnan
cdef extern from *:
    """
    #if defined(_MSC_VER) && _MSC_VER < 1900
    #include <float.h>
    #define isnan(f) _isnan((f))
    #endif
    """
    pass


##################################
####   non-adaptive methods   ####
##################################

cdef int _c_euler1(FusedField fld, real_t x[3], real_t *ds, real_t *dt,
                   real_t max_error, real_t smallest_ds, real_t largest_ds,
                   real_t vscale[3], int cached_idx3[3]) nogil except -1:
    """Simplest 1st order euler integration"""
    cdef real_t v[3]
    cdef real_t vmag
    v[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x, cached_idx3)
    v[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x, cached_idx3)
    v[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x, cached_idx3)
    vmag = sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if vmag == 0.0 or isnan(vmag):
        # logger.warning("vmag issue at: {0} {1} {2}, [{3}, {4}, {5}] == |{6}|".format(
        #                x[0], x[1], x[2], vx, vy, vz, vmag))
        return 1
    x[0] += deref(ds) * v[0] / vmag
    x[1] += deref(ds) * v[1] / vmag
    x[2] += deref(ds) * v[2] / vmag

    if dt != NULL:
        dt[0] += ds[0] / vmag

    return 0

cdef int _c_rk2(FusedField fld, real_t x[3], real_t *ds, real_t *dt,
                real_t max_error, real_t smallest_ds, real_t largest_ds,
                real_t vscale[3], int cached_idx3[3]) nogil except -1:
    """Runge-Kutta midpoint 2nd order integrator

    To get the point x^{l+1} from x^l (note that k_i and x_i are
    3-vectors in xyz space):

        h = ds = spatial step
        x_i = \sum_{j<i} c_{ij} * k_j
        k_i = h * fld[x_i] / |fld[x_i]|

        x^{l+1} = x^l + \sum_i w_i * k_i  <- 2nd order
    """
    # k's have same units as ds
    cdef real_t k1[3]
    cdef real_t k2[3]
    cdef real_t x1[3]
    cdef real_t x2[3]
    cdef real_t x_rk2[3]
    cdef real_t kmag1, kmag2

    cdef real_t h
    # c_{ij} used to calculate k_i (from a butcher table)
    cdef real_t c21 = 0.5
    # w_i used to calculate 2nd order solution (from a butcher table)
    cdef real_t w1 = 0.0
    cdef real_t w2 = 1.0

    x1[0] = x[0]
    x1[1] = x[1]
    x1[2] = x[2]

    h = deref(ds)

    # k1
    k1[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x1, cached_idx3)
    k1[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x1, cached_idx3)
    k1[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x1, cached_idx3)
    kmag1 = sqrt(k1[0]**2 + k1[1]**2 + k1[2]**2)
    if kmag1 == 0.0 or isnan(kmag1):
        # logger.warning("kmag1 issue at: {0} {1} {2}".format(x[0], x[1], x[2]))
        return 1
    k1[0] *= h / kmag1
    k1[1] *= h / kmag1
    k1[2] *= h / kmag1

    x2[0] = x1[0] + (c21 * k1[0])
    x2[1] = x1[1] + (c21 * k1[1])
    x2[2] = x1[2] + (c21 * k1[2])

    # k2
    k2[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x2, cached_idx3)
    k2[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x2, cached_idx3)
    k2[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x2, cached_idx3)
    kmag2 = sqrt(k2[0]**2 + k2[1]**2 + k2[2]**2)
    if kmag2 == 0.0 or isnan(kmag2):
        # logger.warning("kmag2 issue at: {0} {1} {2}".format(x[0], x[1], x[2]))
        return 1
    k2[0] *= h / kmag2
    k2[1] *= h / kmag2
    k2[2] *= h / kmag2

    x[0] += (w1 * k1[0] + w2 * k2[0])
    x[1] += (w1 * k1[1] + w2 * k2[1])
    x[2] += (w1 * k1[2] + w2 * k2[2])

    if dt != NULL:
        dt[0] += h / (w1 * kmag1 + w2 * kmag2)

    return 0

cdef int _c_rk4(FusedField fld, real_t x[3], real_t *ds, real_t *dt,
                real_t max_error, real_t smallest_ds, real_t largest_ds,
                real_t vscale[3], int cached_idx3[3]) nogil except -1:
    """Runge-Kutta 4th order integrator

    To get the point x^{l+1} from x^l (note that k_i and x_i are
    3-vectors in xyz space):

        h = ds = spatial step
        x_i = \sum_{j<i} c_{ij} * k_j
        k_i = h * fld[x_i] / |fld[x_i]|

        x^{l+1} = x^l + \sum_i u_i * k_i  <- 4th order
    """
    # k's have same units as ds
    cdef real_t k1[3]
    cdef real_t k2[3]
    cdef real_t k3[3]
    cdef real_t k4[3]
    cdef real_t x1[3]
    cdef real_t x2[3]
    cdef real_t x3[3]
    cdef real_t x4[3]
    cdef real_t kmag1, kmag2, kmag3, kmag4

    cdef real_t h
    # c_{ij} used to calculate k_i (from a butcher table)
    cdef real_t c21 = 0.5
    cdef real_t c31 = 0.0
    cdef real_t c32 = 0.5
    cdef real_t c41 = 0.0
    cdef real_t c42 = 0.0
    cdef real_t c43 = 1.0
    # u_i used to calculate 4th order solution (from a butcher table)
    cdef real_t u1 = 1.0 / 6.0
    cdef real_t u2 = 1.0 / 3.0
    cdef real_t u3 = 1.0 / 3.0
    cdef real_t u4 = 1.0 / 6.0

    x1[0] = x[0]
    x1[1] = x[1]
    x1[2] = x[2]

    h = deref(ds)

    # k1
    k1[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x1, cached_idx3)
    k1[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x1, cached_idx3)
    k1[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x1, cached_idx3)
    kmag1 = sqrt(k1[0]**2 + k1[1]**2 + k1[2]**2)
    if kmag1 == 0.0 or isnan(kmag1):
        # logger.warning("kmag1 issue at: {0} {1} {2}".format(x[0], x[1], x[2]))
        return 1
    k1[0] *= h / kmag1
    k1[1] *= h / kmag1
    k1[2] *= h / kmag1

    x2[0] = x1[0] + (c21 * k1[0])
    x2[1] = x1[1] + (c21 * k1[1])
    x2[2] = x1[2] + (c21 * k1[2])

    # k2
    k2[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x2, cached_idx3)
    k2[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x2, cached_idx3)
    k2[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x2, cached_idx3)
    kmag2 = sqrt(k2[0]**2 + k2[1]**2 + k2[2]**2)
    if kmag2 == 0.0 or isnan(kmag2):
        # logger.warning("kmag2 issue at: {0} {1} {2}".format(x[0], x[1], x[2]))
        return 1
    k2[0] *= h / kmag2
    k2[1] *= h / kmag2
    k2[2] *= h / kmag2

    x3[0] = x1[0] + (c31 * k1[0] + c32 * k2[0])
    x3[1] = x1[1] + (c31 * k1[1] + c32 * k2[1])
    x3[2] = x1[2] + (c31 * k1[2] + c32 * k2[2])

    # k3
    k3[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x3, cached_idx3)
    k3[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x3, cached_idx3)
    k3[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x3, cached_idx3)
    kmag3 = sqrt(k3[0]**2 + k3[1]**2 + k3[2]**2)
    if kmag3 == 0.0 or isnan(kmag3):
        # logger.warning("kmag3 issue at: {0} {1} {2}".format(x[0], x[1], x[2]))
        return 1
    k3[0] *= h / kmag3
    k3[1] *= h / kmag3
    k3[2] *= h / kmag3

    x4[0] = x1[0] + (c41 * k1[0] + c42 * k2[0] + c43 * k3[0])
    x4[1] = x1[1] + (c41 * k1[1] + c42 * k2[1] + c43 * k3[1])
    x4[2] = x1[2] + (c41 * k1[2] + c42 * k2[2] + c43 * k3[2])

    # k4
    k4[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x4, cached_idx3)
    k4[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x4, cached_idx3)
    k4[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x4, cached_idx3)
    kmag4 = sqrt(k4[0]**2 + k4[1]**2 + k4[2]**2)
    if kmag4 == 0.0 or isnan(kmag4):
        # logger.warning("kmag4 issue at: {0} {1} {2}".format(x[0], x[1], x[2]))
        return 1
    k4[0] *= h / kmag4
    k4[1] *= h / kmag4
    k4[2] *= h / kmag4

    x[0] += (u1 * k1[0] + u2 * k2[0] + u3 * k3[0] + u4 * k4[0])
    x[1] += (u1 * k1[1] + u2 * k2[1] + u3 * k3[1] + u4 * k4[1])
    x[2] += (u1 * k1[2] + u2 * k2[2] + u3 * k3[2] + u4 * k4[2])

    if dt != NULL:
        dt[0] += h / (u1 * kmag1 + u2 * kmag2 + u3 * kmag3 + u4 * kmag4)

    return 0

##############################
####   adaptive methods   ####
##############################

cdef inline int _ts_ctrl(real_t *ds, real_t error_estimate, real_t max_error,
                         real_t smallest_ds, real_t largest_ds) nogil:
    """adjust step size and return whether or not to repeat the previous step"""
    cdef int repeat
    cdef real_t err_ratio = fabs(error_estimate) / max_error
    cdef real_t abs_ds = fabs(deref(ds))
    cdef real_t ds_sign = ds[0] / abs_ds

    # if fabs(ds_sign) - 1.0 != 0.0:
    #     with gil:
    #         print("non-unity sign:", ds_sign)

    if err_ratio == 0.0:
        repeat = 0
    elif err_ratio >= 1.0:
        if abs_ds <= smallest_ds:
            # display warning here?
            ds[0] = ds_sign * smallest_ds
            repeat = 0
        else:
            ds[0] = 0.9 * deref(ds) * (err_ratio ** -0.25)
            if fabs(ds[0]) <= smallest_ds:
                ds[0] = ds_sign * smallest_ds
            repeat = 1
    else:
        if abs_ds >= largest_ds:
            ds[0] = ds_sign * largest_ds
            repeat = 0
        else:
            ds[0] = 0.9 * deref(ds) * (err_ratio ** -0.2)
            if fabs(ds[0]) >= largest_ds:
                ds[0] = ds_sign * largest_ds
            repeat = 0

    # if isnan(ds[0]) or ds[0] == 0.0:
    #     with gil:
    #         print("ds ->", ds[0], "problem:", ds_sign, abs_ds, error_estimate,
    #               max_error, err_ratio, "SM", smallest_ds, "LG", largest_ds)

    return repeat

cdef int _c_euler1a(FusedField fld, real_t x[3], real_t *ds, real_t *dt,
                    real_t max_error, real_t smallest_ds, real_t largest_ds,
                    real_t vscale[3], int cached_idx3[3]) nogil except -1:
    """Adaptive euler, estimates error with a backward step"""
    # k's have same units as ds
    cdef real_t k1[3]
    cdef real_t k2[3]
    cdef real_t x1[3]
    cdef real_t x2[3]
    cdef real_t x3[3]
    cdef real_t kmag1, kmag2
    cdef real_t xdiff[3]
    cdef real_t err_estimate

    cdef real_t h

    x1[0] = x[0]
    x1[1] = x[1]
    x1[2] = x[2]

    while True:
        h = deref(ds)

        # k1
        k1[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x1, cached_idx3)
        k1[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x1, cached_idx3)
        k1[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x1, cached_idx3)
        kmag1 = sqrt(k1[0]**2 + k1[1]**2 + k1[2]**2)
        if kmag1 == 0.0 or isnan(kmag1):
            # logger.warning("kmag1 issue at: {0} {1} {2}".format(x[0], x[1], x[2]))
            return 1
        k1[0] *= h / kmag1
        k1[1] *= h / kmag1
        k1[2] *= h / kmag1

        x2[0] = x1[0] + k1[0]
        x2[1] = x1[1] + k1[1]
        x2[2] = x1[2] + k1[2]

        # k2
        k2[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x2, cached_idx3)
        k2[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x2, cached_idx3)
        k2[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x2, cached_idx3)
        kmag2 = sqrt(k2[0]**2 + k2[1]**2 + k2[2]**2)
        if kmag2 == 0.0 or isnan(kmag2):
            # logger.warning("kmag2 issue at: {0} {1} {2}".format(x[0], x[1], x[2]))
            return 1
        k2[0] *= h / kmag2
        k2[1] *= h / kmag2
        k2[2] *= h / kmag2

        # x1 + an euler step forward - an euler step backward
        # should be close to where we started (x1)
        x3[0] = x1[0] + (k1[0] - k2[0])
        x3[1] = x1[1] + (k1[1] - k2[1])
        x3[2] = x1[2] + (k1[2] - k2[2])

        xdiff[0] = x3[0] - x1[0]
        xdiff[1] = x3[1] - x1[1]
        xdiff[2] = x3[2] - x1[2]
        err_estimate = sqrt(xdiff[0]**2 + xdiff[1]**2 + xdiff[2]**2) / h

        if _ts_ctrl(ds, err_estimate, max_error, smallest_ds, largest_ds):
            continue
        else:
            break

    x[0] = x2[0]
    x[1] = x2[1]
    x[2] = x2[2]

    if dt != NULL:
        dt[0] += h / kmag1

    return 0

cdef int _c_rk12(FusedField fld, real_t x[3], real_t *ds, real_t *dt,
                 real_t max_error, real_t smallest_ds, real_t largest_ds,
                 real_t vscale[3], int cached_idx3[3]) nogil except -1:
    """Runge-Kutta adaptive midpoint 1st order integrator

    To get the point x^{l+1} from x^l (note that k_i and x_i are
    3-vectors in xyz space):

        h = ds = spatial step
        x_i = \sum_{j<i} c_{ij} * k_j
        k_i = h * fld[x_i] / |fld[x_i]|

        x^{l+1} = x^l + \sum_i u_i * k_i  <- 1st order
        x^{l+1} = x^l + \sum_i w_i * k_i  <- 2nd order
    """
    # k's have same units as ds
    cdef real_t k1[3]
    cdef real_t k2[3]
    cdef real_t x1[3]
    cdef real_t x2[3]
    cdef real_t x_rk1[3]
    cdef real_t x_rk2[3]
    cdef real_t kmag1, kmag2
    cdef real_t xdiff[3]
    cdef real_t err_estimate

    cdef real_t h
    # c_{ij} used to calculate k_i (from a butcher table)
    cdef real_t c21 = 0.5
    # u_i used to calculate 1st order solution (from a butcher table)
    cdef real_t u1 = 1.0
    cdef real_t u2 = 0.0
    # w_i used to calculate 2nd order solution (from a butcher table)
    cdef real_t w1 = 0.0
    cdef real_t w2 = 1.0

    x1[0] = x[0]
    x1[1] = x[1]
    x1[2] = x[2]

    while True:
        h = deref(ds)

        # k1
        k1[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x1, cached_idx3)
        k1[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x1, cached_idx3)
        k1[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x1, cached_idx3)
        kmag1 = sqrt(k1[0]**2 + k1[1]**2 + k1[2]**2)
        if kmag1 == 0.0 or isnan(kmag1):
            # logger.warning("kmag1 issue at: {0} {1} {2}".format(x[0], x[1], x[2]))
            return 1
        k1[0] *= h / kmag1
        k1[1] *= h / kmag1
        k1[2] *= h / kmag1

        x2[0] = x1[0] + (c21 * k1[0])
        x2[1] = x1[1] + (c21 * k1[1])
        x2[2] = x1[2] + (c21 * k1[2])

        # k2
        k2[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x2, cached_idx3)
        k2[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x2, cached_idx3)
        k2[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x2, cached_idx3)
        kmag2 = sqrt(k2[0]**2 + k2[1]**2 + k2[2]**2)
        if kmag2 == 0.0 or isnan(kmag2):
            # logger.warning("kmag2 issue at: {0} {1} {2}".format(x[0], x[1], x[2]))
            return 1
        k2[0] *= h / kmag2
        k2[1] *= h / kmag2
        k2[2] *= h / kmag2

        x_rk1[0] = x1[0] + (u1 * k1[0] + u2 * k2[0])
        x_rk1[1] = x1[1] + (u1 * k1[1] + u2 * k2[1])
        x_rk1[2] = x1[2] + (u1 * k1[2] + u2 * k2[2])

        x_rk2[0] = x1[0] + (w1 * k1[0] + w2 * k2[0])
        x_rk2[1] = x1[1] + (w1 * k1[1] + w2 * k2[1])
        x_rk2[2] = x1[2] + (w1 * k1[2] + w2 * k2[2])

        xdiff[0] = x_rk2[0] - x_rk1[0]
        xdiff[1] = x_rk2[1] - x_rk1[1]
        xdiff[2] = x_rk2[2] - x_rk1[2]
        err_estimate = sqrt(xdiff[0]**2 + xdiff[1]**2 + xdiff[2]**2) / h

        if _ts_ctrl(ds, err_estimate, max_error, smallest_ds, largest_ds):
            continue
        else:
            break

    x[0] = x_rk1[0]
    x[1] = x_rk1[1]
    x[2] = x_rk1[2]

    if dt != NULL:
        dt[0] += h / (w1 * kmag1 + w2 * kmag2)

    return 0

cdef int _c_rk45(FusedField fld, real_t x[3], real_t *ds, real_t *dt,
                 real_t max_error, real_t smallest_ds, real_t largest_ds,
                 real_t vscale[3], int cached_idx3[3]) nogil except -1:
    """Runge-Kutta-Fehlberg adaptive 4th order integrator

    To get the point x^{l+1} from x^l (note that k_i and x_i are
    3-vectors in xyz space):

        h = ds = spatial step
        x_i = \sum_{j<i} c_{ij} * k_j
        k_i = h * fld[x_i] / |fld[x_i]|

        x^{l+1} = x^l + \sum_i u_i * k_i  <- 4th order
        x^{l+1} = x^l + \sum_i w_i * k_i  <- 5th order
    """
    # k's have same units as ds
    cdef real_t k1[3]
    cdef real_t k2[3]
    cdef real_t k3[3]
    cdef real_t k4[3]
    cdef real_t k5[3]
    cdef real_t k6[3]
    cdef real_t x1[3]
    cdef real_t x2[3]
    cdef real_t x3[3]
    cdef real_t x4[3]
    cdef real_t x5[3]
    cdef real_t x6[3]
    cdef real_t x_rk4[3]
    cdef real_t x_rk5[3]
    cdef real_t kmag1, kmag2, kmag3, kmag4, kmag5, kmag6
    cdef real_t xdiff[3]
    cdef real_t err_estimate

    cdef real_t h
    # c_{ij} used to calculate k_i (from a butcher table)
    cdef real_t c21 = 0.25
    cdef real_t c31 = 3.0 / 32.0
    cdef real_t c32 = 9.0 / 32.0
    cdef real_t c41 = 1932.0 / 2197.0
    cdef real_t c42 = - 7200.0 / 2197.0
    cdef real_t c43 = 7296.0 / 2197.0
    cdef real_t c51 = 439.0 / 216.0
    cdef real_t c52 = - 8.0
    cdef real_t c53 = 3680.0 / 513.0
    cdef real_t c54 = - 845.0 / 4104.0
    cdef real_t c61 = - 8.0 / 27.0
    cdef real_t c62 = 2.0
    cdef real_t c63 = - 3544.0 / 2565.0
    cdef real_t c64 = 1859.0 / 4104.0
    cdef real_t c65 = - 11.0 / 40.0
    # u_i used to calculate 4th order solution (from a butcher table)
    cdef real_t u1 = 25.0 / 216.0
    cdef real_t u2 = 0.0
    cdef real_t u3 = 1408.0 / 2565.0
    cdef real_t u4 = 2197.0 / 4104.0
    cdef real_t u5 =  - 1.0 / 5.0
    cdef real_t u6 = 0.0
    # w_i used to calculate 5th order solution (from a butcher table)
    cdef real_t w1 = 16.0 / 135.0
    cdef real_t w2 = 0.0
    cdef real_t w3 = 6656.0 / 12825.0
    cdef real_t w4 = 28561.0 / 56430.0
    cdef real_t w5 = - 9.0 / 50.0
    cdef real_t w6 = 2.0 / 55.0

    x1[0] = x[0]
    x1[1] = x[1]
    x1[2] = x[2]

    while True:
        h = deref(ds)

        # k1
        k1[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x1, cached_idx3)
        k1[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x1, cached_idx3)
        k1[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x1, cached_idx3)
        kmag1 = sqrt(k1[0]**2 + k1[1]**2 + k1[2]**2)
        if kmag1 == 0.0 or isnan(kmag1):
            # logger.warning("kmag1 issue at: {0} {1} {2}".format(x[0], x[1], x[2]))
            return 1
        k1[0] *= h / kmag1
        k1[1] *= h / kmag1
        k1[2] *= h / kmag1

        x2[0] = x1[0] + (c21 * k1[0])
        x2[1] = x1[1] + (c21 * k1[1])
        x2[2] = x1[2] + (c21 * k1[2])

        # k2
        k2[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x2, cached_idx3)
        k2[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x2, cached_idx3)
        k2[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x2, cached_idx3)
        kmag2 = sqrt(k2[0]**2 + k2[1]**2 + k2[2]**2)
        if kmag2 == 0.0 or isnan(kmag2):
            # logger.warning("kmag2 issue at: {0} {1} {2}".format(x[0], x[1], x[2]))
            return 1
        k2[0] *= h / kmag2
        k2[1] *= h / kmag2
        k2[2] *= h / kmag2

        x3[0] = x1[0] + (c31 * k1[0] + c32 * k2[0])
        x3[1] = x1[1] + (c31 * k1[1] + c32 * k2[1])
        x3[2] = x1[2] + (c31 * k1[2] + c32 * k2[2])

        # k3
        k3[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x3, cached_idx3)
        k3[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x3, cached_idx3)
        k3[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x3, cached_idx3)
        kmag3 = sqrt(k3[0]**2 + k3[1]**2 + k3[2]**2)
        if kmag3 == 0.0 or isnan(kmag3):
            # logger.warning("kmag3 issue at: {0} {1} {2}".format(x[0], x[1], x[2]))
            return 1
        k3[0] *= h / kmag3
        k3[1] *= h / kmag3
        k3[2] *= h / kmag3

        x4[0] = x1[0] + (c41 * k1[0] + c42 * k2[0] + c43 * k3[0])
        x4[1] = x1[1] + (c41 * k1[1] + c42 * k2[1] + c43 * k3[1])
        x4[2] = x1[2] + (c41 * k1[2] + c42 * k2[2] + c43 * k3[2])

        # k4
        k4[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x4, cached_idx3)
        k4[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x4, cached_idx3)
        k4[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x4, cached_idx3)
        kmag4 = sqrt(k4[0]**2 + k4[1]**2 + k4[2]**2)
        if kmag4 == 0.0 or isnan(kmag4):
            # logger.warning("kmag4 issue at: {0} {1} {2}".format(x[0], x[1], x[2]))
            return 1
        k4[0] *= h / kmag4
        k4[1] *= h / kmag4
        k4[2] *= h / kmag4

        x5[0] = x1[0] + (c51 * k1[0] + c52 * k2[0] + c53 * k3[0] + c54 * k4[0])
        x5[1] = x1[1] + (c51 * k1[1] + c52 * k2[1] + c53 * k3[1] + c54 * k4[1])
        x5[2] = x1[2] + (c51 * k1[2] + c52 * k2[2] + c53 * k3[2] + c54 * k4[2])

        # k5
        k5[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x5, cached_idx3)
        k5[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x5, cached_idx3)
        k5[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x5, cached_idx3)
        kmag5 = sqrt(k5[0]**2 + k5[1]**2 + k5[2]**2)
        if kmag5 == 0.0 or isnan(kmag5):
            # logger.warning("kmag5 issue at: {0} {1} {2}".format(x[0], x[1], x[2]))
            return 1
        k5[0] *= h / kmag5
        k5[1] *= h / kmag5
        k5[2] *= h / kmag5

        x6[0] = x1[0] + (c61 * k1[0] + c62 * k2[0] + c63 * k3[0] + c64 * k4[0] + c65 * k5[0])
        x6[1] = x1[1] + (c61 * k1[1] + c62 * k2[1] + c63 * k3[1] + c64 * k4[1] + c65 * k5[1])
        x6[2] = x1[2] + (c61 * k1[2] + c62 * k2[2] + c63 * k3[2] + c64 * k4[2] + c65 * k5[2])

        # k6
        k6[0] = vscale[0] * _c_interp_trilin[FusedField, real_t](fld, 0, x6, cached_idx3)
        k6[1] = vscale[1] * _c_interp_trilin[FusedField, real_t](fld, 1, x6, cached_idx3)
        k6[2] = vscale[2] * _c_interp_trilin[FusedField, real_t](fld, 2, x6, cached_idx3)
        kmag6 = sqrt(k6[0]**2 + k6[1]**2 + k6[2]**2)
        if kmag6 == 0.0 or isnan(kmag6):
            # logger.warning("kmag6 issue at: {0} {1} {2}".format(x[0], x[1], x[2]))
            return 1
        k6[0] *= h / kmag6
        k6[1] *= h / kmag6
        k6[2] *= h / kmag6


        x_rk4[0] = x[0] + (u1 * k1[0] + u2 * k2[0] + u3 * k3[0] +
                           u4 * k4[0] + u5 * k5[0] + u6 * k6[0])
        x_rk4[1] = x[1] + (u1 * k1[1] + u2 * k2[1] + u3 * k3[1] +
                           u4 * k4[1] + u5 * k5[1] + u6 * k6[1])
        x_rk4[2] = x[2] + (u1 * k1[2] + u2 * k2[2] + u3 * k3[2] +
                           u4 * k4[2] + u5 * k5[2] + u6 * k6[2])

        x_rk5[0] = x[0] + (w1 * k1[0] + w2 * k2[0] + w3 * k3[0] +
                           w4 * k4[0] + w5 * k5[0] + w6 * k6[0])
        x_rk5[1] = x[1] + (w1 * k1[1] + w2 * k2[1] + w3 * k3[1] +
                           w4 * k4[1] + w5 * k5[1] + w6 * k6[1])
        x_rk5[2] = x[2] + (w1 * k1[2] + w2 * k2[2] + w3 * k3[2] +
                           w4 * k4[2] + w5 * k5[2] + w6 * k6[2])

        xdiff[0] = x_rk5[0] - x_rk4[0]
        xdiff[1] = x_rk5[1] - x_rk4[1]
        xdiff[2] = x_rk5[2] - x_rk4[2]
        err_estimate = sqrt(xdiff[0]**2 + xdiff[1]**2 + xdiff[2]**2) / h

        if _ts_ctrl(ds, err_estimate, max_error, smallest_ds, largest_ds):
            continue
        else:
            break

    x[0] = x_rk4[0]
    x[1] = x_rk4[1]
    x[2] = x_rk4[2]

    if dt != NULL:
        dt[0] += h / (w1 * kmag1 + w2 * kmag2 + w3 * kmag3 + w4 * kmag4 +
                      w5 * kmag5 + w6 * kmag6)

    return 0

##
## EOF
##
