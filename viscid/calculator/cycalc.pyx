# cython: boundscheck=False, wraparound=False, cdivision=True, profile=False
#
# Note: a _c_FUNCTION can only be called from another cdef-ed function, or
# a def-ed _py_FUNCTION function because of the use of the fused real_t
# to template both float32 and float64 versions

from __future__ import print_function

import numpy as np

from cython.operator cimport dereference as deref
from cython.view cimport array as cvarray
# from cython.parallel import prange

# from viscid import logger
from viscid import field
from viscid import coordinate
from viscid.calculator import seed

###########
# cimports
cimport cython
cimport numpy as cnp

from libc.math cimport sqrt

from cycalc_util cimport *
from cycalc cimport *

# cdef extern from "math.h":
#     bint isnan(double x)

#####################
# now the good stuff

cdef inline int _c_int_max(int a, int b):
    if a >= b:
        return a
    else:
        return b

cdef inline int _c_int_min(int a, int b):
    if a <= b:
        return a
    else:
        return b

def closest_ind(real_t[:] crd, point, int startind=0):
    cdef int i
    cdef int fallback
    cdef int n = crd.shape[0]
    cdef int forward = n > 1 and crd[1] > crd[0]
    cdef real_t pt = point

    if startind < 0:
        # startind[0] = 0
        startind = 0
    elif startind > n - 1:
        # startind[0] = n - 1
        startind = n - 1

    # search linearly... maybe branch prediction makes this better
    # than bisection for smallish arrays...
    # pt is 'upward' (index wise) of crd[startind]... only search up
    if ((forward and crd[startind] <= pt) or \
        (not forward and crd[startind] >= pt)):
        for i from startind <= i < n - 1:
            if forward and crd[i + 1] >= pt:
                # startind[0] = i
                return i
            if not forward and crd[i + 1] <= pt:
                # startind[0] = i
                return i
        # if we've gone too far, pick the last index
        fallback = _c_int_max(n - 2, 0)
        # startind[0] = fallback
        return fallback

    # startind was too large... go backwards
    for i from startind - 1 >= i >= 0:
        if forward and crd[i] <= pt:
            # startind[0] = i
            return i
        if not forward and crd[i] >= pt:
            # startind[0] = i
            return i
    # if we've gone too far, pick the first index
    fallback = 0
    # startind[0] = fallback
    return fallback

def interp_trilin(fld, seeds):
    """ Points can be list of 3-tuples or a SeedGen instance. If fld
    is a scalar field, the output array has shape (nr_points,) where nr_points
    is the number of seed points. If it's a vector, the output has shape
    (nr_points, nr_comps), where nr_comps is the number of components of the
    vector. The data type of the output is the same as the original field.
    The output is always an array, even if only one point is given.
    """
    if fld.iscentered("Cell"):
        crdz, crdy, crdx = fld.get_crds_cc()
    elif fld.iscentered("Node"):
        crdz, crdy, crdx = fld.get_crds_nc()
    else:
        raise RuntimeError("Dont touch me with that centering.")

    dtype = fld.dtype

    if isinstance(seeds, list):
        seeds = seed.Point(seeds)

    if fld.istype("Vector"):
        if not fld.layout == field.LAYOUT_INTERLACED:
            raise ValueError("Trilin interp only written for interlaced data.")
        nr_comps = fld.nr_comps
        nr_points = seeds.nr_points(center=fld.center)
        ret = np.empty((nr_points, nr_comps), dtype=dtype)

        for j from 0 <= j < nr_comps:
            # print(ret.shape, nr_points, nr_comps)
            ret[:,j] = _py_interp_trilin(dtype, fld.data, j, crdz, crdy, crdx,
                                seeds.iter_points(center=fld.center),
                                nr_points)
        return ret

    elif fld.istype("Scalar"):
        dat = fld.data.reshape(fld.shape + [1])
        nr_points = seeds.nr_points(center=fld.center)
        ret = np.empty((nr_points,), dtype=dtype)
        ret[:] = _py_interp_trilin(dtype, dat, 0, crdz, crdy, crdx,
                                   seeds.iter_points(center=fld.center),
                                   nr_points)
        return ret

    else:
        raise RuntimeError("That centering is not supported for interp_trilin")

def _py_interp_trilin(dtype, real_t[:,:,:,::1] s, cnp.intp_t m,
                      crdz_in, crdy_in, crdx_in, points, int nr_points):
    """ return the scalar value of 3d scalar array s trilinearly interpolated
    to the point x (in z, y, x order) """
    cdef unsigned int i
    cdef real_t[:] crdz = crdz_in
    cdef real_t[:] crdy = crdy_in
    cdef real_t[:] crdx = crdx_in
    cdef real_t[:] *crds = [crdz, crdy, crdx]
    cdef int* start_inds = [0, 0, 0]

    cdef real_t[:] x = np.empty((3,), dtype=dtype)
    cdef real_t[:] ret = np.empty((nr_points,), dtype=dtype)

    # print("nr_points: ", nr_points)
    for i , pt in enumerate(points):
        x[0] = pt[0]
        x[1] = pt[1]
        x[2] = pt[2]
        ret[i] = _c_interp_trilin(s, m, crds, x, start_inds)
    return ret

cdef real_t _c_interp_trilin(real_t[:,:,:,::1] s, cnp.intp_t m, real_t[:] *crds,
                             real_t[:] x, int start_inds[3]):
    cdef int i, j, ind, ncells
    cdef int[3] ix
    cdef int[3] p  # increment, used for 2d fields
    cdef real_t[3] xd

    # cdef real_t[:] *crds = [crdz, crdy, crdx]
    cdef real_t c00, c10, c01, c11, c0, c1, c

    # find closest inds
    for i from 0 <= i < 3:
        # this 'if' is to support 2d fields... could probably be handled
        # more efficiently
        ncells = crds[i].shape[0]
        if ncells > 1:
            # find the closest ind
            # ind = _c_closest_ind[real_t](crds[i], x[i], &start_inds[i])
            # this implementation only works for monotonically increasing crds
            ind = _c_int_max(_c_int_min(start_inds[i], ncells - 2), 0)

            if crds[i][ind] <= x[i]:
                for j from ind <= j < ncells - 1:
                    if crds[i][j + 1] >= x[i]:
                        break
                ind = _c_int_min(j, ncells - 2)
            else:
                for j from ind - 1 >= j >= 0:
                    if crds[i][j] <= x[i]:
                        break
                ind = _c_int_max(j, 0)
            start_inds[i] = ind

            ix[i] = ind
            p[i] = 1
            xd[i] = (x[i] - crds[i][ind]) / (crds[i][ind + 1] - crds[i][ind])
        else:
            ind = 0
            ix[i] = ind
            p[i] = 0
            xd[i] = 1.0

    # INTERLACED ... z first
    c00 = s[ix[0], ix[1]       , ix[2]       , m] + xd[0] * (s[ix[0] + p[0], ix[1]       , ix[2]       , m] - s[ix[0], ix[1]       , ix[2]       , m])
    c10 = s[ix[0], ix[1] + p[1], ix[2]       , m] + xd[0] * (s[ix[0] + p[0], ix[1] + p[1], ix[2]       , m] - s[ix[0], ix[1] + p[1], ix[2]       , m])
    c01 = s[ix[0], ix[1]       , ix[2] + p[2], m] + xd[0] * (s[ix[0] + p[0], ix[1]       , ix[2] + p[2], m] - s[ix[0], ix[1]       , ix[2] + p[2], m])
    c11 = s[ix[0], ix[1] + p[1], ix[2] + p[2], m] + xd[0] * (s[ix[0] + p[0], ix[1] + p[1], ix[2] + p[2], m] - s[ix[0], ix[1] + p[1], ix[2] + p[2], m])
    c0 = c00 + xd[1] * (c10 - c00)
    c1 = c01 + xd[1] * (c11 - c01)
    c = c0 + xd[2] * (c1 - c0)

    # if c == 0.0:
    #     # print("??0:", c00, c10, c01, c11, '|', c0, c1, c, '|', xd[0], xd[1], xd[2])
    #     # print("!!0:", crds[0][ix[0]], crds[1][ix[1]], crds[2][ix[2]], '|',
    #     #       x[0], x[1], x[2], '|',
    #     #       crds[0][ix[0] + 1], crds[1][ix[1] + 1], crds[2][ix[2] + 1])
    #     print("**0", ix[0], ix[1], ix[2], '|', p[0], p[1], p[2])
    #     print("AA0:", s[ix[0]       , ix[1]       , ix[2]       , m],
    #                   s[ix[0] + p[0], ix[1]       , ix[2]       , m])
    #     print("BB0:", s[ix[0]       , ix[1] + p[1], ix[2]       , m],
    #                   s[ix[0] + p[0], ix[1] + p[1], ix[2]       , m])
    #     print("CC0:", s[ix[0]       , ix[1]       , ix[2] + p[2], m],
    #                   s[ix[0] + p[0], ix[1]       , ix[2] + p[2], m])
    #     print("DD0:", s[ix[0]       , ix[1] + p[1], ix[2] + p[2], m],
    #                   s[ix[0] + p[0], ix[1] + p[1], ix[2] + p[2], m])

    return c

def interp_nearest(fld, seeds, fill=None):
    """ Points can be list of 3-tuples or a SeedGen instance. If fld
    is a scalar field, the output array has shape (nr_points,) where nr_points
    is the number of seed points. If it's a vector, the output has shape
    (nr_points, nr_comps), where nr_comps is the number of components of the
    vector. The data type of the output is the same as the original field.
    The output is always an array, even if only one point is given.
    """
    if fld.iscentered("Cell"):
        crdz, crdy, crdx = fld.get_crds_cc()
    elif fld.iscentered("Node"):
        crdz, crdy, crdx = fld.get_crds_nc()
    else:
        raise RuntimeError("Dont touch me with that centering.")

    fld_dtype = fld.dtype
    crd_dtype = crdz.dtype
    cdef bint use_fill
    if fill is None:
        fill = np.array([0], dtype=fld_dtype)
        use_fill = False
    else:
        use_fill = True

    if fld.istype("Vector"):
        if not fld.layout == field.LAYOUT_INTERLACED:
            raise ValueError("Trilin interp only written for interlaced data.")
        nr_comps = fld.nr_comps
        nr_points = seeds.nr_points(center=fld.center)
        ret = np.empty((nr_points, nr_comps), dtype=fld_dtype)

        for j from 0 <= j < nr_comps:
            # print(ret.shape, nr_points, nr_comps)
            ret[:,j] = _py_interp_nearest(fld_dtype, crd_dtype, fld.data, j,
                                          crdz, crdy, crdx,
                                          seeds.iter_points(center=fld.center),
                                          nr_points, fill, use_fill)
        return ret

    elif fld.istype("Scalar"):
        dat = fld.data.reshape(fld.shape + [1])
        nr_points = seeds.nr_points(center=fld.center)
        ret = np.empty((nr_points,), dtype=fld_dtype)
        ret[:] = _py_interp_nearest(fld_dtype, crd_dtype, dat, 0,
                                    crdz, crdy, crdx,
                                    seeds.iter_points(center=fld.center),
                                    nr_points, fill, use_fill)
        return ret

    else:
        raise RuntimeError("That centering is not supported for interp_trilin")

def _py_interp_nearest(dtype_fld, dtype_crds, fld_t[:,:,:,::1] s, cnp.intp_t m,
                       real_t[:] crdz, crdy_in, crdx_in, points,
                       int nr_points, fld_t fill, bint use_fill):
    """ return the scalar value of 3d scalar array s trilinearly interpolated
    to the point x (in z, y, x order) """
    cdef unsigned int i
    # cdef real_t[:] crdz = crdz_in  # need 1 crd to establish fused type
    cdef real_t[:] crdy = crdy_in
    cdef real_t[:] crdx = crdx_in
    cdef real_t[:] *crds = [crdz, crdy, crdx]
    cdef int* start_inds = [0, 0, 0]

    cdef real_t[:] x = np.empty((3,), dtype=dtype_crds)
    cdef fld_t[:] ret = np.empty((nr_points,), dtype=dtype_fld)

    # print("nr_points: ", nr_points)
    for i, pt in enumerate(points):
        x[0] = pt[0]
        x[1] = pt[1]
        x[2] = pt[2]
        ret[i] = _c_interp_nearest(s, m, crds, x, start_inds, fill, use_fill)
    return ret

@cython.wraparound(True)
cdef fld_t _c_interp_nearest(fld_t[:,:,:,::1] s, cnp.intp_t m, real_t[:] *crds,
                             real_t[:] x, int start_inds[3],
                             fld_t fill, bint use_fill):
    cdef int i, j, ind, ncells
    cdef int[3] ix
    cdef int[3] p  # increment, used for 2d fields
    cdef real_t[3] xd

    if use_fill:
        if x[0] < crds[0][0] or x[0] > crds[0][-1] or \
           x[1] < crds[1][0] or x[1] > crds[1][-1] or \
           x[2] < crds[2][0] or x[2] > crds[2][-1]:
            return fill

    # find closest inds
    for i from 0 <= i < 3:
        # this 'if' is to support 2d fields... could probably be handled
        # more efficiently
        ncells = crds[i].shape[0]
        if ncells > 1:
            # find the closest ind
            # ind = _c_closest_ind[real_t](crds[i], x[i], &start_inds[i])
            # this implementation only works for monotonically increasing crds
            ind = _c_int_max(_c_int_min(start_inds[i], ncells - 2), 0)

            if crds[i][ind] <= x[i]:
                for j from ind <= j < ncells - 1:
                    if crds[i][j + 1] >= x[i]:
                        break
                ind = _c_int_min(j, ncells - 2)
            else:
                for j from ind - 1 >= j >= 0:
                    if crds[i][j] <= x[i]:
                        break
                ind = _c_int_max(j, 0)
            start_inds[i] = ind

            ix[i] = ind
            xd[i] = (x[i] - crds[i][ind]) / (crds[i][ind + 1] - crds[i][ind])
            p[i] = <int>(xd[i] + 0.5)
        else:
            ind = 0
            ix[i] = ind
            p[i] = 0

    return s[ix[0] + p[0], ix[1] + p[1], ix[2] + p[2], m]

##
## EOF
##
