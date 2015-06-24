# cython: boundscheck=False, wraparound=False, cdivision=True, profile=False

from __future__ import print_function

import numpy as np

from cython.operator cimport dereference as deref
from libc.math cimport floor, fabs

from cyfield cimport real_t, fld_t
from cyfield cimport CyField, FusedField, make_cyfield
from cycalc cimport int_min, int_max

def interp_trilin(vfield, seeds):
    """Interpolate a field to points described by seeds

    Parameters:
        vfield (viscid.field.Field): Some Vector or Scalar field
        seeds (viscid.claculator.seed): locations for the interpolation

    Returns:
        numpy.ndarray of interpolated values. Shaped (seed.nr_points,)
        or (seed.nr_points, vfield.nr_comps) if vfield is a Scalar or
        Vector field.
    """
    cdef int nr_points = seeds.nr_points(center=vfield.center)
    cdef int nr_comps = vfield.nr_comps
    if nr_comps == 0:
        scalar = True
        nr_comps = 1
    else:
        scalar = False

    fld = make_cyfield(vfield)
    result = np.empty((nr_points, nr_comps), dtype=fld.crd_dtype)

    _py_interp_trilin(fld, seeds.iter_points(center=vfield.center), result)

    if scalar:
        result = result[:, 0]
    return result

def interp_nearest(vfield, seeds):
    """Interpolate a field to points described by seeds

    Parameters:
        vfield (viscid.field.Field): Some Vector or Scalar field
        seeds (viscid.claculator.seed): locations for the interpolation

    Returns:
        numpy.ndarray of interpolated values. Shaped (seed.nr_points,)
        or (seed.nr_points, vfield.nr_comps) if vfield is a Scalar or
        Vector field.
    """
    cdef int nr_points = seeds.nr_points(center=vfield.center)
    cdef int nr_comps = vfield.nr_comps
    if nr_comps == 0:
        scalar = True
        nr_comps = 1
    else:
        scalar = False

    fld = make_cyfield(vfield)
    result = np.empty((nr_points, nr_comps), dtype=fld.crd_dtype)

    _py_interp_nearest(fld, seeds.iter_points(center=vfield.center), result)

    if scalar:
        result = result[:, 0]
    return result

def _py_interp_trilin(FusedField fld, points, real_t[:, ::1] result):
    cdef int i, m
    cdef int nr_comps = result.shape[1]
    cdef real_t x[3]

    i = 0
    for pt in points:
        for m in range(nr_comps):
            x[0] = pt[0]
            x[1] = pt[1]
            x[2] = pt[2]
            result[i, m] = _c_interp_trilin(fld, m, x)
        i += 1

cdef real_t _c_interp_trilin(FusedField fld, int m, real_t x[3]):
    cdef int d, ind
    cdef int[3] ix
    cdef int[3] p  # increment, used for 2d fields
    cdef real_t[3] xd

    cdef real_t c00, c10, c01, c11, c0, c1, c

    # find closest inds
    for d in range(3):
        nr_cells = fld.nr_cells[d]
        if nr_cells > 1:
            ind = closest_preceeding_ind(fld, d, x[d])
            ix[d] = ind
            p[d] = 1
            # if ind + 1 >= fld.nr_cells[d]:
            #     raise ValueError("d: {0}, ind: {1}".format(d, ind))
            xd[d] = ((x[d] - fld.crds[d, ind]) /
                     (fld.crds[d, ind + 1] - fld.crds[d, ind]))
            # xd[d] *= 0.9  # break the interpolation, for testing
        else:
            ind = 0
            ix[d] = ind
            p[d] = 0
            xd[d] = 1.0
            # xd[d] *= 0.9  # break the interpolation, for testing

    # INTERLACED ... z first
    c00 = (fld.data[ix[0], ix[1]       , ix[2]       , m] +
           xd[0] * (fld.data[ix[0] + p[0], ix[1]       , ix[2]       , m] -
                    fld.data[ix[0]       , ix[1]       , ix[2]       , m]))
    c10 = (fld.data[ix[0], ix[1] + p[1], ix[2]       , m] +
           xd[0] * (fld.data[ix[0] + p[0], ix[1] + p[1], ix[2]       , m] -
                    fld.data[ix[0]       , ix[1] + p[1], ix[2]       , m]))
    c01 = (fld.data[ix[0], ix[1]       , ix[2] + p[2], m] +
           xd[0] * (fld.data[ix[0] + p[0], ix[1]       , ix[2] + p[2], m] -
                    fld.data[ix[0]       , ix[1]       , ix[2] + p[2], m]))
    c11 = (fld.data[ix[0], ix[1] + p[1], ix[2] + p[2], m] +
           xd[0] * (fld.data[ix[0] + p[0], ix[1] + p[1], ix[2] + p[2], m] -
                    fld.data[ix[0]       , ix[1] + p[1], ix[2] + p[2], m]))
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

def _py_interp_nearest(FusedField fld, points, real_t[:, ::1] result):
    cdef int i, m
    cdef int nr_comps = result.shape[1]
    cdef real_t[3] x

    i = 0
    for pt in points:
        for m in range(nr_comps):
            x[0] = pt[0]
            x[1] = pt[1]
            x[2] = pt[2]
            result[i, m] = _c_interp_nearest(fld, m, x)
        i += 1

cdef real_t _c_interp_nearest(FusedField fld, int m, real_t x[3]):
    cdef int ind[3]
    cdef int d

    for d in range(3):
        ind[d] = closest_ind(fld, d, x[d])
    return fld.data[ind[0], ind[1], ind[2], m]


cdef int closest_preceeding_ind(FusedField fld, int d, real_t value):
    """Index of the element closest (and to the left) of x = value

    This function returns the closest index preceeding value such that
    fld.data[..., i, ...] and fld.data[..., i + 1, ...] exist.

    Parameters:
        fld (FusedField): field
        d (int): dimension [0..2]
        value (real_t): get index closest to fld.data['d=value']

    Returns:
        int: closest indext preceeding value in the coordinate array d
    """
    cdef real_t frac
    cdef int i, ind, done
    cdef int n = fld.n[d]
    cdef int startind = fld.cached_ind[d]

    if n == 1:
        ind = 0
    elif fld.uniform_crds:
        frac = (value - fld.xl[d]) / (fld.L[d])
        i = <int> floor((fld.nm1[d]) * frac)
        ind = int_min(int_max(i, 0), fld.nm2[d])
    else:
        done = 0

        # if startind >= fld.nr_cells[d]:
        #     raise ValueError("d: {0}, startind: {1}".format(d, startind))
        if fld.crds[d, startind] <= value:
            i = startind
            for i in range(startind, n - 1):
                # if i >= fld.nr_cells[d]:
                #     raise ValueError("d: {0}, i: {1}".format(d, i))
                if fld.crds[d, i + 1] > value:
                    break
            ind = int_min(i, n - 2)
        else:
            i = startind - 1
            for i in range(startind - 1, -1, -1):
                # if i >= fld.nr_cells[d]:
                #     raise ValueError("d: {0}, i: {1}".format(d, i))
                if fld.crds[d, i] <= value:
                    break
            ind = int_max(i, 0)

    fld.cached_ind[d] = ind
    return ind

cdef int closest_ind(FusedField fld, int d, real_t value):
    cdef double d1, d2
    cdef int preceeding_ind = closest_preceeding_ind(fld, d, value)
    d1 = fabs(fld.crds[d, preceeding_ind] - value)
    d2 = fabs(fld.crds[d, preceeding_ind + 1] - value)
    if d1 <= d2:
        return preceeding_ind
    else:
        return preceeding_ind + 1

##
## EOF
##
