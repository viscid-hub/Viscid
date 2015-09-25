# cython: boundscheck=False, wraparound=False, cdivision=True, profile=False

from __future__ import print_function

import numpy as np

from viscid.seed import to_seeds

from cython.operator cimport dereference as deref
from libc.math cimport floor, fabs

from viscid.cython.cyamr cimport FusedAMRField, make_cyamrfield, activate_patch
from viscid.cython.cyfield cimport real_t
from viscid.cython.cyfield cimport CyField, FusedField, make_cyfield
from viscid.cython.misc_inlines cimport int_min, int_max

def interp_trilin(vfield, seeds, force_amr_version=False):
    """Interpolate a field to points described by seeds

    Note:
        Nearest neighbor is used between the last value and
        `vfield.crds.xh`. This is done to keep from extrapolating and
        introducing new maxima.

    Parameters:
        vfield (viscid.field.Field): Some Vector or Scalar field
        seeds (viscid.claculator.seed): locations for the interpolation
        force_amr_version (bool): used for benchmarking amr overhead

    Returns:
        numpy.ndarray of interpolated values. Shaped (seed.nr_points,)
        or (seed.nr_points, vfield.nr_comps) if vfield is a Scalar or
        Vector field.
    """
    seeds = to_seeds(seeds)
    cdef int nr_points = seeds.get_nr_points(center=vfield.center)
    cdef int nr_comps = vfield.nr_comps
    if nr_comps == 0:
        scalar = True
        nr_comps = 1
    else:
        scalar = False

    if vfield.nr_patches > 1 or force_amr_version:
        amrfld = make_cyamrfield(vfield)
        result = np.empty((nr_points, nr_comps), dtype=amrfld.crd_dtype)
        _py_interp_trilin_amr(amrfld, seeds.iter_points(center=vfield.center), result)
    else:
        # about 12% faster than the AMR version on fields w/ 1 patch
        fld = make_cyfield(vfield)
        result = np.empty((nr_points, nr_comps), dtype=fld.crd_dtype)
        _py_interp_trilin(fld, seeds.iter_points(center=vfield.center), result)

    if scalar:
        result = result[:, 0]
    return result

def interp_nearest(vfield, seeds, force_amr_version=False):
    """Interpolate a field to points described by seeds

    Parameters:
        vfield (viscid.field.Field): Some Vector or Scalar field
        seeds (viscid.claculator.seed): locations for the interpolation
        force_amr_version (bool): used for benchmarking amr overhead

    Returns:
        numpy.ndarray of interpolated values. Shaped (seed.nr_points,)
        or (seed.nr_points, vfield.nr_comps) if vfield is a Scalar or
        Vector field.
    """
    seeds = to_seeds(seeds)
    cdef int nr_points = seeds.nr_points(center=vfield.center)
    cdef int nr_comps = vfield.nr_comps
    if nr_comps == 0:
        scalar = True
        nr_comps = 1
    else:
        scalar = False

    if vfield.nr_patches > 1 or force_amr_version:
        amrfld = make_cyamrfield(vfield)
        result = np.empty((nr_points, nr_comps), dtype=amrfld.crd_dtype)
        _py_interp_nearest_amr(amrfld, seeds.iter_points(center=vfield.center), result)
    else:
        # about 6% faster than the AMR version on fields w/ 1 patch
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
            assert len(pt) == 3
            x[0] = pt[0]
            x[1] = pt[1]
            x[2] = pt[2]
            result[i, m] = _c_interp_trilin(fld, m, x)
        i += 1

def _py_interp_trilin_amr(FusedAMRField amrfld, points, real_t[:, ::1] result):
    cdef int i, m
    cdef int nr_comps = result.shape[1]
    cdef real_t x[3]

    i = 0
    for pt in points:
        for m in range(nr_comps):
            assert len(pt) == 3, "Seeds must have 3 spatial dimensions"
            x[0] = pt[0]
            x[1] = pt[1]
            x[2] = pt[2]
            activate_patch[FusedAMRField, real_t](amrfld, x)
            result[i, m] = _c_interp_trilin(amrfld.active_patch, m, x)
        i += 1

cdef real_t _c_interp_trilin(FusedField fld, int m, real_t x[3]):
    cdef int d, ind
    cdef int ix[3]
    cdef int p[3]  # increment, used for 2d fields
    cdef real_t xd[3]

    cdef real_t c00, c10, c01, c11, c0, c1, c

    # find closest inds
    for d in range(3):
        if fld.n[d] == 1 or x[d] <= fld.xl[d]:
            ind = 0
            ix[d] = ind
            p[d] = 0
            xd[d] = 0.0
        elif x[d] >= fld.xh[d]:
            # switch to nearest neighbor for points beyond last value
            ind = fld.n[d] - 1
            p[d] = 0
            xd[d] = 0.0
        else:
            ind = closest_preceeding_ind(fld, d, x[d])
            p[d] = 1
            xd[d] = ((x[d] - fld.crds[d, ind]) /
                     (fld.crds[d, ind + 1] - fld.crds[d, ind]))
        ix[d] = ind

    # INTERLACED ... x first
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
            assert len(pt) == 3, "Seeds must have 3 spatial dimensions"
            x[0] = pt[0]
            x[1] = pt[1]
            x[2] = pt[2]
            result[i, m] = _c_interp_nearest(fld, m, x)
        i += 1

def _py_interp_nearest_amr(FusedAMRField amrfld, points, real_t[:, ::1] result):
    cdef int i, m
    cdef int nr_comps = result.shape[1]
    cdef real_t[3] x

    i = 0
    for pt in points:
        for m in range(nr_comps):
            x[0] = pt[0]
            x[1] = pt[1]
            x[2] = pt[2]
            activate_patch[FusedAMRField, real_t](amrfld, x)
            result[i, m] = _c_interp_nearest(amrfld.active_patch, m, x)
        i += 1

cdef real_t _c_interp_nearest(FusedField fld, int m, real_t x[3]):
    cdef int ind[3]
    cdef int d

    for d in range(3):
        ind[d] = closest_ind(fld, d, x[d])
    return fld.data[ind[0], ind[1], ind[2], m]


cdef inline int closest_preceeding_ind(FusedField fld, int d, real_t value):
    """Index of the element closest (and to the left) of x = value

    Parameters:
        fld (FusedField): field
        d (int): dimension [0..2]
        value (real_t): get index closest to fld.data['d=value']

    Returns:
        int: closest indext preceeding value in the coordinate array d
    """
    cdef real_t frac
    cdef int i, ind, found_ind
    cdef int n = fld.n[d]
    cdef int startind = fld.cached_ind[d]

    if n == 1:
        ind = 0
    elif fld.uniform_crds:
        frac = (value - fld.xl[d]) / (fld.L[d])
        i = <int> floor((fld.nm1[d]) * frac)
        ind = int_min(int_max(i, 0), fld.nm1[d])
    else:
        found_ind = 0
        if fld.crds[d, startind] <= value:
            i = startind
            for i in range(startind, n - 1):
                if fld.crds[d, i + 1] > value:
                    found_ind = 1
                    break
            if not found_ind:
                i = n - 1
        else:
            i = startind - 1
            for i in range(startind - 1, -1, -1):
                if fld.crds[d, i] <= value:
                    found_ind = 1
                    break
            if not found_ind:
                i = 0
        ind = i

    fld.cached_ind[d] = ind
    return ind

cdef inline int closest_ind(FusedField fld, int d, real_t value):
    cdef double d1, d2
    cdef int preceeding_ind = closest_preceeding_ind(fld, d, value)
    if preceeding_ind == fld.n[d] - 1:
        return fld.n[d] - 1
    else:
        d1 = fabs(fld.crds[d, preceeding_ind] - value)
        d2 = fabs(fld.crds[d, preceeding_ind + 1] - value)
        if d1 <= d2:
            return preceeding_ind
        else:
            return preceeding_ind + 1

##
## EOF
##
