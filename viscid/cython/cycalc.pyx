# cython: boundscheck=False, wraparound=False, cdivision=True, profile=False
# cython: emit_code_comments=False

from __future__ import print_function

import numpy as np

from viscid.seed import to_seeds

from cython.operator cimport dereference as deref
from libc.math cimport floor, fabs

from viscid.cython.cyamr cimport CyAMRField
from viscid.cython.cyamr cimport FusedAMRField, make_cyamrfield, activate_patch
from viscid.cython.cyfield cimport real_t
from viscid.cython.cyfield cimport CyField, FusedField, make_cyfield
from viscid.cython.misc_inlines cimport int_min, int_max


def interp(vfield, seeds, kind="linear", wrap=True, method=None):
    """Interpolate a field to points described by seeds

    Note:
        Nearest neighbor is always used between the last value and
        `vfield.crds.xh`. This is done to keep from extrapolating and
        introducing new maxima. As such, AMR grids may be more
        step-like at patch boundaries.

    Parameters:
        vfield (viscid.field.Field): Some Vector or Scalar field. If
            this field is not 3D, then vfield.atleast_3d() is called
        seeds (viscid.claculator.seed): locations for the interpolation
        kind (str): either 'linear' or 'nearest'
        wrap (bool): if true, then call seeds.wrap on the result
        method (str): alias for kind, because why not

    Returns:
        numpy.ndarray of interpolated values. Shaped (seed.nr_points,)
        or (seed.nr_points, vfield.nr_comps) if vfield is a Scalar or
        Vector field.
    """
    kind = kind.strip().lower()
    if method:
        kind = method.strip().lower()

    if vfield.nr_sdims != 3:
        vfield = vfield.atleast_3d()

    seeds = to_seeds(seeds)

    seed_center = seeds.center if hasattr(seeds, 'center') else vfield.center
    if seed_center.lower() in ('face', 'edge'):
        seed_center = 'cell'

    cdef int nr_points = seeds.get_nr_points(center=seed_center)
    cdef int nr_comps = vfield.nr_comps
    if nr_comps == 0:
        scalar = True
        nr_comps = 1
    else:
        scalar = False

    amrfld = make_cyamrfield(vfield)
    # NOTE: this potentially unsafe cast is a hack because Cython's Fused
    # types no longer support returning a specialized fused type? It should
    # be ok (ie, doesn't segfault / return gibberish) as long as
    # make_cyamrfield always returns a CyAMRField subclass, which it should.
    result = np.empty((nr_points, nr_comps), dtype=(<CyAMRField>amrfld).crd_dtype)

    seed_iter = seeds.iter_points(center=seed_center)
    if kind == "nearest":
        _py_interp_nearest(amrfld, seed_iter, result)
    elif kind == "linear" or kind == "trilinear" or kind == "trilin":
        _py_interp_trilin(amrfld, seed_iter, result)
    else:
        raise ValueError("kind '{0}' not understood. Use linear or nearest"
                         "".format(kind))

    if scalar:
        result = result[:, 0]

    if wrap:
        if scalar:
            result = seeds.wrap_field(result, name=vfield.name)
        else:
            result = seeds.wrap_field(result, name=vfield.name,
                                      fldtype="vector", layout="interlaced")

    return result

def interp_trilin(vfield, seeds, wrap=True):
    """Interpolate a field to points described by seeds

    Note:
        Nearest neighbor is used between the last value and
        `vfield.crds.xh`. This is done to keep from extrapolating and
        introducing new maxima.

    Parameters:
        vfield (viscid.field.Field): Some Vector or Scalar field
        seeds (viscid.claculator.seed): locations for the interpolation
        wrap (bool): if true, then call seeds.wrap on the result

    Returns:
        numpy.ndarray of interpolated values. Shaped (seed.nr_points,)
        or (seed.nr_points, vfield.nr_comps) if vfield is a Scalar or
        Vector field.
    """
    return interp(vfield, seeds, wrap=wrap, kind="linear")

interp_linear = interp_trilin

def interp_nearest(vfield, seeds, wrap=True):
    """Interpolate a field to points described by seeds

    Parameters:
        vfield (viscid.field.Field): Some Vector or Scalar field
        seeds (viscid.claculator.seed): locations for the interpolation
        wrap (bool): if true, call seeds.wrap on the result

    Returns:
        numpy.ndarray of interpolated values. Shaped (seed.nr_points,)
        or (seed.nr_points, vfield.nr_comps) if vfield is a Scalar or
        Vector field.
    """
    return interp(vfield, seeds, wrap=wrap, kind="nearest")

def _py_interp_trilin(FusedAMRField amrfld, points, real_t[:, ::1] result):
    cdef int i, m
    cdef int nr_comps = result.shape[1]
    cdef real_t x[3]
    cdef int cached_idx3[3]

    cached_idx3[:] = [0, 0, 0]

    i = 0
    for pt in points:
        for m in range(nr_comps):
            assert len(pt) == 3, "Seeds must have 3 spatial dimensions"
            x[0] = pt[0]
            x[1] = pt[1]
            x[2] = pt[2]
            if amrfld.nr_patches > 1:
                activate_patch[FusedAMRField, real_t](amrfld, x)
            result[i, m] = _c_interp_trilin(amrfld.active_patch, m, x,
                                            cached_idx3)
        i += 1

cdef real_t _c_interp_trilin(FusedField fld, int m, real_t x[3],
                             int cached_idx3[3]) nogil:
    cdef int d, ind
    cdef int ix[3]
    cdef int p[3]  # increment, used for 2d fields
    cdef real_t xd[3]

    cdef real_t c00, c10, c01, c11, c0, c1, c

    # find closest inds
    for d in range(3):
        if fld.n[d] == 1 or x[d] <= fld.xl[m, d]:
            ind = 0
            ix[d] = ind
            p[d] = 0
            xd[d] = 0.0
        elif x[d] >= fld.xh[m, d]:
            # switch to nearest neighbor for points beyond last value
            ind = fld.n[d] - 1
            p[d] = 0
            xd[d] = 0.0
        else:
            ind = closest_preceeding_ind(fld, m, d, x[d], cached_idx3)
            p[d] = 1
            xd[d] = ((x[d] - fld.crds[m, d, ind]) /
                     (fld.crds[m, d, ind + 1] - fld.crds[m, d, ind]))
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

def _py_interp_nearest(FusedAMRField amrfld, points, real_t[:, ::1] result):
    cdef int i, m
    cdef int nr_comps = result.shape[1]
    cdef real_t[3] x
    cdef int cached_idx3[3]

    cached_idx3[:] = [0, 0, 0]

    i = 0
    for pt in points:
        for m in range(nr_comps):
            x[0] = pt[0]
            x[1] = pt[1]
            x[2] = pt[2]
            if amrfld.nr_patches > 1:
                activate_patch[FusedAMRField, real_t](amrfld, x)
            result[i, m] = _c_interp_nearest(amrfld.active_patch, m, x,
                                             cached_idx3)
        i += 1

cdef real_t _c_interp_nearest(FusedField fld, int m, real_t x[3],
                              int cached_idx3[3]) nogil:
    cdef int ind[3]
    cdef int d

    for d in range(3):
        ind[d] = closest_ind(fld, m, d, x[d], cached_idx3)
    return fld.data[ind[0], ind[1], ind[2], m]


cdef inline int closest_preceeding_ind(FusedField fld, int m, int d, real_t value,
                                       int cached_idx3[3]) nogil except -1:
    """Index of the element closest (and to the left) of x = value

    Note:
        The index is limited to at most `fld.n[d] - 2` such that the
        return value, `i`, AND `i + 1` are both valid indices of
        `fld.data` along axis `d`.

    Parameters:
        fld (FusedField): field
        d (int): dimension [0..2]
        value (real_t): get index closest to fld.data['d=value']

    Returns:
        int: closest index preceeding value in the coordinate array d
    """
    cdef real_t frac
    cdef int i, ind, found_ind
    cdef int n = fld.n[d]
    cdef int startidx = cached_idx3[d]  # using fld.cached_ind is not threadsafe

    if n == 1:
        ind = 0
    elif fld.uniform_crds:
        frac = (value - fld.xl[m, d]) / (fld.L[m, d])
        i = <int> floor((fld.nm1[d]) * frac)
        ind = int_min(int_max(i, 0), fld.nm2[d])
    else:
        found_ind = 0
        if fld.crds[m, d, startidx] <= value:
            i = startidx
            for i in range(startidx, n - 1):
                if fld.crds[m, d, i + 1] > value:
                    found_ind = 1
                    break
            if not found_ind:
                i = n - 1
        else:
            i = startidx - 1
            for i in range(startidx - 1, -1, -1):
                if fld.crds[m, d, i] <= value:
                    found_ind = 1
                    break
            if not found_ind:
                i = 0
        ind = i

    cached_idx3[d] = ind  # using fld.cached_ind is not threadsafe
    return ind

cdef inline int closest_ind(FusedField fld, int m, int d, real_t value,
                            int cached_idx3[3]) nogil except -1:
    cdef double d1, d2
    cdef int preceeding_ind = closest_preceeding_ind(fld, m, d, value, cached_idx3)
    if preceeding_ind == fld.n[d] - 1:
        return fld.n[d] - 1
    else:
        d1 = fabs(fld.crds[m, d, preceeding_ind] - value)
        d2 = fabs(fld.crds[m, d, preceeding_ind + 1] - value)
        if d1 <= d2:
            return preceeding_ind
        else:
            return preceeding_ind + 1

##
## EOF
##
