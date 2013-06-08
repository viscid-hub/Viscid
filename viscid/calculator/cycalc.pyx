# cython: boundscheck=True, wraparound=True, profile=False
#
# Note: a _c_FUNCTION can only be called from another cdef-ed function, or
# a def-ed _py_FUNCTION function because of the use of the fused real_t
# to template both float32 and float64 versions

from __future__ import print_function
import logging

import numpy as np
from cython.view cimport array as cvarray
# from cython.parallel import prange

from .. import field
from .. import coordinate
from . import seed

###########
# cimports
cimport cython
cimport numpy as cnp

from libc.math cimport sqrt

from cycalc_util cimport *
from cycalc cimport *

cdef extern from "math.h":
    bint isnan(double x)

#####################
# now the good stuff

def scalar3d_shape(fld):
    return list(fld.shape) + [1] * (3 - len(fld.shape))

def vector3d_shape(fld):
    return list(fld.shape) + [1] * (3 - len(fld.shape)) + [-1]


def magnitude(fld):
    if not fld.layout == field.LAYOUT_INTERLACED:
        raise ValueError("I am only written for interlaced data.")
    nptype = fld.data.dtype.name
    vect = fld.data.reshape(vector3d_shape(fld))
    mag = np.empty(scalar3d_shape(fld), dtype=nptype)
    _py_magnitude3d(vect, mag)
    mag = mag.reshape(fld.shape)
    return field.wrap_field("Scalar", fld.name + " magnitude", fld.crds, mag,
                            center=fld.center, time=fld.time,
                            forget_source=True)

def _py_magnitude3d(real_t[:,:,:,:] vect, real_t[:,:,:] mag):
    return _c_magnitude3d(vect, mag)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _c_magnitude3d(real_t[:,:,:,:] vect, real_t[:,:,:] mag):
    cdef unsigned int nz = vect.shape[0]
    cdef unsigned int ny = vect.shape[1]
    cdef unsigned int nx = vect.shape[2]
    cdef unsigned int nc = vect.shape[3]
    cdef unsigned int i, j, k, c
    cdef real_t val

    for k from 0 <= k < nz:
        for j from 0 <= j < ny:
            for i from 0 <= i < nx:
                val = 0.0
                for c from 0 <= c < nc:
                    val += vect[k, j, i, c]**2
                mag[k,j,i] = sqrt(val)
    return None

@cython.wraparound(True)
def div(fld):
    if not fld.layout == field.LAYOUT_INTERLACED:
        raise ValueError("Div is only written for interlaced data.")
    if fld.dim != 3:
        raise ValueError("Div is only written in 3D.")

    nptype = fld.data.dtype.name
    vect = fld.data

    if fld.center == "Cell":
        crdz, crdy, crdx = fld.crds.get_crd(center="Cell")
        divcenter = "Cell"
        divcrds = coordinate.RectilinearCrds(fld.crds.get_clist(np.s_[1:-1]))
        dest_shape = [n - 2 for n in fld.crds.shape_cc]
        div_arr = np.empty(dest_shape, dtype=nptype)
    elif fld.center == "Node":
        crdz, crdy, crdx = fld.crds.get_crd()
        divcenter = "Node"
        divcrds = coordinate.RectilinearCrds(fld.crds.get_clist(np.s_[1:-1]))
        dest_shape = [n - 2 for n in fld.crds.shape_nc]
        div_arr = np.empty(dest_shape, dtype=nptype)
    else:
        raise NotImplementedError("Can only do cell and node centered divs")

    if crdx.dtype != nptype or crdy.dtype != nptype or crdz.dtype != nptype:
        raise TypeError("Coords must be same dtype as vector data")

    _py_div3d(vect, crdx, crdy, crdz, div_arr)

    return field.wrap_field("Scalar", fld.name + " div", divcrds, div_arr,
                            center=divcenter, time=fld.time,
                            forget_source=True)

def _py_div3d(real_t[:,:,:,:] vect, real_t[:] crdx, real_t[:] crdy,
               real_t[:] crdz, real_t[:,:,:] div_arr):
    return _c_div3d(vect, crdx, crdy, crdz, div_arr)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _c_div3d(real_t[:,:,:,:] vect, real_t[:] crdx, real_t[:] crdy,
              real_t[:] crdz, real_t[:,:,:] div_arr):
    cdef unsigned int nz = div_arr.shape[0]
    cdef unsigned int ny = div_arr.shape[1]
    cdef unsigned int nx = div_arr.shape[2]
    cdef unsigned int i, j, k
    cdef real_t val

    for k from 0 <= k < nz:
        for j from 0 <= j < ny:
            for i from 0 <= i < nx:
                # Note, the centering for the crds isnt correct here
                div_arr[k, j, i] = (vect[k, j, i + 2, 0] - vect[k, j, i, 0]) / \
                                                     (crdx[i + 2] - crdx[i]) + \
                                   (vect[k, j + 2, i, 1] - vect[k, j, i, 1]) / \
                                                     (crdy[j + 2] - crdy[j]) + \
                                   (vect[k + 2, j, i, 2] - vect[k, j, i, 2]) / \
                                                     (crdz[k + 2] - crdz[k])
    return None

def closest_ind(coord_array, value):
    """ returns the integer such that:
    if crd[1] > crd[0], then crd[i] < point <= crd[i+1]
    if crd[1] < crd[0], then crd[i] > point >= crd[i+1]
    ie. you always get the smaller index of the two straddling crds
    """
    return _py_closest_ind(coord_array, value)

def _py_closest_ind(real_t[:] crd, real_t point):
    """ returns the integer such that:
    if crd[1] > crd[0], then crd[i] < point <= crd[i+1]
    if crd[1] < crd[0], then crd[i] > point >= crd[i+1]
    ie. you always get the smaller index of the two straddling crds
    """
    return _c_closest_ind(crd, point)

cdef int _c_closest_ind(real_t[:] crd, real_t point) except -1:
    cdef int i
    # TODO: maybe this could be faster if i have a guess where to
    # start looking
    cdef int startind = 1
    cdef unsigned int n = crd.shape[0]
    cdef int forward = n > 1 and crd[1] > crd[0]

    # search linearly... maybe branch prediction makes this better
    # than bisection for smallish arrays...
    for i from startind <= i < n:
        if forward and crd[i] >= point:
            return i - 1
        if not forward and crd[i] <= point:
            return i - 1
    if n >= 2:
        return crd.shape[0] - 2  # if we've gone too far, pick the last index
    else:  # crd.shape[0] <= 1
        return startind - 1

def trilin_interp(fld, seeds):
    """ Points can be list of 3-tuples or a SeedGen instance. If fld
    is a scalar field, the output array has shape (npts,) where npts
    is the number of seed points. If it's a vector, the output has shape
    (npts, ncomps), where ncomps is the number of components of the vector.
    The data type of the output is the same as the original field.
    The output is always an array, even if only one point is given.
    """
    if fld.center == "Cell":
        crdz, crdy, crdx = fld.crds.get_crd(center="Cell")
    elif fld.center == "Node":
        crdz, crdy, crdx = fld.crds.get_crd()
    else:
        raise RuntimeError("Dont touch me with that centering.")

    dtype = fld.dtype

    if fld.TYPE == "Vector":
        views = fld.component_views()
        ncomp = fld.ncomp
        npts = seeds.n_points(center=fld.center)
        ret = np.empty((npts, ncomp), dtype=dtype)

        for j from 0 <= j < ncomp:
            # print(ret.shape, npts, ncomp)
            ret[:,j] = _py_trilin_interp(dtype, views[j], crdz, crdy, crdx,
                                seeds.iter_points(center=fld.center),
                                npts)
        return ret

    elif fld.TYPE == "Scalar":
        view = fld.data
        npts = seeds.n_points(center=fld.center)
        ret = np.empty((npts,), dtype=dtype)
        ret[:] = _py_trilin_interp(dtype, view, crdz, crdy, crdx,
                                 seeds.iter_points(center=fld.center),
                                 npts)
        return ret
    else:
        raise RuntimeError("That centering is not supported for trilin_interp")

def _py_trilin_interp(dtype, real_t[:,:,:] s, real_t[:] crdz, real_t[:] crdy,
                      real_t[:] crdx, points, int n_points):
    """ return the scalar value of 3d scalar array s trilinearly interpolated
    to the point x (in z, y, x order) """
    cdef unsigned int i
    cdef real_t[:] x = np.empty((3,), dtype=dtype)
    cdef real_t[:] ret = np.empty((n_points,), dtype=dtype)
    # print("n_points: ", n_points)
    #for i, pt in enumerate(points):
    for i , pt in enumerate(points):
        x[0] = pt[0]
        x[1] = pt[1]
        x[2] = pt[2]
        ret[i] = _c_trilin_interp(s, crdz, crdy, crdx, x)
    return ret

cdef real_t _c_trilin_interp(real_t[:,:,:] s, real_t[:] crdz,
                             real_t[:] crdy, real_t[:] crdx, real_t[:] x):
    cdef int i, ind
    cdef int[3] ix
    cdef int[3] p  # increment, used for 2d fields
    cdef real_t[3] xd
    cdef real_t[3] xdm  # will be 1 - xd
    
    crds = [crdz, crdy, crdx]
    cdef real_t[:] crd
    cdef real_t c00, c10, c01, c11, c0, c1, c

    zp = 1 if crdz.shape[0] > 1 else 0
    yp = 1 if crdy.shape[0] > 1 else 0
    xp = 1 if crdx.shape[0] > 1 else 0

    # TODO: improve performance by making this not a list
    # crds = [crdz, crdy, crdx]
    for i from 0 <= i < 3:
        crd = crds[i]
        ind = _c_closest_ind[real_t](crd, x[i])
        ix[i] = ind
        # this bit to support 2d fields could probably be handled
        # more efficiently
        if crd.shape[0] > 1:
            p[i] = 1
            xd[i] = (x[i] - crd[ind]) / (crd[ind + 1] - crd[ind])
        else:
            p[i] = 0
            xd[i] = 1.0
        xdm[i] = 1.0 - xd[i]

    # this algorithm is shamelessly taken from the trilinear interpolation
    # wikipedia article
    c00 = s[ix[0]       , ix[1]       , ix[2]       ] * xdm[0] + \
          s[ix[0] + p[0], ix[1]       , ix[2]       ] * xd[0]
    c10 = s[ix[0]       , ix[1] + p[1], ix[2]       ] * xdm[0] + \
          s[ix[0] + p[0], ix[1] + p[1], ix[2]       ] * xd[0]
    c01 = s[ix[0]       , ix[1]       , ix[2] + p[2]] * xdm[0] + \
          s[ix[0] + p[0], ix[1]       , ix[2] + p[2]] * xd[0]
    c11 = s[ix[0]       , ix[1] + p[1], ix[2] + p[2]] * xdm[0] + \
          s[ix[0] + p[0], ix[1] + p[1], ix[2] + p[2]] * xd[0]

    c0 = c00 * xdm[1] + c10 * xd[1]
    c1 = c01 * xdm[1] + c11 * xd[1]
    c = c0 * xdm[2] + c1 * xd[2]
    # if isnan(c):
    #     logging.warning("NAN: {0}".format([s.shape[0], s.shape[1], s.shape[2], 
    #                                        ix[0], ix[1], ix[2], p[0], p[1], p[2],
    #                                        s[ix[0]       , ix[1]       , ix[2]       ],
    #                                        s[ix[0] + p[0], ix[1]       , ix[2]       ],
    #                                        s[ix[0]       , ix[1] + p[1], ix[2]       ],
    #                                        s[ix[0] + p[0], ix[1] + p[1], ix[2]       ],
    #                                        s[ix[0]       , ix[1]       , ix[2] + p[2]],
    #                                        s[ix[0] + p[0], ix[1]       , ix[2] + p[2]],
    #                                        s[ix[0]       , ix[1] + p[1], ix[2] + p[2]],
    #                                        s[ix[0] + p[0], ix[1] + p[1], ix[2] + p[2]]
    #                                       ]))
    return c

##
## EOF
##
