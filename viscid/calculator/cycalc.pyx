#cython: boundscheck=True, wraparound=True
from __future__ import print_function

import numpy as np
from .. import field
from .. import coordinate

cimport cython
cimport numpy as np
from libc.math cimport sqrt
from cycalc_util cimport *

#from cython.parallel import prange

import time

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
    _magnitude3(vect, mag)
    mag = mag.reshape(fld.shape)
    return field.wrap_field("Scalar", fld.name + " magnitude", fld.crds, mag,
                            center=fld.center, time=fld.time,
                            forget_source=True)

# can't cdef this because real_t is fused and numpy arrays are ducktyped
def _magnitude3(real_t[:,:,:,:] vect, real_t[:,:,:] mag):
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

def div(fld):
    if not fld.layout == field.LAYOUT_INTERLACED:
        raise ValueError("I am only written for interlaced data.")
    if fld.dim != 3:
        raise ValueError("I am only written for 3D divergence.")

    nptype = fld.data.dtype.name
    vect = fld.data

    if fld.center == "Cell":
        crdz, crdy, crdx = fld.crds.get_cc()
        divcenter = "Cell"
        divcrds = coordinate.RectilinearCrds(fld.crds.get_clist(np.s_[1:-1]))
        div_arr = np.empty([n - 2 for n in fld.crds.shape_cc], dtype=nptype)
    elif fld.center == "Node":
        crdz, crdy, crdx = fld.crds.get_nc()
        divcenter = "Node"
        divcrds = coordinate.RectilinearCrds(fld.crds.get_clist(np.s_[1:-1]))
        div_arr = np.empty([n - 2 for n in fld.crds.shape_nc], dtype=nptype)
    else:
        raise NotImplementedError("Can only do cell and node centered divs")

    if crdx.dtype != nptype or crdy.dtype != nptype or crdz.dtype != nptype:
        raise TypeError("Coords must be same dtype as vector data")

    _div3(vect, crdx, crdy, crdz, div_arr)

    return field.wrap_field("Scalar", fld.name + " div", divcrds, div_arr,
                            center=divcenter, time=fld.time,
                            forget_source=True)

def _div3(real_t[:,:,:,:] vect, real_t[:] crdx, real_t[:] crdy,
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

# cdef np.ndarray[real_t, ndim=4] calc_div1(np.ndarray[real_t, ndim=4] arr):# except 0:
#     cdef unsigned int nz = arr.shape[0]
#     cdef unsigned int ny = arr.shape[1]
#     cdef unsigned int nx = arr.shape[2]
#     cdef unsigned int nc = arr.shape[3]  # number of components
#     cdef unsigned int i, j, k, c
#     cdef double val

#     cdef np.ndarray[real_t, ndim=4] div = np.empty([nz, ny, nx, 1], dtype=arr.dtype)
#     # print(nz, ny, nx, nc)
#     for i from 0 <= i < nz:
#         for j from 0 <= j < ny:
#             for k from 0 <= k < nx:
#                 val = 0.0
#                 for c from 0 <= c < nc:
#                     val += arr[i,j,k,c]**2
#                 div[i,j,k,0] = sqrt(val)
#     return div

# def print_arr(np.ndarray[np.float64_t, ndim=3] arr):
#     c_print_arr(<double*>arr.data, arr.size)
#     print(arr.flags)

# cdef void c_print_arr(double *arr, int N):
#     for i in range(N):
#         print("{0} ".format(arr[i]))
