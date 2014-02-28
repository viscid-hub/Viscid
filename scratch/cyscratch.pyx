import numpy as np
from libc.math cimport sqrt
cimport numpy as cnp
cimport cython

from viscid import field

ctypedef fused real_t:
    cnp.float32_t
    cnp.float64_t

ctypedef fused fld_t:
    cnp.float32_t
    cnp.float64_t
    cnp.int32_t
    cnp.int64_t

def scalar3d_shape(fld):
    return list(fld.shape) + [1] * (3 - len(fld.shape)) + [1]

def vector3d_shape(fld):
    return list(fld.shape) + [1] * (3 - len(fld.shape)) + [-1]

def magnitude(fld):
    if not fld.layout == field.LAYOUT_INTERLACED:
        raise ValueError("I am only written for interlaced data.")
    nptype = fld.data.dtype.name
    # vect = fld.data.reshape(vector3d_shape(fld))
    # mag = np.empty(scalar3d_shape(fld), dtype=nptype)
    vect = fld.data
    mag = np.empty_like(vect)
    _py_magnitude3d(vect, mag)
    mag = mag.reshape(fld.shape)
    return field.wrap_field("Vector", fld.name + " magnitude", fld.crds, mag,
                            center=fld.center, time=fld.time,
                            forget_source=True)

def _py_magnitude3d(real_t[:,:,:,:] vect, real_t[:,:,:,:] mag):
    return _c_magnitude3d(vect, mag)

cdef _c_magnitude3d(real_t[:,:,:,:] vect, real_t[:,:,:,:] mag):
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
                mag[k,j,i,0] = sqrt(val)
    return None
