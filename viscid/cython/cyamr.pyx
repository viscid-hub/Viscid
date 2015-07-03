# cython: boundscheck=True, wraparound=False, cdivision=True, profile=False

from __future__ import print_function

import numpy as np

###########
# cimports
cimport cython

from libc.math cimport fabs
cimport numpy as cnp

from viscid.cython.cyamr cimport *
# from viscid.cython.cyamr cimport _contains_block
from viscid.cython.cyfield cimport MAX_FLOAT, real_t
from viscid.cython.cyfield cimport CyField, FusedField, make_cyfield
from viscid.cython.misc_inlines cimport real_min, real_max
from viscid.cython.misc_inlines cimport isclose, less_close
from viscid.cython.misc_inlines cimport less_not_close, greater_not_close

def discover_neighbors(skel):
    """Find which patches touch

    Args:
        skel (:py:class:`viscid.amr_grid.AMRSkeleton`): A skeleton with
            valid xm and L.

    Returns:
        (nr_neighbors, neighbors, neighbor_mask)

        * `nr_neighbors` (ndarray with shape (npatches,)): how many
          neighbors a patch has
        * `neighbors` (int ndarray with shape (npatches, 48)): gives
          index of all the neighbors of a given patch. Empty values
          are filled with -1
        * `neighbor_mask` (int ndarray with shape (npatches, 48)): Bit
          mask of the relationship to the neighboring patch. Touching
          in x is 1 or 2, touching in y is 3 or 4, touching in z is 5
          or 6. The lesser value is used if the neighboring patch is
          to the "right".

    Note:
        The rules for `neighbor_mask` are:

        * `mask >> 6` will be a bitmask of 3 bits that says if patches
          touch in a given direction. So if they touch in x,
          `mask >> 6 == 0b100` and if they touch in x and y,
          `mask >> 6 == 0b110`
        * TODO: document the 6 least significant bits
    """
    # 24 possible face neighbors, 16 edges, 8 corners; assumes grid is
    # properly nested
    cdef int max_neighbors = 24 + 16 + 8
    cdef int npatches = len(skel.patches)
    nr_neighbors = np.zeros(npatches, dtype='i')
    neighbors = -1 * np.ones((npatches, max_neighbors), dtype='i')
    neighbor_mask = np.zeros((npatches, max_neighbors), dtype='i')

    _py_discover_neighbors(skel, skel.xm, skel.L, nr_neighbors, neighbors,
                           neighbor_mask)
    return nr_neighbors, neighbors, neighbor_mask

def _py_discover_neighbors(skel, real_t[:, ::1] xm, real_t[:, ::1] L,
                           int[:] nr_neighbors, int[:, ::1] neighbors,
                           int[:, ::1] neighbor_mask):
    cdef int i, j, k, possible_neighbor, is_neighbor
    cdef int f_flag, i_flag, fmask, imask
    cdef real_t r[3]
    cdef real_t r_abs[3]
    cdef real_t d[3]

    for i in range(nr_neighbors.shape[0]):
        patch = skel.patches[i]
        for j in range(i):
            other = skel.patches[j]
            possible_neighbor = 1
            fmask = 0
            imask = 0

            # check if any distances are > max distance
            for k in range(3):
                r[k] = xm[i, k] - xm[j, k]
                r_abs[k] = fabs(r[k])
                d[k] = 0.5 * (L[i, k] + L[j, k])
                if not less_close(r_abs[k], d[k]):
                    possible_neighbor = 0
                    break

            # check if the two patches actually touch
            if possible_neighbor:
                for k in range(3):
                    if isclose(r_abs[k] - d[k], 0.0):
                        if r[k] >= 0:
                            f_flag, i_flag = 1, 2
                        else:
                            f_flag, i_flag = 2, 1
                        fmask = fmask | 1 << (6 + k) | f_flag << (2 * k)
                        imask = imask | 1 << (6 + k) | i_flag << (2 * k)

            # if we have a non-zero mask the patches touch
            if fmask:
                neighbors[i, nr_neighbors[i]] = j
                neighbors[j, nr_neighbors[j]] = i
                neighbor_mask[i, nr_neighbors[i]] = fmask
                neighbor_mask[j, nr_neighbors[j]] = imask
                nr_neighbors[i] += 1
                nr_neighbors[j] += 1
    return nr_neighbors, neighbors, neighbor_mask


cdef CyAMRField make_cyamrfield(vfield):
    fld_dtype = np.dtype(vfield.dtype)

    if fld_dtype == np.dtype('i4'):
        amrfld = _init_cyamrfield(AMRField_I4_Crd_F8(), vfield, 'f8')
    elif fld_dtype == np.dtype('i8'):
        amrfld = _init_cyamrfield(AMRField_I8_Crd_F8(), vfield, 'f8')
    if fld_dtype == np.dtype('f4'):
        amrfld = _init_cyamrfield(AMRField_F4_Crd_F4(), vfield, 'f4')
    elif fld_dtype == np.dtype('f8'):
        amrfld = _init_cyamrfield(AMRField_F8_Crd_F8(), vfield, 'f8')
    else:
        raise RuntimeError("Bad field dtype for cython code {0}"
                           "".format(fld_dtype))
    return amrfld

cdef FusedAMRField _init_cyamrfield(FusedAMRField amrfld, vfield, crd_dtype):
    cdef int i, j
    cdef int nr_blocks = vfield.nr_blocks
    cdef CyField _block

    amrfld.crd_dtype = crd_dtype
    amrfld.nr_blocks = nr_blocks

    try:
        amrfld.nr_neighbors = vfield.skeleton.nr_neighbors
        amrfld.neighbors = vfield.skeleton.neighbors
        amrfld.neighbor_mask = vfield.skeleton.neighbor_mask

        amrfld.xl = vfield.skeleton.xl.astype(crd_dtype, copy=False)
        amrfld.xm = vfield.skeleton.xm.astype(crd_dtype, copy=False)
        amrfld.xh = vfield.skeleton.xh.astype(crd_dtype, copy=False)

        for i in range(3):
            # print("> setting global[", i, "] xl =", vfield.skeleton.global_xl[i], vfield.skeleton.global_xh[i])
            amrfld.global_xl[i] = vfield.skeleton.global_xl[i]  # .astype(crd_dtype, copy=False)
            amrfld.global_xh[i] = vfield.skeleton.global_xh[i]  # .astype(crd_dtype, copy=False)

    except AttributeError:
        for i in range(3):
            amrfld.global_xl[i] = vfield.crds.xl_nc[i]
            amrfld.global_xh[i] = vfield.crds.xh_nc[i]

    # give max some breathing room?
    amrfld.min_dx = MAX_FLOAT
    amrfld.blocks = []
    for i in range(nr_blocks):
        _block = make_cyfield(vfield.blocks[i])
        if _block.min_dx < amrfld.min_dx:
            amrfld.min_dx = _block.min_dx
        amrfld.blocks.append(_block)

    amrfld.active_block_index = 0
    amrfld.active_block = amrfld.blocks[0]

    return amrfld

cdef inline int _contains_block(FusedAMRField amrfld, int iblock, real_t x[3]):
    cdef int i
    for i in range(3):
        if (less_not_close(x[i], amrfld.xl[iblock, i]) or
            greater_not_close(x[i], amrfld.xh[iblock, i])):
            return 0
    return 1

cdef CyField activate_block(FusedAMRField amrfld, real_t x[3]):
    cdef int active_idx, j, k, ineighbor, closest
    cdef real_t rsq, closest_rsq

    if amrfld.nr_blocks == 1:
        return amrfld.active_block
    else:
        active_idx = amrfld.active_block_index
        if _contains_block[FusedAMRField, real_t](amrfld, active_idx, x):
            return amrfld.active_block

        # search neighbors of the current active block
        for ineighbor in range(amrfld.nr_neighbors[active_idx]):
            j = amrfld.neighbors[active_idx, ineighbor]
            if _contains_block[FusedAMRField, real_t](amrfld, j, x):
                amrfld.active_block = amrfld.blocks[j]
                amrfld.active_block_index = j
                return amrfld.active_block

        # search all blocks
        for j in range(amrfld.nr_blocks):
            if _contains_block[FusedAMRField, real_t](amrfld, j, x):
                amrfld.active_block = amrfld.blocks[j]
                amrfld.active_block_index = j
                return amrfld.active_block

        # TODO: find closest block?
        closest = 0
        closest_rsq = MAX_FLOAT
        for j in range(amrfld.nr_blocks):
            rsq = 0.0
            for k in range(3):
                rsq += (x[k] - amrfld.xm[j, k])**2
            if rsq < closest_rsq:
                closest_rsq = rsq
                closest = j
        amrfld.active_block = amrfld.blocks[closest]
        amrfld.active_block_index = closest
        return amrfld.active_block

##
## EOF
##
