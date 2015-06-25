# cython: boundscheck=False, wraparound=False, cdivision=True, profile=False

from __future__ import print_function

import numpy as np

###########
# cimports
cimport cython
cimport numpy as cnp

from libc.math cimport fabs

from cyfield cimport real_t


cdef inline int isclose(real_t a, real_t b):
    # tolerances don't need to be that exact
    cdef real_t atol = 1e-6
    cdef real_t rtol = 1e-5
    return fabs(a - b) <= (atol + rtol * fabs(b))

cdef inline int less_close(real_t a, real_t b):
    """Returns True if a and b are close, or if a < b"""
    return a <= b or isclose(a, b)

cdef inline int greater_close(real_t a, real_t b):
    """Returns True if a and b are close, or if a >= b"""
    return a >= b or isclose(a, b)


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
          to the ``right''.

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

##
## EOF
##
