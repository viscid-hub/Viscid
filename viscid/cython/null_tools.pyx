# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: emit_code_comments=False

from itertools import count

import numpy as np
import viscid

cimport cython
cimport numpy as cnp

ctypedef fused real_t:
    cnp.float32_t
    cnp.float64_t


def find_classified_nulls(fld, ibound=0.0, rtol=1e-5, atol=1e-8):
    """Find nulls, and classify them as in Cowley 1973

    Args:
        fld (VectorField): Magnetic field with nulls
        ibound (float): ignore points within ibound of origin
        rtol (float): used in np.isclose when classifying nulls
        atol (float): used in np.isclose when classifying nulls

    Returns:
        dict: {"O": [Oinds, Opts], "A": [...], "B": [...]}
            each of these are 3xN ndarrays of either ingegers of floats
    """
    null_inds, null_pts = find_nulls(fld, ibound=ibound)
    Oi, Ai, Bi = _classify_nulls(fld, null_inds, rtol=rtol, atol=atol)
    Oinds, Ainds, Binds = null_inds[:, Oi], null_inds[:, Ai], null_inds[:, Bi]
    Opts, Apts, Bpts = null_pts[:, Oi], null_pts[:, Ai], null_pts[:, Bi]
    return dict(O=[Oinds, Opts], A=[Ainds, Apts], B=[Binds, Bpts])

def find_nulls(fld, ibound=0.0):
    """Just find null points and closest indices in fld

    Args:
        fld (VectorField): Magnetic field with nulls
        ibound (float): ignore points within ibound of origin

    Returns:
        tuple: (null_inds, null_pts) both of which are 3xN ndarrays,
            the first is integers, the second floats
    """
    fld = fld.as_interlaced()
    candidate_inds = _reduce_candidates(fld, ibound=ibound)
    null_inds, null_pts = _find_null_cells_from_candidates(fld, candidate_inds)
    return null_inds, null_pts

def _classify_nulls(b, null_inds, rtol=1e-5, atol=1e-8):
    """rtol and atol used te determine what is close to 0"""
    # using terminology from Cowley 1973
    O_type = []
    A_type = []
    B_type = []

    ixs, iys, izs = null_inds
    for i, ix, iy, iz in zip(count(), ixs, iys, izs):
        e_vals, e_vecs = viscid.jacobian_eig_at_ind(b, ix, iy, iz)
        # categorize e-vals as 0, real, or complex
        real_evals, complex_evals, zero_evals = [], [], []
        real_evecs, complex_evecs, zero_evecs = [], [], []
        for j, e_val in enumerate(e_vals):
            if np.isclose(e_val, 0.0, rtol=rtol, atol=atol):
                zero_evals.append(0.0)
                zero_evecs.append(e_vecs[:, j])
            elif np.isclose(e_val.imag, 0.0, rtol=rtol, atol=atol):
                real_evals.append(e_val.real)
                real_evecs.append(e_vecs[:, j])
            else:
                complex_evals.append(e_val)
                complex_evecs.append(e_vecs[:, j])

        if len(zero_evals) == 1:
            O_type.append(i)
        elif len(real_evals) == 1 and len(zero_evals) == 0:
            assert np.isclose((complex_evals[0] + complex_evals[1]).imag, 0.0,
                              rtol=rtol, atol=atol)
            if real_evals[0] > 0.0:
                A_type.append(i)
            else:
                B_type.append(i)
        elif len(real_evals) == 3:
            positive = 0
            for e_val in real_evals:
                if e_val > 0.0:
                    positive += 1
            if positive == 1:
                A_type.append(i)
            else:
                B_type.append(i)
        else:
            print("unknown type of null:", e_vals)

    return (np.array(O_type, dtype='i'),
            np.array(A_type, dtype='i'),
            np.array(B_type, dtype='i'))

@cython.wraparound(True)
def _reduce_candidates(fld, ibound=0.0):
    # these crds are same centering as fld
    dat = fld.as_interlaced().data
    x, y, z = fld.get_crds("xyz")
    candidate_inds = _py_reduce_candidates(dat, x, y, z, ibound)
    return np.array(candidate_inds)

def _find_null_cells_from_candidates(fld, inds):
    assert fld.nr_sdims == 3
    assert fld.nr_comps == 3
    # izs = np.array(izs, dtype='int32', copy=False)
    # iys = np.array(iys, dtype='int32', copy=False)
    # ixs = np.array(ixs, dtype='int32', copy=False)
    dat = fld.as_interlaced().data
    null_inds = np.array(_py_find_null_cells_from_candidates(dat, inds))
    x, y, z = fld.get_crds()
    null_pts = np.zeros_like(null_inds).astype(dat.dtype)
    null_pts[0, :] = x[null_inds[0, :]]
    null_pts[1, :] = y[null_inds[1, :]]
    null_pts[2, :] = z[null_inds[2, :]]
    return null_inds, null_pts

def _py_reduce_candidates(real_t[:, :, :, ::1] b,
                          real_t[:] x, real_t[:] y, real_t[:] z, ibound):
    """ find candidate cells where the Bx, By, and Bz all
    go through 0, there is no guarentee these curves intersect
    at a null point
    """
    cdef cnp.int8_t[:, :, :, ::1] sign
    cdef cnp.int8_t[:, :, ::1] mask
    cdef cnp.int8_t s000, s001, s010, s011, s100, s101, s110, s111
    cdef cnp.int8_t different[3]
    cdef int i, j, k, d
    cdef int nz, ny, nx, nc

    cdef real_t ibound_sq = ibound**2
    cdef real_t r_sq

    nx, ny, nz, nc = b.shape[:4]

    sign = np.empty((nx, ny, nz, nc), dtype=np.int8)
    ixs, iys, izs = [], [], []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                sign[i, j, k, 0] = b[i, j, k, 0] > 0.0
                sign[i, j, k, 1] = b[i, j, k, 1] > 0.0
                sign[i, j, k, 2] = b[i, j, k, 2] > 0.0

    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                r_sq = x[i]**2 + y[j]**2 + z[k]**2
                if r_sq < ibound_sq:
                    continue

                for d in range(nc):
                    s000 = sign[i    , j    , k    , d]
                    s001 = sign[i    , j    , k + 1, d]
                    s010 = sign[i    , j + 1, k    , d]
                    s011 = sign[i    , j + 1, k + 1, d]
                    s100 = sign[i + 1, j    , k    , d]
                    s101 = sign[i + 1, j    , k + 1, d]
                    s110 = sign[i + 1, j + 1, k    , d]
                    s111 = sign[i + 1, j + 1, k + 1, d]
                    # this is the other logic that works
                    # ne.evaluate("~((s000 & s001 & s010 & s011 & "
                    #             "   s100 & s101 & s110 & s111) | "
                    #             " ~(s000 | s001 | s010 | s011 | "
                    #             "   s100 | s101 | s110 | s111))")
                    if ((s000 ^ s001) | (s001 ^ s010) |
                        (s010 ^ s011) | (s011 ^ s100) |
                        (s100 ^ s101) | (s101 ^ s110) |
                        (s110 ^ s111)):
                        different[d] = 1  # True
                    else:
                        different[d] = 0  # False
                if different[0] and different[1] and different[2]:
                    ixs.append(i)
                    iys.append(j)
                    izs.append(k)
    return ixs, iys, izs

def _py_find_null_cells_from_candidates(real_t[:, :, :, ::1] f,
                                        long[:, ::1] inds):
    """ a vector field and candidiate cell indices, find which cells actually
    contain null points """
    cdef real_t a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3
    cdef real_t rt1, rt2, ff3
    cdef int ic1, ic2, ic3
    cdef int iz, iy, ix, i, ic, icandidate
    cdef int all_roots, positive_roots, have_null

    # setup x,y,z offsets for all 4 coefficients for all 6 faces
    #Face:           XY    YZ    ZX
    cdef int* oxa = [0, 0, 0, 1, 0, 0]
    cdef int* oya = [0, 0, 0, 0, 0, 1]
    cdef int* oza = [0, 1, 0, 0, 0, 0]

    cdef int* oxb = [1, 1, 0, 1, 0, 0]
    cdef int* oyb = [0, 0, 1, 1, 0, 1]
    cdef int* ozb = [0, 1, 0, 0, 1, 1]

    cdef int* oxc = [0, 0, 0, 1, 1, 1]
    cdef int* oyc = [1, 1, 0, 0, 0, 1]
    cdef int* ozc = [0, 1, 1, 1, 0, 0]

    cdef int* oxd = [1, 1, 0, 1, 1, 1]
    cdef int* oyd = [0, 0, 1, 1, 0, 1]
    cdef int* ozd = [0, 1, 1, 1, 1, 1]
    cdef int[:, ::1] permuted_comps = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]],
                                               dtype="int32")
    # cdef int[:, ::1] permuted_comps = np.array([[0, 1, 2]], dtype="int32")

    null_ixs, null_iys, null_izs = [], [], []

    for icandidate in range(inds.shape[1]): # iz, iy, ix in zip(izs, iys, ixs):
        ix = inds[0, icandidate]
        iy = inds[1, icandidate]
        iz = inds[2, icandidate]

        have_null = 0

        # this is to check different fields on the faces since sometimes a
        # field is 0 on a whole face, and it wouldn't be picked up
        for ic in range(permuted_comps.shape[0]):
            ic1 = permuted_comps[ic, 0]
            ic2 = permuted_comps[ic, 1]
            ic3 = permuted_comps[ic, 2]
            positive_roots = 0
            all_roots = 0

            # loop over each cube face
            for i in range(6):
                a1 = f[ix + oxa[i], iy + oya[i], iz + oza[i], ic1]
                b1 = f[ix + oxb[i], iy + oyb[i], iz + ozb[i], ic1] - a1
                c1 = f[ix + oxc[i], iy + oyc[i], iz + ozc[i], ic1] - a1
                d1 = f[ix + oxd[i], iy + oyd[i], iz + ozd[i], ic1] - c1 - b1 - a1

                a2 = f[ix + oxa[i], iy + oya[i], iz + oza[i], ic2]
                b2 = f[ix + oxb[i], iy + oyb[i], iz + ozb[i], ic2] - a2
                c2 = f[ix + oxc[i], iy + oyc[i], iz + ozc[i], ic2] - a2
                d2 = f[ix + oxd[i], iy + oyd[i], iz + ozd[i], ic2] - c2 - b2 - a2

                a3 = f[ix + oxa[i], iy + oya[i], iz + oza[i], ic3]
                b3 = f[ix + oxb[i], iy + oyb[i], iz + ozb[i], ic3] - a3
                c3 = f[ix + oxc[i], iy + oyc[i], iz + ozc[i], ic3] - a3
                d3 = f[ix + oxd[i], iy + oyd[i], iz + ozd[i], ic3] - c3 - b3 - a3

                roots1, roots2 = find_roots_face(a1, b1, c1, d1, a2, b2, c2, d2)
                for rt1, rt2 in zip(roots1, roots2):
                    all_roots += 1
                    ff3 = a3 + b3 * rt1 + c3 * rt2 + d3 * rt1 * rt2
                    # print("!!! face root:", rt1, rt2, f3)
                    if ff3 > 0.0:
                        positive_roots += 1
                    elif ff3 == 0.0 and i % 2 == 0: # to only check 3 faces
                        # print("!!! null on face")
                        have_null = 1

            # if all_roots % 2 != 0:
            #     # print("!!!! raging bollox")
            #     pass

            # if even number of total roots, and odd number of positive roots
            # OR, if there is a null on a face (have_null == True)
            if ((positive_roots % 2 == 1) and all_roots % 2 == 0) or have_null:
                have_null = 1
                break

        if have_null:
            null_ixs.append(ix)
            null_iys.append(iy)
            null_izs.append(iz)
        else:
            pass

    return null_ixs, null_iys, null_izs

cdef inline find_roots_face(real_t a1, real_t b1, real_t c1, real_t d1,
                            real_t a2, real_t b2, real_t c2, real_t d2):
    """ This function just finds roots of a quadratic equation
    that solves the intersection of f1 = f2 = 0.
    f1(x, y) = a1 + b1 * x + c1 * y + d1 * x * y
    f2(x, y) = a1 + b1 * x + c1 * y + d1 * x * y
    where {x, y} are in the range [0.0, 1.0]

    Returns ndarray(x), ndarray(y) such that
    f1(xi, yi) = f2(xi, yi) = 0 and
    0 <= xi <= 1.0 && 0 < yi < 1.0

    Note: the way this prunes edges is different for x and y,
    this is to keep from double counting edges of a cell
    when the faces are visited xy yz zx
    """
    # F(x) = a * x**2 + b * x + c = 0 has the same roots xi
    # as f1(x, ?) = f2(x, ?) = 0, so solve that, then
    # substitute back into f1(xi, yi) = 0 to find yi
    cdef real_t a, b, c
    a = b1 * d2 - d1 * b2
    b = (a1 * d2 - d1 * a2) + (b1 * c2 - c1 * b2)
    c = a1 * c2 - c1 * a2

    if a == 0.0:
        if b != 0.0:
            roots_x = np.array([-c / b])
            # print("::", roots_x)
        else:
            roots_x = np.array([])
            # print("::", roots_x)
    else:
        desc = b**2 - (4.0 * a * c)
        if desc > 0.0:
            # il y a deux solutions
            rootx1 = (-b + np.sqrt(desc)) / (2.0 * a)
            rootx2 = (-b - np.sqrt(desc)) / (2.0 * a)
            roots_x = np.array([rootx1, rootx2])
            # print("::", roots_x)
        elif desc == 0.0:
            # il y a seulment une solution
            roots_x = np.array([-b / (2.0 * a)])
            # print("::", roots_x)
        else:
            roots_x = np.array([])
            # print("::", roots_x)

    roots_y = - (a1 + b1 * roots_x) / (c1 + d1 * roots_x)

    # remove roots that are outside the box
    keep = ((roots_x >= 0.0) & (roots_x <= 1.0) &
            (roots_y > 0.0) & (roots_y < 1.0))
    roots_x = roots_x[keep]
    roots_y = roots_y[keep]

    return roots_x, roots_y

##
## EOF
##
