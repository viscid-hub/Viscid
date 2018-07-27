#!/usr/bin/env python
"""Minimum Variance Analysis and boundary normal crd tools"""

from __future__ import print_function, division

import numpy as np
import viscid


__all__ = ["minvar", "minvar_series", "minvar_around",
           "find_minvar_lmn", "find_minvar_lmn_around"]


def minvar(B, p1, p2, n=40):
    """Find minimum variance eigenvectors of `B` between `p1` and `p2`

    Do minimum variance analysis (MVA) of vector field `B` between
    two points, i.e., `p1` and `p2` are like two spacecraft locations
    on either side of the boundary for the MVA.

    Args:
        B (:py:class:`viscid.field.VectorField`): Vector field for MVA
        p1 (sequence): xyz point on one side of the boundary
        p2 (sequence): xyz point on the other side of the boundary
        n (int): number of points to sample B for the MVA

    Returns:
        (evals, evecs)
        All are ordered toward increasing eigenvalue magnitude. `evecs`
        is a 2d array where the vectors are columns (2nd axis), i.e.,
        the min variance eigenvector is evecs[0, :]
    """
    p1 = np.array(p1).reshape((3,))
    p2 = np.array(p2).reshape((3,))
    line = viscid.Line(p1, p2, n)
    b = viscid.interp_trilin(B, line)

    return minvar_series(b.data)

def minvar_series(arr, warn_finite=True):
    """Find minimum variance eigenvectors of vector series

    Args:
        arr (ndarray): vector time series with shape (nsamples, 3)

    Returns:
        (evals, evecs)
        All are ordered toward increasing eigenvalue magnitude. `evecs`
        is a 2d array where the vectors are columns (2nd axis), i.e.,
        the min variance eigenvector is evecs[0, :]
    """
    mask = np.all(np.isfinite(arr), axis=1, keepdims=False)
    if np.sum(mask) < 3:
        if warn_finite:
            viscid.logger.warning("Minvar says you need > 3 finite samples")
        return [np.full([3], np.nan, dtype=arr.dtype),
                np.full([3, 3], np.nan, dtype=arr.dtype)]

    arr = arr[mask]
    m = np.zeros((3, 3))
    for i in range(3):
        for j in range(i+1):
            bibj = np.mean(arr[:, i] * arr[:, j])
            m[i, j] = bibj - (np.mean(arr[:, i]) * np.mean(arr[:, j]))
            m[j, i] = bibj - (np.mean(arr[:, i]) * np.mean(arr[:, j]))

    evals, evecs = np.linalg.eig(m)
    eval_mags = np.abs(evals)
    sorted_inds = np.argsort(eval_mags)
    evec_min = evecs[:, sorted_inds[0]].flatten()
    evec_int = evecs[:, sorted_inds[1]].flatten()
    evec_max = evecs[:, sorted_inds[2]].flatten()

    eval_min = eval_mags[sorted_inds[0]]
    eval_int = eval_mags[sorted_inds[1]]
    warn_thresh = 0.05
    if (eval_int - eval_min) / eval_min < warn_thresh:
        viscid.logger.warning("Minvar says minimum and intermediate eigenvalues "
                              "are too close together: {0:g} - {1:g} < {2:g}%"
                              "".format(eval_int, eval_min, warn_thresh))
    return evals[sorted_inds], np.array([evec_min, evec_int, evec_max]).T

def minvar_around(B, p0, l=1.0, path_dir=(1, 0, 0), n=40):
    """Find minimum variance eigenvectors of `B` around point `p0`

    Do minimum variance analysis (MVA) of vector field `B` a distance
    `l`/2 around point `p0` in the direction of `path_dir`, i.e.,
    `path_dir` is like the spacecraft travel direction.

    Args:
        B (:py:class:`viscid.field.VectorField`): Vector field for MVA
        p0 (sequence): xyz center point, within -l/2 and +l/2 of the
            boundary in the path_dir directon
        l (float): distance on either side of center point
        path_dir (tuple): spacecraft travel direction
        n (int): number of points to sample B for the MVA

    Returns:
        (evals, evecs)
        All are ordered toward increasing eigenvalue magnitude. `evecs`
        is a 2d array where the vectors are columns (2nd axis), i.e.,
        the min variance eigenvector is evecs[0, :]

    See Also:
        :py:func:`minvar`
    """
    p0 = np.array(p0).reshape((3,))
    path_dir = np.array(path_dir).reshape((3,))
    path_dir = path_dir / np.linalg.norm(path_dir)
    p1 = p0 - 0.5 * l * path_dir
    p2 = p0 + 0.5 * l * path_dir
    return minvar(B, p1, p2, n=n)

def _minvar_lmn_directions(evec_min, evec_max, l_basis=(0, 0, 1)):
    """Find rotation matrix for going to boundary normal (lmn) crds

    Args:
        evec_min (sequence): minvar minimum eigenvector
        evec_max (sequence): minvar maximum eigenvector
        l_basis (sequence): vector used to define the l-direction,
            i.e., the l direction will be `l_basis` projected
            perpendicular to the boundary normal directon. If None,
            then `l_basis` will be set to `evec_max`.

    Returns:
        ndarray: 3x3 array of l,m,n basis column (2nd axis) vectors

    Note:
        If l_basis is parallel to the normal direction, evec_max will
        be used
    """
    evec_min = np.array(evec_min).reshape((3,))
    evec_min = evec_min / np.linalg.norm(evec_min)
    n_dir = evec_min

    try:
        l_basis = np.array(l_basis).reshape((3,))
        if np.allclose(n_dir, l_basis / np.linalg.norm(l_basis)):
            viscid.logger.warning("LMN says l_basis is parallel to normal, using "
                                  "MVAs max eigenvector direction")
            raise ValueError
    except ValueError:
        l_basis = np.array(evec_max).reshape((3,))

    l_dir = l_basis - np.dot(l_basis, n_dir) * n_dir
    l_dir = l_dir / np.linalg.norm(l_dir)

    m_dir = np.cross(n_dir, l_dir)
    return np.array([l_dir, m_dir, n_dir]).T

def find_minvar_lmn(B, p1, p2, n=40, l_basis=(0, 0, 1)):
    """Find rotation matrix for going to boundary normal (lmn) crds

    Args:
        B (:py:class:`viscid.field.VectorField`): Vector field for MVA
        p1 (sequence): xyz point on one side of the boundary
        p2 (sequence): xyz point on the other side of the boundary
        n (int): number of points to sample B for the MVA
        l_basis (sequence): vector used to define the l-direction,
            i.e., the l direction will be `l_basis` projected
            perpendicular to the boundary normal directon. If None,
            then `l_basis` will be set to `evec_max`.

    Returns:
        ndarray: 3x3 array of l,m,n basis column (2nd axis) vectors
    """
    _, evecs = minvar(B, p1, p2, n=n)
    return _minvar_lmn_directions(evecs[:, 0], evecs[:, 2], l_basis=l_basis)

def find_minvar_lmn_around(B, p0, l=1.0, path_dir=(1, 0, 0), n=40, l_basis=(0, 0, 1)):
    """Find rotation matrix for going to boundary normal (lmn) crds

    Finds boundary normal from minimum variance analysis (MVA) of
    vector field `B` a distance `l`/2 around point `p0` in the
    direction of `path_dir`, i.e., `path_dir` is like the spacecraft
    travel direction.

    Args:
        B (:py:class:`viscid.field.VectorField`): Vector field for MVA
        p0 (sequence): xyz center point, within -l/2 and +l/2 of the
            boundary in the path_dir directon
        l (float): distance on either side of center point
        path_dir (tuple): spacecraft travel direction
        n (int): number of points to sample B for the MVA
        l_basis (sequence): vector used to define the l-direction,
            i.e., the l direction will be `l_basis` projected
            perpendicular to the boundary normal directon. If None,
            then `l_basis` will be set to `evec_max`.

    Returns:
        ndarray: 3x3 array of l,m,n basis column (2nd axis) vectors

    See Also:
        :py:func:`find_nlm`
    """
    _, evecs = minvar_around(B, p0, l=l, path_dir=path_dir, n=n)
    return _minvar_lmn_directions(evecs[:, 0], evecs[:, 2], l_basis=l_basis)

def _main():
    x = np.linspace(-1, 1, 128)
    y = z = np.linspace(-0.25, 0.25, 8)
    B = viscid.zeros((x, y, z), nr_comps=3, layout='interlaced', name="B")
    X, Y, Z = B.get_crds("xyz", shaped=True)  # pylint: disable=unused-variable
    xl, yl, zl = B.xl  # pylint: disable=unused-variable
    xh, yh, zh = B.xh  # pylint: disable=unused-variable
    xm, ym, zm = 0.5 * (B.xl + B.xh)  # pylint: disable=unused-variable

    B['x'] = 0.0  # np.sin(1.0 * np.pi * X / (xh - xl) + 0.5 * np.pi)
    B['y'] = np.sin(1.0 * np.pi * X / (xh - xl) + 0.5 * np.pi)
    B['z'] = np.sin(1.0 * np.pi * X / (xh - xl) - 1.0 * np.pi)
    B += 0.33 * np.random.random_sample(B.shape)

    # R = viscid.a2b_rotm((1, 0, 0), (1, 0, 1))
    # B[...] = np.einsum("ij,lmnj->lmni", R, B)

    lmn = find_minvar_lmn(B, (xl, ym, zm), (xh, ym, zm), l_basis=None)
    # lmn = find_minvar_lmn(B, (xl, ym, zm), (xh, ym, zm), l_basis=(0, 0, 1))
    print("LMN matrix:\n", lmn, sep='')

    ##########
    from viscid.plot import vpyplot as vlt
    from matplotlib import pyplot as plt
    p0 = np.array((xm, ym, zm)).reshape((3,))
    pl = p0 + 0.25 * lmn[:, 0]
    pm = p0 + 0.25 * lmn[:, 1]
    pn = p0 + 0.25 * lmn[:, 2]

    print("p0", p0)
    print("pl", pl)
    print("pm", pm)
    print("pn", pn)

    vlt.subplot(211)
    vlt.plot2d_quiver(B['z=0j'])
    plt.plot([p0[0], pl[0]], [p0[1], pl[1]], color='r', ls='-')
    plt.plot([p0[0], pm[0]], [p0[1], pm[1]], color='c', ls='-')
    plt.plot([p0[0], pn[0]], [p0[1], pn[1]], color='b', ls='-')
    plt.ylabel("Y")

    vlt.subplot(212)
    vlt.plot2d_quiver(B['y=0j'])
    plt.plot([p0[0], pl[0]], [p0[2], pl[2]], color='r', ls='-')
    plt.plot([p0[0], pm[0]], [p0[2], pm[2]], color='c', ls='-')
    plt.plot([p0[0], pn[0]], [p0[2], pn[2]], color='b', ls='-')
    plt.xlabel("X")
    plt.ylabel("Z")

    vlt.show()
    ##########

    return 0

if __name__ == "__main__":
    import sys  # pylint: disable=wrong-import-position,wrong-import-order
    sys.exit(_main())

##
## EOF
##
