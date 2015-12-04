#!/usr/bin/env python

from __future__ import division, print_function

import numpy as np
import viscid
from viscid.cython import streamline


def get_sep_pts_bitor(fld, min_depth=1, max_depth=10, multiple=True,
                      plot=False, sep_val=streamline.TOPOLOGY_MS_SEPARATOR,
                      mask_limit=0b1111, periodic="00", pt_bnds=()):
    """Find separator as intersection of all global topologies

    Neighbors are bitwise ORed until at least one value matches
    `sep_val` which is presumably (Close | Open N | Open S | SW).
    This happens between min_depth and max_depth times,
    where the resolution of each iteration is reduced by a factor
    of two, ie, worst case 2**(max_depth).

    Args:
        fld (Field): Topology (bitmask) as a field
        min_depth (int): Iterate at least this many times
        max_depth (int): Iterate at most this many times
        multiple (bool): passed to :py:func:`viscid.cluster`
        sep_val (int): Value of bitmask that indicates a separator
        plot (bool): Make a 2D plot of Fld and the sep candidates
        mask_limit (int): if > 0, then bitmask fld with mask_limit,
            i.e., fld = fld & mask_limit (bitwise and)
        periodic (sequence): indicate whether that direction is
            periodic, and if so, whether the coordinate arrays are
            overlapped or not. Values can be True, False, or '+'. '+'
            indicates that x[0] and x[-1] are not colocated, so assume
            they're dx apart where dx = x[-1] - x[-2].
        pt_bnd (sequence): Boundaries that come to a point, i.e., all
            values along that boundary are neighbors such as the poles
            of a sphere. Specified like "0-" for lower boundary of
            dimension 0 or "1+" for the upper boundary of dimension 1.

    Returns:
        ndarray: 2xN for N clusters of separator points in the same
        coordinates as `fld`
    """
    pd = [False if pi == "0" else bool(pi) for pi in periodic]
    fld = fld.slice_reduce(":")
    if mask_limit:
        fld = np.bitwise_and(fld, mask_limit)
    a = fld.data
    x, y = fld.get_crds()

    for i in range(max_depth):
        if pd[0]:
            a[(0, -1), :] |= a[(-1, 0), :]
        if pd[1]:
            a[:, (0, -1)] |= a[:, (-1, 0)]

        a = (a[ :-1,  :-1] | a[ :-1, 1:  ] |  # pylint: disable=bad-whitespace
             a[1:  ,  :-1] | a[1:  , 1:  ])   # pylint: disable=bad-whitespace
        x = 0.5 * (x[1:] + x[:-1])
        y = 0.5 * (y[1:] + y[:-1])

        # bitwise_or an entire bounary if all points are neighbors, like
        # at the poles of a sphere
        for bnd in pt_bnds:
            slc = [slice(None), slice(None)]
            slc[int(bnd[0])] = -1 if bnd[1] == "+" else 0
            a[slc] = np.bitwise_or.reduce(a[slc])

        indx, indy = np.where(a == sep_val)
        if i + 1 >= min_depth and len(indx):
            break

    pts = viscid.cluster(indx, indy, x, y, multiple=multiple,
                         periodic=periodic)

    if plot:
        from viscid.plot import mpl

        mpl.clf()
        mpl.subplot(121)
        mpl.plot(fld, title=True)

        mpl.subplot(122)
        or_fld = viscid.arrays2field(a, (x, y), name="OR")
        mpl.plot(or_fld, title=True)

        _x, _y = or_fld.get_crds()
        mpl.plt.plot(_x[indx], _y[indy], 'ko')
        # mpl.plt.show()

        mpl.plt.plot(pts[0], pts[1], 'y^')
        mpl.plt.show()

    return pts

def get_sep_pts_bitor_spherical(fld, theta_phi=False, overlap=False, cap=False,
                                **kwargs):
    """Wrap :py:func:`find_sep_points_cartesian` for spheres and caps

    This is kind of a janky interface since data about
    theta_phi / overlap / cap could exist in the field
    """
    pd = "1" if overlap else "+"
    if theta_phi:
        kwargs['periodic'] = (False, pd)
        # kwargs['pt_bnds'] = ("0-",) if cap else ("0-", "0+")
    else:
        kwargs['periodic'] = (pd, False)
        # kwargs['pt_bnds'] = ("1-",) if cap else ("1-", "1+")
    return get_sep_pts_bitor(fld, **kwargs)

def get_sep_pts_bisect(fld, seed):
    raise NotImplementedError()

##
## EOF
##
