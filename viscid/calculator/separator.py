#!/usr/bin/env python

from __future__ import division, print_function
from itertools import count

import numpy as np
import viscid
from viscid.cython import streamline


UNEVEN_MASK = 0b1000
UNEVEN_HALF = 0.6


def topology_bitor_clusters(fld, min_depth=1, max_depth=10, multiple=True,
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
        ax0 = mpl.subplot(121)
        mpl.plot(fld, title=True)

        mpl.subplot(122, sharex=ax0, sharey=ax0)
        or_fld = viscid.arrays2field(a, (x, y), name="OR")
        mpl.plot(or_fld, title=True)

        _x, _y = or_fld.get_crds()
        mpl.plt.plot(_x[indx], _y[indy], 'ko')

        mpl.plt.plot(pts[0], pts[1], 'y^')
        if plot.strip().lower() == "show":
            mpl.plt.show()

    return pts

def get_sep_pts_bitor(fld, seed, trace_opts=None, make_3d=False, **kwargs):
    """Wrap :py:func:`find_sep_points_cartesian` for spheres and caps

    This is kind of a janky interface since data about
    theta_phi / overlap / cap could exist in the field
    """
    trace_opts.setdefault('ibound', 5.5)
    trace_opts.setdefault('output', viscid.OUTPUT_TOPOLOGY)
    topo = viscid.calc_streamlines(fld, seed, **trace_opts)[1]

    try:
        pt_bnds = seed.pt_bnds
    except AttributeError:
        pt_bnds = ()

    try:
        periodic = seed.periodic
    except AttributeError:
        periodic = "00"

    kwargs.setdefault('pt_bnds', pt_bnds)
    kwargs.setdefault('periodic', periodic)

    pts = topology_bitor_clusters(topo, **kwargs)
    if make_3d:
        pts = seed.uv_to_3d(pts)
    return pts

def perimeter_check_bitwise_or(arr):
    """Does perimeter of arr topologies contain a separator?

    Returns:
        bool
    """
    return bool(np.bitwise_or.reduce(arr) == streamline.TOPOLOGY_MS_SEPARATOR)

def get_sep_pts_bisect(fld, seed, trace_opts=None, min_depth=3, max_depth=7,
                       plot=False, perimeter_check=perimeter_check_bitwise_or,
                       make_3d=False, _base_quadrent="", _recurse0=True,
                       _uneven_mask=0):
    if len(_base_quadrent) == max_depth:
        return [_base_quadrent]  # causes pylint to complain
    if trace_opts is None:
        trace_opts = dict()

    nx, ny = seed.uv_shape
    (xlim, ylim) = seed.uv_extent

    if _recurse0 and plot:
        from viscid.plot import mpl
        mpl.clf()
        _, all_topo = viscid.calc_streamlines(fld, seed, max_length=300.0,
                                              ibound=6.0, topo_style="msphere",
                                              output=viscid.OUTPUT_TOPOLOGY,
                                              **trace_opts)
        mpl.plot(np.bitwise_and(all_topo, 15), show=False)

    # quadrents and lines are indexed as follows...
    # directions are counter clackwise around the quadrent with
    # lower index (which matters for lines which are shared among
    # more than one quadrent, aka, lines 1,2,6,7). Notice that even
    # numbered lines are horizontal, like the interstate system :)
    # -<--10-----<-8---
    # |       ^       ^
    # 11  2   9   3   7
    # \/      |       |
    # --<-2-----<-6----
    # |       ^       ^
    # 3   0   1   1   5
    # \/      |       |
    # ----0->-----4->--

    # find low(left), mid(center), and high(right) crds in x and y
    low_quad = "{0}{1:x}".format(_base_quadrent, 0 | _uneven_mask)
    high_quad = "{0}{1:x}".format(_base_quadrent, 3 | _uneven_mask)
    xl, xm, yl, ym = _quadrent_limits(low_quad, xlim, ylim)
    _, xh, _, yh = _quadrent_limits(high_quad, xlim, ylim)
    segsx, segsy = [None] * 12, [None] * 12
    topo = [None] * 12
    nxm, nym = nx //2, ny // 2

    # make all the line segments
    segsx[0], segsy[0] = np.linspace(xl, xm, nxm), np.linspace(yl, yl, nxm)
    segsx[1], segsy[1] = np.linspace(xm, xm, nym), np.linspace(yl, ym, nym)
    segsx[2], segsy[2] = np.linspace(xm, xl, nxm), np.linspace(ym, ym, nxm)
    segsx[3], segsy[3] = np.linspace(xl, xl, nym), np.linspace(ym, yl, nym)

    segsx[4], segsy[4] = np.linspace(xm, xh, nxm), np.linspace(yl, yl, nxm)
    segsx[5], segsy[5] = np.linspace(xh, xh, nym), np.linspace(yl, ym, nym)
    segsx[6], segsy[6] = np.linspace(xh, xm, nxm), np.linspace(ym, ym, nxm)

    segsx[7], segsy[7] = np.linspace(xh, xh, nym), np.linspace(ym, yh, nym)
    segsx[8], segsy[8] = np.linspace(xh, xm, nxm), np.linspace(yh, yh, nxm)
    segsx[9], segsy[9] = np.linspace(xm, xm, nym), np.linspace(ym, yh, nym)

    segsx[10], segsy[10] = np.linspace(xm, xl, nxm), np.linspace(yh, yh, nxm)
    segsx[11], segsy[11] = np.linspace(xl, xl, nym), np.linspace(yh, ym, nym)

    allx = np.concatenate(segsx)
    ally = np.concatenate(segsy)

    # print("plot::", _base_quadrent, '|', _uneven_mask, '|', len(allx), len(ally))

    pts3d = seed.to_3d(seed.uv_to_local(np.array([allx, ally])))
    _, all_topo = viscid.calc_streamlines(fld, pts3d, max_length=300.0,
                                          ibound=6.0, topo_style="msphere",
                                          output=viscid.OUTPUT_TOPOLOGY,
                                          **trace_opts)

    topo[0] = all_topo[:len(segsx[0])]
    cnt = len(topo[0])
    for i, segx in zip(count(1), segsx[1:]):
        topo[i] = all_topo[cnt:cnt + len(segx)]
        # print("??", i, cnt, cnt + len(segx), np.bitwise_and.reduce(topo[i]))
        cnt += len(topo[i])

    # assemble the lines into the four quadrents
    quad_topo = [None] * 4

    # all arrays snip off the last element since those are
    # duplicated by the next line... reversed arrays do the
    # snipping with -1:0:-1
    quad_topo[0] = np.concatenate([topo[0][:-1], topo[1][:-1],
                                   topo[2][:-1], topo[3][:-1]])

    quad_topo[1] = np.concatenate([topo[4][:-1], topo[5][:-1],
                                   topo[6][:-1], topo[1][-1:0:-1]])

    quad_topo[2] = np.concatenate([topo[2][-1:0:-1], topo[9][:-1],
                                   topo[10][:-1], topo[11][:-1]])

    quad_topo[3] = np.concatenate([topo[6][-1:0:-1], topo[7][:-1],
                                   topo[8][:-1], topo[9][-1:0:-1]])

    # now that the quad arrays are populated, decide which quadrents
    # still contain the separator (could be > 1)
    required_uneven_subquads = False
    ret = []
    for i in range(4):
        if perimeter_check(quad_topo[i]):
            next_quad = "{0}{1:x}".format(_base_quadrent, i | _uneven_mask)
            subquads = get_sep_pts_bisect(fld, seed, trace_opts=trace_opts,
                                          min_depth=min_depth,
                                          max_depth=max_depth, plot=plot,
                                          _base_quadrent=next_quad,
                                          _recurse0=False, _uneven_mask=0)
            ret += subquads

    if len(ret) == 0:
        perimeter = np.concatenate([topo[0][::-1], topo[4][::-1],
                                    topo[5][::-1], topo[7][::-1],
                                    topo[8][::-1], topo[10][::-1],
                                    topo[11][::-1], topo[3][::-1]])
        if _uneven_mask:
            if len(_base_quadrent) > min_depth:
                print("aw shucks, but min depth reached: {0} > {1}"
                      "".format(len(_base_quadrent), min_depth))
                ret = [_base_quadrent]
            else:
                print("aw shucks, the separator ended prematurely")
        elif perimeter_check(perimeter):
            ret = get_sep_pts_bisect(fld, seed, trace_opts=trace_opts,
                                     min_depth=min_depth, max_depth=max_depth,
                                     plot=plot, _base_quadrent=_base_quadrent,
                                     _recurse0=False, _uneven_mask=UNEVEN_MASK)
            required_uneven_subquads = True

    if plot and not required_uneven_subquads:
        from viscid.plot import mpl
        mpl.plt.scatter(allx, ally, color=np.bitwise_and(all_topo, 15),
                        vmin=0, vmax=15, marker='o', edgecolor='y', s=40)

    if _recurse0:
        # turn quadrent strings into locations
        xc = np.empty(len(ret))
        yc = np.empty(len(ret))
        for i, r in enumerate(ret):
            xc[i], yc[i] = _quadrent_center(r, xlim, ylim)
        pts_uv = np.array([xc, yc])
        if plot:
            from viscid.plot import mpl
            mpl.plt.plot(pts_uv[0], pts_uv[1], "y*", ms=20,
                         markeredgecolor='k', markeredgewidth=1.0)
            if plot.strip().lower() == "show":
                mpl.show()
        # return seed.to_3d(seed.uv_to_local(pts_uv))
        if make_3d:
            return seed.uv_to_3d(pts_uv)
        else:
            return pts_uv
    else:
        return ret

def _quadrent_limits(quad_str, xlim, ylim):
    xmin, xmax = xlim
    ymin, ymax = ylim
    xl, xh = xmin, xmax
    yl, yh = ymin, ymax

    for _, quad in enumerate(quad_str):
        quadi = int(quad, base=16)
        midpt = UNEVEN_HALF if quadi & UNEVEN_MASK else 0.5

        xm = xl + midpt * (xh - xl)
        if quadi & 1:
            xl = xm
        else:
            xh = xm

        ym = yl + midpt * (yh - yl)
        if quadi & 2:
            yl = ym
        else:
            yh = ym

    return xl, xh, yl, yh

def _quadrent_center(quad_str, xlim, ylim):
    xl, xh, yl, yh = _quadrent_limits(quad_str, xlim, ylim)
    midpt = UNEVEN_HALF if int(quad_str[-1], base=16) & UNEVEN_MASK else 0.5
    xm = xl + midpt * (xh - xl)
    ym = yl + midpt * (yh - yl)
    return xm, ym

def _make_square_segments(xl, xh, yl, yh, nx, ny):
    x = np.linspace(xl, xh, nx)
    y = np.linspace(yl, yh, ny)
    bottom = np.vstack((x, [y[0]] * x.shape[0]))
    right = np.vstack(([x[-1]] * y.shape[0], y))
    top = np.vstack((x[::-1], [y[-1]] * x.shape[0]))
    left = np.vstack(([x[0]] * (y.shape[0] - 1), y[::-1][:-1]))
    return bottom, right, top, left

def _make_square(xl, xh, yl, yh, nx, ny):
    bottom, right, top, left = _make_square_segments(xl, xh, yl, yh, nx, ny)
    return np.concatenate((bottom, right, top, left), axis=1)


# def perimeter_check_pattern(arr):
#     """Does perimeter of arr topologies contain a separator?
#     Returns:
#         bool
#     """
#     cyc = np.array([x[0] for x in groupby(arr)])
#     cyc = np.roll(cyc, -1 * np.argmin(cyc))
#     cyc_rev = np.roll(cyc[::-1], -1 * (len(cyc) - 1))
#     watch_list = [(0, 1, 2, 4)]
#     return bool(cyc in watch_list or cyc_rev in watch_list)

##
## EOF
##
