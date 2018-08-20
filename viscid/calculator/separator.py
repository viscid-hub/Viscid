#!/usr/bin/env python

from __future__ import division, print_function
from itertools import count

import numpy as np
import viscid
from viscid.compat import string_types
from viscid.cython import streamline


UNEVEN_MASK = 0b1000
UNEVEN_HALF = 0.65


__all__ = ["trace_separator", "topology_bitor_clusters", "get_sep_pts_bitor",
           "get_sep_pts_bisect"]


def trace_separator(grid, b_slcstr="x=-25j:15j, y=-30j:30j, z=-15j:15j",
                    r=1.0, plot=False, trace_opts=None, cache=True,
                    cache_dir=None):
    """Trace a separator line from most dawnward null

    **Still in testing** Uses the bisection algorithm.

    Args:
        grid (Grid): A grid that has a "b" field
        b_slcstr (str): Some valid slice for B field
        r (float): spatial step of separator line
        plot (bool): make debugging plots
        trace_opts (dict): passed to streamline function
        cache (bool, str): Save to and load from cache, if "force",
            then don't load from cache if it exists, but do save a
            cache at the end
        cache_dir (str): Directory for cache, if None, same directory
            as that file to which the grid belongs

    Raises:
        IOError: Description

    Returns:
        tuple: (separator_lines, nulls)

          - **separator_lines** (list): list of M 3xN ndarrays that
            represent M separator lines with N points
          - **nulls** (ndarray): 3xN array of N null points
    """
    if not cache_dir:
        cache_dir = grid.find_info("_viscid_dirname", "./")
    run_name = grid.find_info("run")
    sep_fname = "{0}/{1}.sep.{2:06.0f}".format(cache_dir, run_name, grid.time)

    try:
        if isinstance(cache, string_types) and cache.strip().lower() == "force":
            raise IOError()
        with np.load(sep_fname + ".npz") as dat:
            sep_iter = (f for f in dat.files if f.startswith("arr_"))
            _it = sorted(sep_iter, key=lambda s: int(s[len("arr_"):]))
            seps = [dat[n] for n in _it]
            nulls = dat['nulls']
    except IOError:
        _b = grid['b'][b_slcstr]

        _, nulls = viscid.find_nulls(_b['x=-30j:15j'], ibound=5.0)

        # get most dawnward null, nulls2 is all nulls except p0
        nullind = np.argmin(nulls[1, :])
        p0 = nulls[:, nullind]
        nulls2 = np.concatenate([nulls[:, :nullind], nulls[:, (nullind + 1):]],
                                axis=1)

        if plot:
            from viscid.plot import vlab
            vlab.plot_earth_3d(crd_system='gse')
            vlab.points3d(nulls2[0], nulls2[1], nulls2[2],
                          color=(0, 0, 0), scale_factor=1.0)
            vlab.points3d(nulls[0, nullind], nulls[1, nullind], nulls[2, nullind],
                          color=(1, 1, 1), scale_factor=1.0)

        seed = viscid.Sphere(p0=p0, r=r, ntheta=30, nphi=60,
                             theta_endpoint=True, phi_endpoint=True)
        p1 = viscid.get_sep_pts_bisect(_b, seed, max_depth=12, plot=plot,
                                       trace_opts=trace_opts)
        # print("p1 shape", p1.shape)

        # if p1.shape[1] > 2:
        #     raise RuntimeError("Invalid B field, should be no branch @ null")

        seps = []
        sep_stubs = []
        for i in range(p1.shape[1]):
            sep_stubs.append([p0, p1[:, i]])

        # print("??", sep_stubs)

        while sep_stubs:
            sep = sep_stubs.pop(0)
            # print("!!! new stub")

            for i in count():
                # print("::", i)
                seed = viscid.SphericalPatch(p0=sep[-1], p1=sep[-1] - sep[-2],
                                             r=r, nalpha=240, nbeta=240)
                pn = viscid.get_sep_pts_bisect(_b, seed, max_depth=8, plot=plot,
                                               trace_opts=trace_opts)
                if pn.shape[1] == 0:
                    # print("END: pn.shape[1] == 0")
                    break
                # print("++", nulls2.shape, pn.shape)
                closest_null_dist = np.min(np.linalg.norm(nulls2 - pn[:, :1], axis=0))
                # print("closest_null_dist:", closest_null_dist)
                if closest_null_dist < 1.01 * r:
                    # print("END: within 1.01 of a null")
                    break

                # print("??", pn)
                for j in range(1, pn.shape[1]):
                    # print("inserting new stub")
                    sep_stubs.insert(0, [sep[-1], pn[:, j]])
                sep.append(pn[:, 0])

            # print("sep", sep)
            seps.append(np.stack(sep, axis=1))

        if cache:
            np.savez_compressed(sep_fname, *seps, nulls=nulls)
    return seps, nulls

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
            slc = (slice(None), slice(None))
            slc[int(bnd[0])] = -1 if bnd[1] == "+" else 0
            a[slc] = np.bitwise_or.reduce(a[slc])

        indx, indy = np.where(a == sep_val)
        if i + 1 >= min_depth and len(indx):
            break

    pts = viscid.find_clusters(indx, indy, x, y, multiple=multiple,
                               periodic=periodic)

    if plot:
        from viscid.plot import vpyplot as vlt
        from matplotlib import pyplot as plt

        vlt.clf()
        ax0 = vlt.subplot(121)
        vlt.plot(fld, title=True)

        vlt.subplot(122, sharex=ax0, sharey=ax0)
        or_fld = viscid.arrays2field((x, y), a, name="OR")
        vlt.plot(or_fld, title=True)

        _x, _y = or_fld.get_crds()
        plt.plot(_x[indx], _y[indy], 'ko')

        plt.plot(pts[0], pts[1], 'y^')
        plt.show()

    return pts

def _prep_trace_opt_defaults(trace_opts):
    if trace_opts is None:
        trace_opts = dict()
    else:
        trace_opts = dict(trace_opts)
    trace_opts.setdefault('ibound', 2.5)
    trace_opts.setdefault('output', viscid.OUTPUT_TOPOLOGY)
    trace_opts.setdefault('max_length', 300.0)
    trace_opts.setdefault('topo_style', 'msphere')
    return trace_opts

def get_sep_pts_bitor(fld, seed, trace_opts=None, make_3d=True, **kwargs):
    """bitor topologies to find separator points in uv map from seed

    Args:
        fld (VectorField): Magnetic Field
        seed (viscid.seed.SeedGen): Any Seed generator with a 2d local
            representation
        trace_opts (dict): kwargs for calc_streamlines
        make_3d (bool): convert result from uv to 3d space
        **kwargs: passed to :py:func:`topology_bitor_clusters`

    Returns:
        3xN ndarray of N separator points in uv space or 3d space
        depending on the `make_3d` kwarg
    """
    trace_opts = _prep_trace_opt_defaults(trace_opts)
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
                       make_3d=True):
    """bisect uv map of seed to find separator points

    Args:
        fld (VectorField): Magnetic Field
        seed (viscid.seed.SeedGen): Any Seed generator with a 2d local
            representation
        trace_opts (dict): kwargs for calc_streamlines
        min_depth (int): Min allowable bisection depth
        max_depth (int): Max bisection depth
        plot (bool): Useful for debugging the algorithm
        perimeter_check (func): Some func that returns a bool with the
            same signature as :py:func:`perimeter_check_bitwise_or`
        make_3d (bool): convert result from uv to 3d space

    Returns:
        3xN ndarray of N separator points in uv space or 3d space
        depending on the `make_3d` kwarg
    """
    trace_opts = _prep_trace_opt_defaults(trace_opts)
    pts_even = _get_sep_pts_bisect(fld, seed, trace_opts=trace_opts,
                                   min_depth=0, max_depth=2, plot=False,
                                   perimeter_check=perimeter_check,
                                   make_3d=False)
    pts_uneven = _get_sep_pts_bisect(fld, seed, trace_opts=trace_opts,
                                     min_depth=0, max_depth=2, plot=False,
                                     perimeter_check=perimeter_check,
                                     make_3d=False, start_uneven=True)
    if pts_uneven.shape[1] > pts_even.shape[1]:
        start_uneven = True
    else:
        start_uneven = False

    # start_uneven = True
    return _get_sep_pts_bisect(fld, seed, trace_opts=trace_opts,
                               min_depth=min_depth, max_depth=max_depth,
                               plot=plot, start_uneven=start_uneven,
                               perimeter_check=perimeter_check,
                               make_3d=make_3d)

def _get_sep_pts_bisect(fld, seed, trace_opts=None, min_depth=3, max_depth=7,
                        plot=False, perimeter_check=perimeter_check_bitwise_or,
                        make_3d=True, start_uneven=False, _base_quadrent="",
                        _uneven_mask=0, _first_recurse=True):
    if len(_base_quadrent) == max_depth:
        return [_base_quadrent]  # causes pylint to complain
    if trace_opts is None:
        trace_opts = dict()

    nx, ny = seed.uv_shape
    (xlim, ylim) = seed.uv_extent

    if _first_recurse and start_uneven:
        _uneven_mask = UNEVEN_MASK

    if _first_recurse and plot:
        from viscid.plot import vlab
        from viscid.plot import vpyplot as vlt
        vlt.clf()
        _, all_topo = viscid.calc_streamlines(fld, seed, **trace_opts)
        vlt.plot(np.bitwise_and(all_topo, 15), show=False)
        verts, arr = seed.wrap_mesh(all_topo.data)
        vlab.mesh(verts[0], verts[1], verts[2], scalars=arr, opacity=0.75)

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
    _, all_topo = viscid.calc_streamlines(fld, pts3d, **trace_opts)

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
            subquads = _get_sep_pts_bisect(fld, seed, trace_opts=trace_opts,
                                           min_depth=min_depth,
                                           max_depth=max_depth, plot=plot,
                                           _base_quadrent=next_quad,
                                           _uneven_mask=0,
                                           _first_recurse=False)
            ret += subquads

    if len(ret) == 0:
        perimeter = np.concatenate([topo[0][::-1], topo[4][::-1],
                                    topo[5][::-1], topo[7][::-1],
                                    topo[8][::-1], topo[10][::-1],
                                    topo[11][::-1], topo[3][::-1]])
        if _uneven_mask:
            if len(_base_quadrent) > min_depth:
                print("sep trace issue, but min depth reached: {0} > {1}"
                      "".format(len(_base_quadrent), min_depth))
                ret = [_base_quadrent]
            else:
                print("sep trace issue, the separator ended prematurely")
        elif perimeter_check(perimeter):
            ret = _get_sep_pts_bisect(fld, seed, trace_opts=trace_opts,
                                      min_depth=min_depth, max_depth=max_depth,
                                      plot=plot, _base_quadrent=_base_quadrent,
                                      _uneven_mask=UNEVEN_MASK,
                                      _first_recurse=False)
            required_uneven_subquads = True

    if plot and not required_uneven_subquads:
        from viscid.plot import vlab
        from viscid.plot import vpyplot as vlt
        from matplotlib import pyplot as plt
        _pts3d = seed.to_3d(seed.uv_to_local(np.array([allx, ally])))
        vlab.points3d(_pts3d[0], _pts3d[1], _pts3d[2],
                      all_topo.data.reshape(-1), scale_mode='none',
                      scale_factor=0.02)
        plt.scatter(allx, ally, color=np.bitwise_and(all_topo, 15),
                        vmin=0, vmax=15, marker='o', edgecolor='y', s=40)

    if _first_recurse:
        # turn quadrent strings into locations
        xc = np.empty(len(ret))
        yc = np.empty(len(ret))
        for i, r in enumerate(ret):
            xc[i], yc[i] = _quadrent_center(r, xlim, ylim)
        pts_uv = np.array([xc, yc])
        if plot:
            from viscid.plot import vlab
            from viscid.plot import vpyplot as vlt
            from matplotlib import pyplot as plt
            plt.plot(pts_uv[0], pts_uv[1], "y*", ms=20,
                         markeredgecolor='k', markeredgewidth=1.0)
            vlt.show(block=False)
            vlab.show(stop=True)
        # return seed.to_3d(seed.uv_to_local(pts_uv))
        # if pts_uv.size == 0:
        #     return None
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
        try:
            quadi = int(quad, base=16)
        except TypeError:
            raise
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
