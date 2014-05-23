from __future__ import print_function
import logging
import itertools
try:
    from itertools import izip
except ImportError:
    izip = zip

import numpy as np
try:
    import numexpr as ne
except ImportError:
    pass

from viscid import parallel
from viscid import field
from viscid import coordinate
from viscid.calculator import seed

# these compiled are needed for fluid following
try:
    from viscid.calculator import cycalc
    from viscid.calculator import streamline
    from viscid.calculator.streamline import streamlines
except ImportError:
    pass

def get_dipole(m=None, l=None, h=None, n=None, twod=False):
    dtype = 'float64'
    if l is None:
        l = [-5] * 3
    if h is None:
        h = [5] * 3
    if n is None:
        n = [256] * 3
    x = np.array(np.linspace(l[0], h[0], n[0]), dtype=dtype)
    y = np.array(np.linspace(l[1], h[1], n[1]), dtype=dtype)
    z = np.array(np.linspace(l[2], h[2], n[2]), dtype=dtype)
    if twod:
        y = np.array(np.linspace(-0.1, 0.1, 2), dtype=dtype)
    crds = coordinate.wrap_crds("nonuniform_cartesian", (('z', z), ('y', y), ('x', x)))

    one = np.array([1.0], dtype=dtype) #pylint: disable=W0612
    three = np.array([3.0], dtype=dtype) #pylint: disable=W0612
    if m is None:
        m = [0.0, 0.0, -1.0]
    m = np.array(m, dtype=dtype)
    mx, my, mz = m #pylint: disable=W0612

    Zcc, Ycc, Xcc = crds.get_crds_cc(shaped=True) #pylint: disable=W0612

    rsq = ne.evaluate("Xcc**2 + Ycc**2 + Zcc**2") #pylint: disable=W0612
    mdotr = ne.evaluate("mx * Xcc + my * Ycc + mz * Zcc") #pylint: disable=W0612
    Bx = ne.evaluate("((three * Xcc * mdotr / rsq) - mx) / rsq**1.5")
    By = ne.evaluate("((three * Ycc * mdotr / rsq) - my) / rsq**1.5")
    Bz = ne.evaluate("((three * Zcc * mdotr / rsq) - mz) / rsq**1.5")

    fld = field.VectorField("B_cc", crds, [Bx, By, Bz],
                            center="Cell", forget_source=True,
                            _force_layout=field.LAYOUT_INTERLACED,
                           )
    # fld_rsq = field.ScalarField("r", crds, hmm,
    #                             center="Cell", forget_source=True)
    return fld  # , fld_rsq

def get_trilinear_field():
    xl, xh, nx = -1.0, 1.0, 41
    yl, yh, ny = -1.5, 1.5, 41
    zl, zh, nz = -2.0, 2.0, 41
    x = np.linspace(xl, xh, nx)
    y = np.linspace(yl, yh, ny)
    z = np.linspace(zl, zh, nz)
    crds = coordinate.wrap_crds("nonuniform_cartesian",
                                [('z', z), ('y', y), ('x', x)])
    b = field.empty("Vector", "f", crds, 3, center="Cell",
                      layout="interlaced")
    Z, Y, X = b.get_crds(shaped=True)

    x01, y01, z01 = 0.5, 0.5, 0.5
    x02, y02, z02 = 0.5, 0.5, 0.5
    x03, y03, z03 = 0.5, 0.5, 0.5

    b['x'][:] = 0.0 + 1.0 * (X - x01) + 1.0 * (Y - y01) + 1.0 * (Z - z01) + \
                1.0 * (X - x01) * (Y - y01) + 1.0 * (Y - y01) * (Z - z01) + \
                1.0 * (X - x01) * (Y - y01) * (Z - z01)
    b['y'][:] = 0.0 + 1.0 * (X - x02) - 1.0 * (Y - y02) + 1.0 * (Z - z02) + \
                1.0 * (X - x02) * (Y - y02) + 1.0 * (Y - y02) * (Z - z02) - \
                1.0 * (X - x02) * (Y - y02) * (Z - z02)
    b['z'][:] = 0.0 + 1.0 * (X - x03) + 1.0 * (Y - y03) - 1.0 * (Z - z03) + \
                1.0 * (X - x03) * (Y - y03) + 1.0 * (Y - y03) * (Z - z03) + \
                1.0 * (X - x03) * (Y - y03) * (Z - z03)
    return b

def multiplot(file_, plot_vars, nprocs=1, time_slice=":", global_popts=None,
              share_axes=False, show=False, kwopts=None):
    """Make lots of plots

    This is the function used by the ``p2d`` script. It may be useful
    to you.
    """
    grid_iter = izip(
                     itertools.count(),
                     file_.iter_times(time_slice),
                     itertools.repeat(plot_vars),
                     itertools.repeat(global_popts),
                     itertools.repeat(share_axes),
                     itertools.repeat(show),
                     itertools.repeat(kwopts),
                    )
    parallel.map(nprocs, _do_multiplot, grid_iter)

def _do_multiplot(tind, grid, plot_vars, global_popts=None, share_axes=False,
                  show=False, kwopts=None):
    import matplotlib.pyplot as plt
    from viscid.plot import mpl

    logging.info("Plotting timestep: {0}, {1}".format(tind, grid.time))

    if kwopts is None:
        kwopts = {}
    transpose = kwopts.get("transpose", False)
    plot_size = kwopts.get("plot_size", None)
    dpi = kwopts.get("dpi", None)
    out_prefix = kwopts.get("out_prefix", None)
    out_format = kwopts.get("out_format", "png")
    selection = kwopts.get("selection", None)
    fancytime = kwopts.get("fancytime", False)

    # nrows = len(plot_vars)
    nrows = len([pv[0] for pv in plot_vars if not pv[0].startswith('^')])
    ncols = 1
    if transpose:
        nrows, ncols = ncols, nrows

    if nrows == 0:
        logging.warn("I have no variables to plot")
        return

    fig = plt.gcf()
    if plot_size is not None:
        fig.set_size_inches(*plot_size, forward=True)
    if dpi is not None:
        fig.set_dpi(dpi)

    shareax = None

    this_row = -1
    for i, fld_meta in enumerate(plot_vars):
        if not fld_meta[0].startswith('^'):
            this_row += 1
            same_axis = False
        else:
            same_axis = True

        fld_name_meta = fld_meta[0].lstrip('^')
        fld_name_split = fld_name_meta.split(',')
        fld_name = fld_name_split[0]
        fld_slc = ",".join(fld_name_split[1:])
        if selection is not None:
            # fld_slc += ",{0}".format(selection)
            if fld_slc != "":
                fld_slc = ",".join([fld_slc, selection])
            else:
                fld_slc = selection

        # print("fld_time:", fld.time)
        if this_row < 0:
            raise ValueError("first plot can't begin with a +")
        row = this_row
        col = 0
        if transpose:
            row, col = col, row
        if not same_axis:
            ax = plt.subplot2grid((nrows, ncols), (row, col),
                                  sharex=shareax, sharey=shareax)
        if i == 0 and share_axes:
            shareax = ax

        if not "plot_opts" in fld_meta[1]:
            fld_meta[1]["plot_opts"] = global_popts
        elif global_popts is not None:
            fld_meta[1]["plot_opts"] = "{0},{1}".format(
                fld_meta[1]["plot_opts"], global_popts)

        with grid[fld_name] as fld:
            mpl.plot(fld, selection=fld_slc, mask_nan=True, **fld_meta[1])
        # print("fld cache", grid[fld_meta[0]]._cache)

    if fancytime:
        # TODO: look for actual UT time in the grid?
        hrs = int(grid.time / 3600)
        mins = int((grid.time / 60) % 60)
        secs = grid.time % 60
        # plt.suptitle("t = {0:.2f}".format(grid.time))
        plt.suptitle("t = {0}:{1:02}:{2:05.2f}".format(hrs, mins, secs))
    else:
        plt.suptitle("t = {0:g}".format(grid.time))

    mpl.tighten(rect=[0, 0.03, 1, 0.90])

    if out_prefix:
        plt.savefig("{0}_{1:06d}.{2}".format(out_prefix, tind + 1, out_format))
    if show:
        plt.show()
    plt.clf()

def follow_fluid(vfile, time_slice, initial_seeds, plot_function,
                 stream_opts, add_seed_cadence=0.0, add_seed_pts=None,
                 speed_scale=1.0):
    """Trace fluid elements

    Parameters:
        vfile: a vFile object that we can call iter_times on
        time_slice: string, slice notation, like 1200:2400:1
        initial_seeds: any SeedGen object
        plot_function: function that is called each time step,
            arguments should be exactly: (i [int], grid, v [Vector
            Field], v_lines [result of streamline trace],
            root_seeds [SeedGen])
        stream_opts: a dict of streamline options for flow lines
        add_seed_cadence: how often to add the add_seeds points
        add_seeds: an n x 3 ndarray of n points to add every add_seed_cadence (zyx)
        speed_scale: speed_scale * v should be in units of ds0 / dt
        stream_opts must have ds0 and max_length, maxit will be automatically calculated

    Returns:
        root points after following the fluid
    """
    times = np.array([grid.time for grid in vfile.iter_times(time_slice)])
    dt = np.roll(times, -1) - times  # Note: last element makes no sense
    root_seeds = initial_seeds

    # setup maximum number of iterations for a streamline
    if "ds0" in stream_opts and "max_length" in stream_opts:
        max_length = stream_opts["max_length"]
        ds0 = stream_opts["ds0"]
        stream_opts["maxit"] = (max_length // ds0) + 1
    else:
        raise KeyError("ds0 and max_length must be keys of stream_opts, "
                       "otherwise I don't know how to follow the fluid")

    # iterate through the time steps from time_slice
    for i, grid in enumerate(vfile.iter_times(time_slice)):
        if i == 0:
            last_add_time = grid.time

        root_pts = _follow_fluid_step(i, dt[i], grid, root_seeds,
                                      plot_function, stream_opts, speed_scale)

        # maybe add some new seed points to account for those that have left
        if add_seed_pts and abs(grid.time - last_add_time) >= add_seed_cadence:
            root_pts = np.concatenate([root_pts, add_seed_pts.T], axis=1)
            last_add_time = grid.time
        root_seeds = seed.Point(root_pts)

    return root_seeds

def _follow_fluid_step(i, dt, grid, root_seeds, plot_function, stream_opts,
                       speed_scale):
    logging.info("working on timestep {0} {1}".format(i, grid.time))
    v = grid["v"]
    logging.debug("finished reading V field")

    logging.debug("calculating new streamline positions")
    flow_lines = streamlines(v, root_seeds,  # pylint: disable=W0142
                             output=streamline.OUTPUT_STREAMLINES,
                             **stream_opts)[0]

    logging.debug("done with that, now i'm plotting...")
    plot_function(i, grid, v, flow_lines, root_seeds)

    ############################################################
    # now recalculate the seed positions for the next timestep
    logging.debug("finding new seed positions...")
    root_pts = root_seeds.genr_points()
    valid_pt_inds = []
    for i in range(root_pts.shape[1]):
        valid_pt = True

        # get the index of the root point in teh 2d flow line array
        dist = flow_lines[i] - root_pts[:, [i]]
        root_ind = np.argmin(np.sum(dist**2, axis=0))
        # print("!!!", root_pts[:, i], "==", flow_lines[i][:, root_ind])
        # interpolate velocity onto teh flow line, and get speed too
        v_interp = cycalc.interp_trilin(v, seed.Point(flow_lines[i]))
        speed = np.sqrt(np.sum(v_interp * v_interp, axis=1))
        direction = int(dt / np.abs(dt))

        # this is a super slopy way to integrate velocity
        # keep marching along the flow line until we get to the next timestep
        t = 0.0
        ind = root_ind
        while np.abs(t) < np.abs(dt):
            ind += direction
            if ind < 0 or ind >= v_interp.shape[0]:
                # set valid_pt to True if you want to keep that point for
                # future time steps, but most likely if we're here, the seed
                # has gone out of our region of interest
                ind = max(min(ind, v_interp.shape[0] - 1), 0)
                valid_pt = False
                logging.warning("OOPS: ran out of streamline, increase "
                                "max_length when tracing flow lines if this "
                                "is unexpected")
                break

            t += stream_opts["ds0"] / (speed_scale * speed[ind])

        root_pts[:, i] = flow_lines[i][:, ind]
        if valid_pt:
            valid_pt_inds.append(i)

    # remove seeds that have flown out of our region of interest
    # (aka, beyond the flow line we drew)
    root_pts = root_pts[:, valid_pt_inds]

    logging.debug("ok, done with all that :)")
    return root_pts

##
## EOF
##
