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

from . import parallel
from . import field
from . import coordinate

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
    if not m:
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
                            deep_meta={"force_layout": field.LAYOUT_INTERLACED},
                           )
    # fld_rsq = field.ScalarField("r", crds, hmm,
    #                             center="Cell", forget_source=True)
    return fld  # , fld_rsq

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

    nrows = len(plot_vars)
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

    for i, fld_meta in enumerate(plot_vars):
        fld_name_split = fld_meta[0].split(',')
        fld_name = fld_name_split[0]
        fld_slc = ",".join(fld_name_split[1:])
        if selection is not None:
            # fld_slc += ",{0}".format(selection)
            if fld_slc != "":
                fld_slc = ",".join([fld_slc, selection])
            else:
                fld_slc = selection

        # print("fld_time:", fld.time)
        row = i
        col = 0
        if transpose:
            row, col = col, row
        ax = plt.subplot2grid((nrows, ncols), (row, col),
                              sharex=shareax, sharey=shareax)
        if i == 0 and share_axes:
            shareax = ax

        if not "plot_opts" in fld_meta[1]:
            fld_meta[1]["plot_opts"] = global_popts
        elif global_popts is not None:
            fld_meta[1]["plot_opts"] = "{0},{1}".format(
                fld_meta[1]["plot_opts"], global_popts)

        # FIXME: this method of plotting T, beta, etc. really sucks
        # also... do i really need with statements here?
        if fld_name == "T":
            with grid["pp"] as pp, grid["rr"] as rr:
                mpl.plot(pp / rr, selection=fld_slc, **fld_meta[1])
        elif fld_name == "beta":
            with grid["bx"] as bx, grid["by"] as by, \
                 grid["bz"] as bz, grid["pp"] as pp:
                beta = pp / (bx**2 + by**2 + bz**2)
                mpl.plot(beta, selection=fld_slc, **fld_meta[1])
        else:
            with grid[fld_name] as fld:
                mpl.plot(fld, selection=fld_slc, **fld_meta[1])
        # print("fld cache", grid[fld_meta[0]]._cache)

    hrs = int(grid.time / 3600)
    mins = int((grid.time / 60) % 60)
    secs = grid.time % 60
    # plt.suptitle("t = {0:.2f}".format(grid.time))
    plt.suptitle("\nt = {0}:{1:02}:{2:05.2f}".format(hrs, mins, secs))
    mpl.tighten()

    if out_prefix:
        plt.savefig("{0}_{1:06d}.{2}".format(out_prefix, tind + 1, out_format))
    if show:
        plt.show()
    plt.clf()

def multiplot(file_, plot_vars, np=1, time_slice=":", global_popts=None,
              share_axes=False, show=False, kwopts=None):
    grid_iter = izip(
                     itertools.count(),
                     file_.iter_times(time_slice),
                     itertools.repeat(plot_vars),
                     itertools.repeat(global_popts),
                     itertools.repeat(share_axes),
                     itertools.repeat(show),
                     itertools.repeat(kwopts),
                    )
    parallel.map(np, _do_multiplot, grid_iter)

##
## EOF
##
