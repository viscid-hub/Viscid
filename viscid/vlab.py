from __future__ import print_function
import multiprocessing as mp
import itertools
import logging

from . import readers
from . import verror

def load_vfile(fname):
    readers.load(fname)

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
        with grid[fld_name] as fld:
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
            mpl.plot(fld, selection=fld_slc, **fld_meta[1])
        # print("fld cache", grid[fld_meta[0]]._cache)
    hrs = int(grid.time / 3600)
    mins = int((grid.time / 60) % 60)
    secs = grid.time % 60
    # plt.suptitle("t = {0:.2f}".format(grid.time))
    plt.suptitle("t = {0}:{1:02}:{2:05.2f}".format(hrs, mins, secs))
    # mpl.tighten()

    if out_prefix:
        plt.savefig("{0}_{1:06d}.{2}".format(out_prefix, tind + 1, out_format))
    if show:
        plt.show()
    plt.clf()

def _do_multiplot_star(all_args):
    try:
        return _do_multiplot(*all_args) #pylint: disable=W0142
    except KeyboardInterrupt:
        raise verror.KeyboardInterruptError()

def multiplot(files, plot_vars, np=1, time_slice=":", global_popts=None,
              share_axes=False, show=False, kwopts=None):
    grid_iter = itertools.izip(
                               itertools.count(),
                               files[0].iter_times(time_slice),
                               itertools.repeat(plot_vars),
                               itertools.repeat(global_popts),
                               itertools.repeat(share_axes),
                               itertools.repeat(show),
                               itertools.repeat(kwopts),
                              )
    if np == 1:
        for args in grid_iter:
            _do_multiplot_star(args)
    else:
        pool = mp.Pool(np)
        pool.map(_do_multiplot_star, grid_iter)
        pool.close()
        pool.join()

##
## EOF
##
