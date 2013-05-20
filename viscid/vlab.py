from __future__ import print_function
import multiprocessing as mp
import itertools
import logging

from . import readers
from . import verror

def load_vfile(fname):
    readers.load(fname)

def _do_multiplot(tind, grid, plot_vars, global_popts=None, share_axes=False,
                  out_prefix=None, show=False):
    import matplotlib.pyplot as plt
    from viscid.plot import mpl
    
    logging.info("Plotting timestep: {0}, {1}".format(tind, grid.time))

    nrows = len(plot_vars)
    ncols = 1

    if nrows == 0:
        logging.warn("I have no variables to plot")
        return

    shareax = None

    for i, fld_meta in enumerate(plot_vars):
        with grid[fld_meta[0]] as fld:
            # print("fld_time:", fld.time)
            ax = plt.subplot2grid((nrows, ncols), (i, 0),
                                  sharex=shareax, sharey=shareax)
            if i == 0 and share_axes:
                shareax = ax

            if not "plot_opts" in fld_meta[1]:
                fld_meta[1]["plot_opts"] = global_popts
            elif global_popts is not None:
                fld_meta[1]["plot_opts"] = "{0},{1}".format(
                                    fld_meta[1]["plot_opts"], global_popts)
            mpl.plot(fld, **fld_meta[1])
        # print("fld cache", grid[fld_meta[0]]._cache)
    hrs = int(grid.time / 3600)
    mins = int((grid.time / 60) % 3600)
    secs = grid.time % 60
    # plt.suptitle("t = {0:.2f}".format(grid.time))
    plt.suptitle("t = {0}:{1}:{2:.2f}".format(hrs, mins, secs))

    if out_prefix:
        plt.savefig("{0}_{1:06d}.png".format(out_prefix, tind + 1))
    if show:
        plt.show()
    plt.clf()

def _do_multiplot_star(all_args):
    try:
        return _do_multiplot(*all_args) #pylint: disable=W0142
    except KeyboardInterrupt:
        raise verror.KeyboardInterruptError()

def multiplot(files, plot_vars, np=1, time_slice=":", global_popts=None,
              share_axes=False, out_prefix=None, show=False):
    grid_iter = itertools.izip(
                               itertools.count(),
                               files[0].iter_times(time_slice),
                               itertools.repeat(plot_vars),
                               itertools.repeat(global_popts),
                               itertools.repeat(share_axes),
                               itertools.repeat(out_prefix),
                               itertools.repeat(show),
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
