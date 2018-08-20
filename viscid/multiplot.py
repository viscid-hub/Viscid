#!/usr/bin/env python
"""Deal with efficiently making large series of plots"""

from __future__ import print_function
import itertools

from viscid.compat import izip
from viscid import logger
from viscid import parallel


__all__ = ['make_multiplot']


def make_multiplot(vfile, plot_func=None, nr_procs=1, time_slice=":", **kwargs):
    """Make lots of plots

    Calls plot_func (or `_do_multiplot` if plot_func is None) with 2
    positional arguments (int, Grid), and all the kwargs given to
    multiplot.

    Grid is determined by vfile.iter_times(time_slice).

    plot_func gets additional keyword arguments first_run (bool) and
    first_run_result (whatever is returned from plot_func by the first
    call).

    This is the function used by the ``p2d`` script. It may be useful
    to you.

    Args:
        vfile (VFile, Grid): Something that has iter_times
        plot_func (callable): Function that makes a single plot. It
            must take an int (index of time slice), a Grid, and any
            number of keyword argumets. If None, _do_multiplot is used
        nr_procs (int): number of parallel processes to farm out
            plot_func to
        time_slice (str): passed to vfile.iter_times()
        **kwargs: passed as keword aguments to plot_func
    """
    # make sure time slice yields >= 1 actual time slice
    try:
        next(vfile.iter_times(time_slice))
    except StopIteration:
        raise ValueError("Time slice '{0}' yields no data".format(time_slice))

    if plot_func is None:
        plot_func = _do_multiplot

    grid_iter = izip(itertools.count(), vfile.iter_times(time_slice))

    args_kw = kwargs.copy()
    args_kw["first_run"] = True
    args_kw["first_run_result"] = None

    if "subplot_params" not in args_kw.get("kwopts", {}):
        r = parallel.map(1, plot_func, [next(grid_iter)], args_kw=args_kw,
                         force_subprocess=(nr_procs > 1))

    # now get back to your regularly scheduled programming
    args_kw["first_run"] = False
    args_kw["first_run_result"] = r[0]
    parallel.map(nr_procs, plot_func, grid_iter, args_kw=args_kw)

def _do_multiplot(tind, grid, plot_vars=None, global_popts=None, kwopts=None,
                  share_axes=False, show=False, subplot_params=None,
                  first_run_result=None, first_run=False, **kwargs):
    from viscid.plot import vpyplot as vlt
    import matplotlib.pyplot as plt

    logger.info("Plotting timestep: %d, %g", tind, grid.time)

    if plot_vars is None:
        raise ValueError("No plot_vars given to `_do_multiplot` :(")
    if kwargs:
        logger.info("Unused kwargs: {0}".format(kwargs))

    if kwopts is None:
        kwopts = {}
    transpose = kwopts.get("transpose", False)
    plot_size = kwopts.get("plot_size", None)
    dpi = kwopts.get("dpi", None)
    out_prefix = kwopts.get("out_prefix", None)
    out_format = kwopts.get("out_format", "png")
    selection = kwopts.get("selection", None)
    timeformat = kwopts.get("timeformat", ".02f")
    tighten = kwopts.get("tighten", False)
    # wicked hacky
    # subplot_params = kwopts.get("subplot_params", _subplot_params)

    # nrows = len(plot_vars)
    nrows = len([pv[0] for pv in plot_vars if not pv[0].startswith('^')])
    ncols = 1
    if transpose:
        nrows, ncols = ncols, nrows

    if nrows == 0:
        logger.warning("I have no variables to plot")
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
        if '=' in fld_name_split[0]:
            # if fld_name is actually an equation, assume
            # there's no slice, and commas are part of the
            # equation
            fld_name = ",".join(fld_name_split)
            fld_slc = ""
        else:
            fld_name = fld_name_split[0]
            fld_slc = ",".join(fld_name_split[1:])
        if selection is not None:
            # fld_slc += ",{0}".format(selection)
            if fld_slc != "":
                fld_slc = ",".join([fld_slc, selection])
            else:
                fld_slc = selection
        if fld_slc.strip() == "":
            fld_slc = Ellipsis

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

        if "plot_opts" not in fld_meta[1]:
            fld_meta[1]["plot_opts"] = global_popts
        elif global_popts is not None:
            fld_meta[1]["plot_opts"] = "{0},{1}".format(
                fld_meta[1]["plot_opts"], global_popts)

        with grid.get_field(fld_name, slc=fld_slc) as fld:
            vlt.plot(fld, masknan=True, **fld_meta[1])
        # print("fld cache", grid[fld_meta[0]]._cache)

    if timeformat and timeformat.lower() != "none":
        plt.suptitle(grid.format_time(timeformat))

    # for adjusting subplots / tight_layout and applying the various
    # hacks to keep plots from dancing around in movies
    if not subplot_params and first_run_result:
        subplot_params = first_run_result
    if tighten:
        tighten = dict(rect=[0, 0.03, 1, 0.90])
    ret = vlt.auto_adjust_subplots(tight_layout=tighten,
                                   subplot_params=subplot_params)
    if not first_run:
        ret = None

    if out_prefix:
        plt.savefig("{0}_{1:06d}.{2}".format(out_prefix, tind + 1, out_format))
    if show:
        plt.show()
    plt.clf()

    return ret

##
## EOF
##
