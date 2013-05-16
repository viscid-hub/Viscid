#!/usr/bin/env python

from __future__ import print_function
import argparse
import subprocess as sub
import multiprocessing as mp
import itertools
import logging

from viscid import readers
from viscid import vutil


class KeyboardInterruptError(Exception):
    pass


def splitopt(string, ret_type=float, delim=None):
    return [ret_type(s) for s in string.split(delim)]

def grid_printer(grid):
    print(grid)

def do_plot(tind, grid, plot_flds, args):
    import matplotlib.pyplot as plt
    from viscid.plot import mpl
    
    nrows = len(plot_flds)
    ncols = 1

    logging.info("Plotting timestep: {0}, {1}".format(tind, grid.time))

    for i, fld_meta in enumerate(plot_flds):
        with grid[fld_meta[0]] as fld:
            # print("fld_time:", fld.time)
            plt.subplot2grid((nrows, ncols), (i, 0))
            mpl.plot(fld, **fld_meta[1])
        # print("fld cache", grid[fld_meta[0]]._cache)
    plt.suptitle("t = {0:.2f}".format(grid.time))
    plt.savefig("{0}_{1:06d}.png".format(args.prefix, tind + 1))
    if args.show:
        plt.show()
    plt.clf()

def do_plot_star(all_args):
    try:
        return do_plot(*all_args) #pylint: disable=W0142
    except KeyboardInterrupt:
        raise KeyboardInterruptError()

def main():
    parser = argparse.ArgumentParser(description="Load some data files")
    parser.add_argument("-t", default=":", help="times to plot in slice "
                        "notation: ex : for all 60.0: for 60 mins on, and "
                        ":120 for the first 120mins")
    parser.add_argument("-a", "--animate", default=None,
                        help="animate results")
    parser.add_argument('-r', '--rate', dest='framerate', type=int, default=5,
                      help="animation frame rate (default 5).")
    parser.add_argument('--qscale', dest='qscale', default='2',
                      help="animation quality flag (default 2).")
    parser.add_argument('-k', dest='keep', action='store_true',
                      help="keep temporary files.")
    parser.add_argument('-w', '--show', dest='show', action="store_true",
                      help="show plots with plt.show()")
    parser.add_argument("-n", "--np", type=int, default=1,
                        help="run n simultaneous processes (not yet working)")
    parser.add_argument('file', nargs=1, help='input file')
    args = vutil.common_argparse(parser)
    args.prefix = "tmp_image"
    print(args)

    plot_flds = [("pp", {"plot_opts":"x_-15_15,y_-15_15,log"}),
                 ("bz", {"plot_opts":"x_-15_15,y_-15_15,lin_-10_10"})]

    files = readers.load(args.file)

    grid_iter = itertools.izip(itertools.count(),
                               files[0].iter_times(args.t),
                               itertools.repeat(plot_flds),
                               itertools.repeat(args),
                              )
    
    if True:
        map(do_plot_star, grid_iter)
    else:
        # h5py is unhappy about this
        pool = mp.Pool(args.np)
        pool.map(do_plot_star, grid_iter)
        pool.close()
        pool.join()
    
    if args.animate:
        sub.Popen("ffmpeg -r {0} -qscale {1} -i {2}_%06d.png {3}".format(
                  args.framerate, args.qscale, args.prefix, args.animate),
                  shell=True).communicate()
    if not args.keep:
        sub.Popen("rm {0}_*.png".format(args.prefix), shell=True).communicate()


if __name__ == "__main__":
    main()

##
## EOF
##
