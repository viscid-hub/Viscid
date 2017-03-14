#!/usr/bin/env python
""" test a ggcm grid wrapper """

from __future__ import print_function
import argparse
import os

from viscid_test_common import next_plot_fname

import viscid
from viscid import sample_dir
from viscid import vutil
from viscid.plot import mpl
from viscid.plot.mpl import plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)

    ####### test binary files
    f_bin = viscid.load_file(os.path.join(sample_dir, 'ath_sample.*.bin'))

    for i, grid in enumerate(f_bin.iter_times(":")):
        plt.subplot2grid((2, 2), (0, i))
        mpl.plot(grid['bx'])
        plt.subplot2grid((2, 2), (1, i))
        mpl.plot(grid['by'])
    mpl.plt.suptitle("athena bin (binary) files")
    mpl.auto_adjust_subplots(subplot_params=dict(top=0.9))

    mpl.plt.savefig(next_plot_fname(__file__))
    if args.show:
        mpl.show()
    plt.clf()

    ####### test ascii files
    f_tab = viscid.load_file(os.path.join(sample_dir, 'ath_sample.*.tab'))

    for i, grid in enumerate(f_tab.iter_times(":")):
        plt.subplot2grid((2, 2), (0, i))
        mpl.plot(grid['bx'])
        plt.subplot2grid((2, 2), (1, i))
        mpl.plot(grid['by'])
    mpl.plt.suptitle("athena tab (ascii) files")
    mpl.auto_adjust_subplots(subplot_params=dict(top=0.9))

    mpl.plt.savefig(next_plot_fname(__file__))
    if args.show:
        mpl.show()
    plt.clf()

if __name__ == "__main__":
    main()

##
## EOF
##
