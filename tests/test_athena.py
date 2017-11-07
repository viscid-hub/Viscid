#!/usr/bin/env python
"""Test the Athena bin and tab readers"""

from __future__ import print_function
import argparse
import sys
import os

import matplotlib.pyplot as plt

from viscid_test_common import next_plot_fname

import viscid
from viscid import sample_dir
from viscid import vutil
from viscid.plot import vpyplot as vlt


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)

    ####### test binary files
    f_bin = viscid.load_file(os.path.join(sample_dir, 'ath_sample.*.bin'))

    plt.figure()
    for i, grid in enumerate(f_bin.iter_times(":")):
        plt.subplot2grid((2, 2), (0, i))
        vlt.plot(grid['bx'])
        plt.subplot2grid((2, 2), (1, i))
        vlt.plot(grid['by'])
    plt.suptitle("athena bin (binary) files")
    vlt.auto_adjust_subplots(subplot_params=dict(top=0.9))

    plt.savefig(next_plot_fname(__file__))
    if args.show:
        vlt.show()
    plt.close()

    ####### test ascii files
    f_tab = viscid.load_file(os.path.join(sample_dir, 'ath_sample.*.tab'))

    plt.figure()
    for i, grid in enumerate(f_tab.iter_times(":")):
        plt.subplot2grid((2, 2), (0, i))
        vlt.plot(grid['bx'])
        plt.subplot2grid((2, 2), (1, i))
        vlt.plot(grid['by'])
    plt.suptitle("athena tab (ascii) files")
    vlt.auto_adjust_subplots(subplot_params=dict(top=0.9))

    plt.savefig(next_plot_fname(__file__))
    if args.show:
        vlt.show()
    plt.close()

    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
