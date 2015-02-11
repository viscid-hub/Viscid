#!/usr/bin/env python
""" test a ggcm grid wrapper """

from __future__ import print_function
import sys
import os
import argparse

_viscid_root = os.path.realpath(os.path.dirname(__file__) + '/../viscid/')
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)

import viscid
from viscid import vutil
from viscid.plot import mpl
from viscid.plot.mpl import plt

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)

    ####### test binary files
    f_bin = viscid.load_file(_viscid_root + '/../sample/ath_sample.*.bin')

    for i, grid in enumerate(f_bin.iter_times(":")):
        plt.subplot2grid((2, 2), (0, i))
        mpl.plot(grid['bx'])
        plt.subplot2grid((2, 2), (1, i))
        mpl.plot(grid['by'])
    if args.show:
        mpl.tighten()
        mpl.mplshow()
    plt.clf()

    ####### test ascii files
    f_tab = viscid.load_file(_viscid_root + '/../sample/ath_sample.*.tab')

    for i, grid in enumerate(f_tab.iter_times(":")):
        plt.subplot2grid((2, 2), (0, i))
        mpl.plot(grid['bx'])
        plt.subplot2grid((2, 2), (1, i))
        mpl.plot(grid['by'])
    if args.show:
        mpl.tighten()
        mpl.mplshow()
    plt.clf()

if __name__ == "__main__":
    main()

##
## EOF
##
