#!/usr/bin/env python

from __future__ import division, print_function
import argparse
import sys

from viscid_test_common import sample_dir, next_plot_fname

import numpy as np
import viscid
from viscid.plot import mpl


def run_test(fld, seeds, kind, show=False):
    mpl.plt.clf()
    mpl.plot(viscid.interp(fld, seeds, kind=kind))
    mpl.plt.title(kind)

    mpl.plt.savefig(next_plot_fname(__file__))
    if show:
        mpl.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", "--plot", action="store_true")
    args = viscid.vutil.common_argparse(parser, default_verb=0)

    img = np.load(sample_dir + "/logo.npy")
    x = np.linspace(-1, 1, img.shape[0])
    y = np.linspace(-1, 1, img.shape[1])
    z = np.linspace(-1, 1, img.shape[2])
    logo = viscid.arrays2field(img, [x, y, z])

    seeds = viscid.Volume([-0.8, -0.8, 0.0], [0.8, 0.8, 0.0],
                          n=[64, 64, 1])

    run_test(logo, seeds, "Nearest", show=args.show)
    run_test(logo, seeds, "Trilinear", show=args.show)

    return 0

if __name__ == "__main__":
    sys.exit(main())

##
## EOF
##
