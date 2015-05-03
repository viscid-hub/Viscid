#!/usr/bin/env python
""" kick the tires on making matplotlib plots """

from __future__ import print_function
import sys
import os
import argparse

import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt

_viscid_root = os.path.realpath(os.path.dirname(__file__) + '/../viscid/')
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)

import viscid
from viscid import logger
from viscid import vutil
from viscid.plot import mpl

dtype = 'float64'

def run_mpl_testA(show=False):
    logger.info("2D cell centered tests")

    x = np.array(np.linspace(-10, 10, 100), dtype=dtype)
    y = np.array(np.linspace(-10, 10, 120), dtype=dtype)
    z = np.array(np.linspace(-1, 1, 2), dtype=dtype)

    fld_s = viscid.empty([z, y, x], center='cell')
    Zcc, Ycc, Xcc = fld_s.get_crds_cc(shaped=True)  # pylint: disable=unused-variable
    fld_s[:, :, :] = ne.evaluate("(sin(Xcc) + cos(Ycc))")

    nrows = 4
    ncols = 1

    plt.subplot2grid((nrows, ncols), (0, 0))
    mpl.plot(fld_s, "y=20.0", show=False, plot_opts="lin_0")
    plt.subplot2grid((nrows, ncols), (1, 0))
    mpl.plot(fld_s, "x=0.0:20.0,y=0.0:5.0", earth=True, show=False,
             plot_opts="x_-10_0,y_0_7")
    plt.subplot2grid((nrows, ncols), (2, 0))
    mpl.plot(fld_s, "y=0.0", show=False, plot_opts="lin_-1_1")
    plt.subplot2grid((nrows, ncols), (3, 0))
    mpl.plot(fld_s, "z=0.0,x=-20.0:0.0", earth=True, show=False, plot_opts="lin_-5_5")

    mpl.tighten()
    if show:
        mpl.mplshow()

def run_mpl_testB(show=False):
    logger.info("3D node centered tests")

    x = np.array(np.linspace(-10, 10, 100), dtype=dtype)
    y = np.array(np.linspace(-10, 10, 120), dtype=dtype)
    z = np.array(np.linspace(-10, 10, 140), dtype=dtype)

    fld_s = viscid.empty([z, y, x], center='node')
    Z, Y, X = fld_s.get_crds_nc(shaped=True) #pylint: disable=W0612
    fld_s[:, :, :] = ne.evaluate("sin(X) + cos(Y) - cos(Z)")
    # print("shape: ", fld_s.data.shape)

    nrows = 4
    ncols = 1

    plt.subplot2grid((nrows, ncols), (0, 0))
    mpl.plot(fld_s, "z=0,x=:30", earth=True, plot_opts="lin_0")
    plt.subplot2grid((nrows, ncols), (1, 0))
    mpl.plot(fld_s, "z=0.75,x=-4:-1,y=-3.0:3.0", earth=True)
    plt.subplot2grid((nrows, ncols), (2, 0))
    mpl.plot(fld_s, "x=-0.5:,y=-3.0:3.0,z=0.0", earth=True)
    plt.subplot2grid((nrows, ncols), (3, 0))
    mpl.plot(fld_s, "x=0.0,y=-5.0:5.0", earth=True, plot_opts="log,g")

    mpl.tighten()
    if show:
        mpl.mplshow()

def main():
    parser = argparse.ArgumentParser(description="Test calc")
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)

    run_mpl_testA(show=args.show)
    run_mpl_testB(show=args.show)


if __name__ == "__main__":
    main()

##
## EOF
##
