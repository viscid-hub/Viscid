#!/usr/bin/env python

from __future__ import print_function
import argparse

import numpy as np
import numexpr as ne
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from viscid import field
from viscid import coordinate
from viscid import readers
from viscid.calculator import cycalc
from viscid.plot import mpl

verb = 0

def get_dipole(m=[0.0, 0.0, 1.0]):
    dtype = 'float32'
    x = np.array(np.linspace(0.01, 10, 128), dtype=dtype)
    y = np.array(np.linspace(-10, 10, 256), dtype=dtype)
    z = np.array(np.linspace(-10, 10, 256), dtype=dtype)
    crds = coordinate.wrap_crds("Rectilinear", (('z', z), ('y', y), ('x', x)))

    one = np.array([1.0], dtype=dtype) #pylint: disable=W0612
    three = np.array([3.0], dtype=dtype) #pylint: disable=W0612
    m = np.array(m, dtype=dtype)
    mx, my, mz = m

    Zcc, Ycc, Xcc = crds.get_cc(shaped=True) #pylint: disable=W0612

    rsq = ne.evaluate("Xcc**2 + Ycc**2 + Zcc**2")
    Bx = ne.evaluate("(three * Xcc * (mx * Xcc + my * Ycc + mz * Zcc) - (mx / rsq)) / rsq**1.5")
    By = ne.evaluate("(three * Ycc * (mx * Xcc + my * Ycc + mz * Zcc) - (my / rsq)) / rsq**1.5")
    Bz = ne.evaluate("(three * Zcc * (mx * Xcc + my * Ycc + mz * Zcc) - (mz / rsq)) / rsq**1.5")
    # hmm = ne.evaluate("one / sqrt(rsq)")

    fld = field.VectorField("B_cc", crds, [Bx, By, Bz],
                            force_layout=field.LAYOUT_INTERLACED,
                            center="Cell", forget_source=True)
    # fld_rsq = field.ScalarField("r", crds, hmm,
    #                             center="Cell", forget_source=True)
    return fld  # , fld_rsq

def main():
    parser = argparse.ArgumentParser(description="Load some data files")
    parser.add_argument('--show', '-s', action='store_true',
                        help='increase verbosity')
    parser.add_argument('-v', action='count', default=0,
                        help='increase verbosity')
    parser.add_argument('-q', action='count', default=0,
                        help='decrease verbosity')
    parser.add_argument('files', nargs="*", help='input files')
    args = parser.parse_args()
    verb = args.v - args.q

    B = get_dipole()
    bx, by, bz = B.component_fields()

    # nrows = 3
    # ncols = 2

    # pl.subplot2grid((nrows, ncols), (0, 0))
    # mpl.plot(bx, "x=0", show=False)
    # pl.subplot2grid((nrows, ncols), (1, 0))
    # mpl.plot(by, "x=0", show=False)
    # pl.subplot2grid((nrows, ncols), (2, 0))
    # mpl.plot(bz, "x=0", show=False)

    # pl.subplot2grid((nrows, ncols), (0, 1))
    # mpl.plot(bx, "z=0", show=False)
    # pl.subplot2grid((nrows, ncols), (1, 1))
    # mpl.plot(by, "z=0", show=False)
    # pl.subplot2grid((nrows, ncols), (2, 1))
    # mpl.plot(bz, "z=0", show=False)
    # pl.show()

    lines = cycalc.streamlines(B, [[1.0, -5.0, 5.0],
                                   [1.0, 0.0, 5.0],
                                   [1.0, 5.0, 5.0],
                                   [0.0, -5.0, 5.0],
                                   [0.0, 0.0, 5.0],
                                   [0.0, 5.0, 5.0],
                                   [-1.0, -5.0, 5.0],
                                   [-1.0, 0.0, 5.0],
                                   [-1.0, 5.0, 5.0],
                                  ], 0.05)

    fig = pl.figure()
    ax = fig.gca(projection='3d')
    for line in lines:
        line = np.array(line)
        z = line[:, 0]
        y = line[:, 1]
        x = line[:, 2]
        ax.plot(x, y, z)
    pl.xlabel("x")
    pl.ylabel("y")
    pl.show()


if __name__ == "__main__":
    main()

##
## EOF
##
