#!/usr/bin/env python
# Tests calculator divergence function on synthetic vector data...
# If numexpr or cython are not installed, the test fails
# The test also fails if the two results aren't almost exactly equal, or
# if the result isn't close enough to the analytical divergence
# There is a systematic error in this case because the initial condition is
# sign waves and we use a central difference divergence

from __future__ import print_function
import sys
import os

import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt

_viscid_root = os.path.realpath(os.path.dirname(__file__) + '/../src/viscid/')
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)

from viscid import field
from viscid import coordinate
from viscid.plot import mpl

dtype = 'float64'
verb = 0

def run_mpl_testA(show=False):
    if verb:
        print("2D Cell centered tests")

    x = np.array(np.linspace(-10, 10, 100), dtype=dtype)
    y = np.array(np.linspace(-10, 10, 120), dtype=dtype)
    z = np.array(np.linspace(-1, 1, 2), dtype=dtype)
    crds = coordinate.wrap_crds("Rectilinear", (('z', z), ('y', y), ('x', x)))
    Zcc, Ycc, Xcc = crds.get_cc(shaped=True) #pylint: disable=W0612

    s = ne.evaluate("(sin(Xcc) + cos(Ycc))")
    fld_s = field.ScalarField("s", crds, s, center="Cell", forget_source=True)

    # print("shape: ", fld_s.data.shape)

    nrows = 4
    ncols = 1

    plt.subplot2grid((nrows, ncols), (0, 0))
    mpl.plot(fld_s, "y=20", show=False)
    plt.subplot2grid((nrows, ncols), (1, 0))
    mpl.plot(fld_s, "x=0i:20i,y=0:5", earth=True, show=False)
    plt.subplot2grid((nrows, ncols), (2, 0))
    mpl.plot(fld_s, "y=0", show=False)
    plt.subplot2grid((nrows, ncols), (3, 0))
    mpl.plot(fld_s, "z=0,x=-20:0", earth=True, show=False)

    mpl.tighten()
    if show:
        mpl.mplshow()

def run_mpl_testB(show=False):
    if verb:
        print("3D Node centered tests")

    x = np.array(np.linspace(-10, 10, 100), dtype=dtype)
    y = np.array(np.linspace(-10, 10, 120), dtype=dtype)
    z = np.array(np.linspace(-10, 10, 140), dtype=dtype)
    crds = coordinate.wrap_crds("Rectilinear", (('z', z), ('y', y), ('x', x)))
    Z, Y, X = crds.get_nc(shaped=True) #pylint: disable=W0612

    s = ne.evaluate("(sin(X) + cos(Y) - cos(Z))")
    fld_s = field.ScalarField("s", crds, s, center="Node", forget_source=True)

    # print("shape: ", fld_s.data.shape)

    nrows = 4
    ncols = 1

    plt.subplot2grid((nrows, ncols), (0, 0))
    mpl.plot(fld_s, "z=0i,x=:30i", earth=True, verb=verb)
    plt.subplot2grid((nrows, ncols), (1, 0))
    mpl.plot(fld_s, "z=0.75,x=-4i:-1i,y=-3:3", earth=True, verb=verb)
    plt.subplot2grid((nrows, ncols), (2, 0))
    mpl.plot(fld_s, "x=-0.5:,y=-3:3,z=0", earth=True, verb=verb)
    plt.subplot2grid((nrows, ncols), (3, 0))
    mpl.plot(fld_s, "x=0,y=-5:5", earth=True, verb=verb)

    mpl.tighten()
    if show:
        mpl.mplshow()

def main():

    show = "--plot" in sys.argv or "--show" in sys.argv

    run_mpl_testA(show=show)
    run_mpl_testB(show=show)


if __name__ == "__main__":
    if "-v" in sys.argv:
        verb += 1
    main()

##
## EOF
##
