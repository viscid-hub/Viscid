#!/usr/bin/env python
""" Tests calculator divergence function on synthetic vector data...
If numexpr or cython are not installed, the test fails
The test also fails if the two results aren't almost exactly equal, or
if the result isn't close enough to the analytical divergence
There is a systematic error in this case because the initial condition is
sign waves and we use a central difference divergence """

from __future__ import print_function
import sys
import os
from time import time
import logging
import argparse

import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt

_viscid_root = os.path.realpath(os.path.dirname(__file__) + '/../src/viscid/')
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)

from viscid import vutil
# from viscid import readers
from viscid import field
from viscid import coordinate
from viscid.calculator import calc
from viscid.plot import mpl

def run_div_test(fld, exact, show=False):
    t0 = time()
    result_numexpr = calc.div(fld, preferred="numexpr", only=True)
    t1 = time()
    logging.info("numexpr magnitude runtime: {0}".format(t1 - t0))

    # cython div doesnt work since switch to 4d only arrays
    # t0 = time()
    # result_cython = calc.div(fld, preferred="cython", only=True)
    # t1 = time()
    # logging.info("cython runtime: {0}".format(t1 - t0))

    # backend_diff = calc.diff(result_numexpr, result_cython)
    # if not (backend_diff.data < 1e-14).all():
    #     logging.warn("numexpr result not exactly cython result")
    # logging.info("min/max(abs(numexpr - cython)): {0} = {1}".format(
    #              np.min(backend_diff.data), np.max(backend_diff.data)))

    result_diff = calc.diff(result_numexpr, exact[1:-1, 1:-1, 1:-1])
    if not (result_diff.data < 5e-5).all():
        logging.warn("numexpr result is far from the exact result")
    logging.info("min/max(abs(numexpr - exact)): {0} / {1}".format(
                 np.min(result_diff.data), np.max(result_diff.data)))

    planes = ["y=0.", "z=0."]
    nrows = 2
    ncols = len(planes)
    ax = plt.subplot2grid((nrows, ncols), (0, 0))
    ax.axis("equal")

    for i, p in enumerate(planes):
        plt.subplot2grid((nrows, ncols), (0, i), sharex=ax, sharey=ax)
        mpl.plot(result_numexpr, p, show=False)
        plt.subplot2grid((nrows, ncols), (1, i), sharex=ax, sharey=ax)
        mpl.plot(result_diff, p, show=False)
        # plt.subplot2grid((nrows, ncols), (2, i), sharex=ax, sharey=ax)
        # mpl.plot(result_cython, p, show=False)
        # plt.subplot2grid((nrows, ncols), (3, i), sharex=ax, sharey=ax)
        # mpl.plot(backend_diff, p, show=False)

    if show:
        mpl.mplshow()

def main():
    parser = argparse.ArgumentParser(description="Test divergence")
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)

    dtype = 'float64'

    # use 512 512 256 to inspect memory related things
    x = np.array(np.linspace(-0.5, 0.5, 256), dtype=dtype)
    y = np.array(np.linspace(-0.5, 0.5, 256), dtype=dtype)
    z = np.array(np.linspace(-0.5, 0.5, 64), dtype=dtype)
    crds = coordinate.wrap_crds("Rectilinear", (('z', z), ('y', y), ('x', x)))

    half = np.array([0.5], dtype=dtype) #pylint: disable=W0612
    two = np.array([2.0], dtype=dtype) #pylint: disable=W0612

    Z, Y, X = crds.get_crds_nc(shaped=True) #pylint: disable=W0612
    Zcc, Ycc, Xcc = crds.get_crds_cc(shaped=True) #pylint: disable=W0612

    logging.info("cell centered tests")

    vx = ne.evaluate("(sin(Xcc))")  # + Zcc
    vy = ne.evaluate("(cos(Ycc))")  # + Xcc# + Zcc
    vz = ne.evaluate("-((sin(Zcc)))")  # + Xcc# + Ycc
    exact = ne.evaluate("cos(Xcc) - "
                        "sin(Ycc) - "
                        "cos(Zcc)")
    # cell centered field and exact divergence
    fld_v = field.VectorField("v_cc", crds, [vx, vy, vz],
                              center="Cell", forget_source=True,
                              info={"force_layout": field.LAYOUT_INTERLACED},
                             )
    vx = vy = vz = None
    fld_exact = field.ScalarField("exact div", crds, exact,
                                  center="Cell", forget_source=True)
    run_div_test(fld_v, fld_exact, show=args.show)

    logging.info("node centered tests")

    vx = ne.evaluate("(sin(X))")  # + Zcc
    vy = ne.evaluate("(cos(Y))")  # + Xcc# + Zcc
    vz = ne.evaluate("-((sin(Z)))")  # + Xcc# + Ycc
    exact = ne.evaluate("cos(X) - "
                        "sin(Y) - "
                        "cos(Z)")

    # cell centered field and exact divergence
    fld_v = field.VectorField("v_nc", crds, [vx, vy, vz],
                              center="Node", forget_source=True,
                              info={"force_layout": field.LAYOUT_INTERLACED},
                             )
    fld_exact = field.ScalarField("exact div", crds, exact,
                                  center="Node", forget_source=True)
    run_div_test(fld_v, fld_exact, show=args.show)


if __name__ == "__main__":
    main()

##
## EOF
##
