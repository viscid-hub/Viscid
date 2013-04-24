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
from time import time

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

verb = 0

def run_div_test(fld, exact, show=False):
    t0 = time()
    result_numexpr = calc.div(fld, backends="numexpr", only=True)
    t1 = time()
    if verb:
        print("numexpr magnitude runtime: ", t1 - t0)

    t0 = time()
    result_cython = calc.div(fld, backends="cython", only=True)
    t1 = time()
    if verb:
        print("cython runtime: ", t1 - t0)

    backend_diff = calc.difference(result_numexpr, result_cython)
    if not (backend_diff.data < 1e-14).all():
        vutil.warn("numexpr result not exactly cython result")
    if verb:
        print("min/max(abs(numexpr - cython)): ", np.min(backend_diff.data),
              "/", np.max(backend_diff.data))

    result_diff = calc.difference(result_numexpr, exact, slb=[np.s_[1:-1]]*3)
    if not (result_diff.data < 5e-6).all():
        vutil.warn("numexpr result is far from the exact result")
    if verb:
        print("min/max(abs(numexpr - exact)): ", np.min(result_diff.data),
              "/", np.max(result_diff.data))

    planes = ["y=0.", "z=0."]
    nrows = 4
    ncols = len(planes)
    ax = plt.subplot2grid((nrows, ncols), (0, 0))
    ax.axis("equal")

    for i, p in enumerate(planes):
        plt.subplot2grid((nrows, ncols), (0, i), sharex=ax, sharey=ax)
        mpl.plot(result_numexpr, p, show=False, verb=verb)
        plt.subplot2grid((nrows, ncols), (1, i), sharex=ax, sharey=ax)
        mpl.plot(result_cython, p, show=False, verb=verb)
        plt.subplot2grid((nrows, ncols), (2, i), sharex=ax, sharey=ax)
        mpl.plot(backend_diff, p, show=False, verb=verb)
        plt.subplot2grid((nrows, ncols), (3, i), sharex=ax, sharey=ax)
        mpl.plot(result_diff, p, show=False, verb=verb)

    if show:
        mpl.mplshow()

def main():
    dtype = 'float64'

    show = "--plot" in sys.argv or "--show" in sys.argv

    x = np.array(np.linspace(-0.5, 0.5, 512), dtype=dtype)
    y = np.array(np.linspace(-0.5, 0.5, 512), dtype=dtype)
    z = np.array(np.linspace(-0.5, 0.5, 256), dtype=dtype)
    crds = coordinate.wrap_crds("Rectilinear", (('z', z), ('y', y), ('x', x)))

    half = np.array([0.5], dtype=dtype) #pylint: disable=W0612
    two = np.array([2.0], dtype=dtype) #pylint: disable=W0612

    Z, Y, X = crds.get_crd(shaped=True)
    Zcc, Ycc, Xcc = crds.get_crd(shaped=True, center="Cell")

    if verb:
        print("Cell centered tests")

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
    fld_exact = field.ScalarField("exact div", crds, exact,
                                  center="Cell", forget_source=True)
    run_div_test(fld_v, fld_exact, show=show)

    if verb:
        print("Node centered tests")

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
    run_div_test(fld_v, fld_exact, show=show)


if __name__ == "__main__":
    if "-v" in sys.argv:
        verb += 1
    main()

##
## EOF
##
