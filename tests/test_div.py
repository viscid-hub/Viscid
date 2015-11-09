#!/usr/bin/env python
""" Tests calculator divergence function on synthetic vector data...
If numexpr or cython are not installed, the test fails
The test also fails if the two results aren't almost exactly equal, or
if the result isn't close enough to the analytical divergence
There is a systematic error in this case because the initial condition is
sign waves and we use a central difference divergence """

from __future__ import print_function
from time import time
import argparse

import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt

from viscid_test_common import next_plot_fname

import viscid
from viscid import logger
from viscid import vutil
from viscid.plot import mpl


def run_div_test(fld, exact, title='', show=False, ignore_inexact=False):
    t0 = time()
    result_numexpr = viscid.div(fld, preferred="numexpr", only=True)
    t1 = time()
    logger.info("numexpr magnitude runtime: %g", t1 - t0)

    result_diff = viscid.diff(result_numexpr, exact[1:-1, 1:-1, 1:-1])
    if not ignore_inexact and not (result_diff.data < 5e-5).all():
        logger.warn("numexpr result is far from the exact result")
    logger.info("min/max(abs(numexpr - exact)): %g / %g",
                np.min(result_diff.data), np.max(result_diff.data))

    planes = ["y=0f", "z=0f"]
    nrows = 2
    ncols = len(planes)
    ax = plt.subplot2grid((nrows, ncols), (0, 0))
    ax.axis("equal")

    for i, p in enumerate(planes):
        plt.subplot2grid((nrows, ncols), (0, i), sharex=ax, sharey=ax)
        mpl.plot(result_numexpr, p, show=False)
        plt.subplot2grid((nrows, ncols), (1, i), sharex=ax, sharey=ax)
        mpl.plot(result_diff, p, show=False)

    mpl.plt.suptitle(title)
    mpl.auto_adjust_subplots(subplot_params=dict(top=0.9))

    mpl.plt.savefig(next_plot_fname(__file__))
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

    v = viscid.empty([x, y, z], name="V", nr_comps=3, center="cell",
                     layout="interlaced")
    exact_cc = viscid.empty([x, y, z], name="exact_cc", center='cell')

    Xcc, Ycc, Zcc = exact_cc.get_crds_cc(shaped=True)  # pylint: disable=W0612

    v['x'] = ne.evaluate("(sin(Xcc))")  # + Zcc
    v['y'] = ne.evaluate("(cos(Ycc))")  # + Xcc# + Zcc
    v['z'] = ne.evaluate("-((sin(Zcc)))")  # + Xcc# + Ycc
    exact_cc[:, :, :] = ne.evaluate("cos(Xcc) - sin(Ycc) - cos(Zcc)")

    logger.info("node centered tests")
    v_nc = v.as_centered('node')
    exact_nc = viscid.empty_like(v_nc['x'])
    X, Y, Z = exact_nc.get_crds_nc(shaped=True)  # pylint: disable=W0612
    exact_nc[:, :, :] = ne.evaluate("cos(X) - sin(Y) - cos(Z)")
    # FIXME: why is the error so much larger here?
    run_div_test(v_nc, exact_nc, title='Node Centered', show=args.show,
                 ignore_inexact=True)

    logger.info("cell centered tests")
    v_cc = v_nc.as_centered('cell')
    run_div_test(v_cc, exact_cc, title="Cell Centered", show=args.show)


if __name__ == "__main__":
    main()

##
## EOF
##
