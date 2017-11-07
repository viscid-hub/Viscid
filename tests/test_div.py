#!/usr/bin/env python
"""Tests divergence function on synthetic vector data

A warning is printed if the results aren't close to the analytic
result, but the test still passes. To verify that the results are
accurate, one should compare the results with the reference plots.

There is a systematic error in this case because the initial condition is
sign waves and we use central differences
"""

from __future__ import print_function
import argparse
import sys
from timeit import default_timer as time

import matplotlib.pyplot as plt
import numpy as np

from viscid_test_common import next_plot_fname

import viscid
from viscid import logger
from viscid import vutil
from viscid.plot import vpyplot as vlt

try:
    import numexpr as ne
    HAS_NUMEXPR = True
except ImportError:
    HAS_NUMEXPR = False


def run_div_test(fld, exact, title='', show=False, ignore_inexact=False):
    t0 = time()
    result_numexpr = viscid.div(fld, preferred="numexpr", only=False)
    t1 = time()
    logger.info("numexpr magnitude runtime: %g", t1 - t0)

    result_diff = viscid.diff(result_numexpr, exact)['x=1:-1, y=1:-1, z=1:-1']
    if not ignore_inexact and not (result_diff.data < 5e-5).all():
        logger.warn("numexpr result is far from the exact result")
    logger.info("min/max(abs(numexpr - exact)): %g / %g",
                np.min(result_diff.data), np.max(result_diff.data))

    planes = ["y=0f", "z=0f"]
    nrows = 2
    ncols = len(planes)
    ax = plt.subplot2grid((nrows, ncols), (0, 0))

    for i, p in enumerate(planes):
        plt.subplot2grid((nrows, ncols), (0, i))
        vlt.plot(result_numexpr, p, show=False)
        plt.subplot2grid((nrows, ncols), (1, i))
        vlt.plot(result_diff, p, show=False)

    plt.suptitle(title)
    vlt.auto_adjust_subplots(subplot_params=dict(top=0.9))

    plt.savefig(next_plot_fname(__file__))
    if show:
        vlt.mplshow()

def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prof", action="store_true")
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

    if HAS_NUMEXPR:
        v['x'] = ne.evaluate("(sin(Xcc))")  # + Zcc
        v['y'] = ne.evaluate("(cos(Ycc))")  # + Xcc# + Zcc
        v['z'] = ne.evaluate("-((sin(Zcc)))")  # + Xcc# + Ycc
        exact_cc[:, :, :] = ne.evaluate("cos(Xcc) - sin(Ycc) - cos(Zcc)")
    else:
        v['x'] = (np.sin(Xcc))  # + Zcc
        v['y'] = (np.cos(Ycc))  # + Xcc# + Zcc
        v['z'] = -((np.sin(Zcc)))  # + Xcc# + Ycc
        exact_cc[:, :, :] = np.cos(Xcc) - np.sin(Ycc) - np.cos(Zcc)

    if args.prof:
        print("Without boundaries")
        viscid.timeit(viscid.div, v, bnd=False, timeit_repeat=10,
                      timeit_print_stats=True)
        print("With boundaries")
        viscid.timeit(viscid.div, v, bnd=True, timeit_repeat=10,
                      timeit_print_stats=True)

    logger.info("node centered tests")
    v_nc = v.as_centered('node')
    exact_nc = viscid.empty_like(v_nc['x'])
    X, Y, Z = exact_nc.get_crds_nc(shaped=True)  # pylint: disable=W0612
    if HAS_NUMEXPR:
        exact_nc[:, :, :] = ne.evaluate("cos(X) - sin(Y) - cos(Z)")
    else:
        exact_nc[:, :, :] = np.cos(X) - np.sin(Y) - np.cos(Z)
    # FIXME: why is the error so much larger here?
    run_div_test(v_nc, exact_nc, title='Node Centered', show=args.show,
                 ignore_inexact=True)

    logger.info("cell centered tests")
    v_cc = v_nc.as_centered('cell')
    run_div_test(v_cc, exact_cc, title="Cell Centered", show=args.show)

    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
