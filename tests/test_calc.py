#!/usr/bin/env python
"""Test calculator magnitude function on synthetic vector data"""

from __future__ import print_function
import argparse
import sys
from timeit import default_timer as time

import numpy as np
import matplotlib.pyplot as plt

from viscid_test_common import next_plot_fname, xfail

import viscid
from viscid import logger
from viscid import vutil
from viscid.plot import mpl


def run_mag_test(fld, title="", show=False):
    vx, vy, vz = fld.component_views()  # pylint: disable=W0612
    vx, vy, vz = fld.component_fields()

    try:
        t0 = time()
        mag_ne = viscid.magnitude(fld, preferred="numexpr", only=False)
        t1 = time()
        logger.info("numexpr mag runtime: %g", t1 - t0)
    except viscid.verror.BackendNotFound:
        xfail("Numexpr is not installed")

    planes = ["z=0", "y=0"]
    nrows = 4
    ncols = len(planes)
    ax = plt.subplot2grid((nrows, ncols), (0, 0))
    ax.axis("equal")

    for ind, p in enumerate(planes):
        plt.subplot2grid((nrows, ncols), (0, ind), sharex=ax, sharey=ax)
        mpl.plot(vx, p, show=False)
        plt.subplot2grid((nrows, ncols), (1, ind), sharex=ax, sharey=ax)
        mpl.plot(vy, p, show=False)
        plt.subplot2grid((nrows, ncols), (2, ind), sharex=ax, sharey=ax)
        mpl.plot(vz, p, show=False)
        plt.subplot2grid((nrows, ncols), (3, ind), sharex=ax, sharey=ax)
        mpl.plot(mag_ne, p, show=False)

    mpl.plt.suptitle(title)
    mpl.auto_adjust_subplots(subplot_params=dict(top=0.9))
    mpl.plt.gcf().set_size_inches(6, 7)

    mpl.plt.savefig(next_plot_fname(__file__))
    if show:
        mpl.mplshow()

def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)
    dtype = 'float32'

    x = np.array(np.linspace(-5, 5, 512), dtype=dtype)
    y = np.array(np.linspace(-5, 5, 256), dtype=dtype)
    z = np.array(np.linspace(-5, 5, 256), dtype=dtype)
    v = viscid.empty([x, y, z], name="V", nr_comps=3, center='node',
                     layout='interlaced')
    X, Y, Z = v.get_crds_nc(shaped=True)
    v['x'] = (0.5 * X**2) + (      Y   ) + (0.0 * Z   )
    v['y'] = (0.0 * X   ) + (0.5 * Y**2) + (0.0 * Z   )
    v['z'] = (0.0 * X   ) + (0.0 * Y   ) + (0.5 * Z**2)

    logger.info("Testing node centered magnitudes")
    run_mag_test(v, title="node centered", show=args.show)

    logger.info("Testing cell centered magnitudes")
    v = v.as_centered('cell')
    run_mag_test(v, title="cell centered", show=args.show)

    # print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    # print("ne: ", timereps(10, Div1ne, [vx, vy, vz]))
    # print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    # print("inline: ", timereps(10, Div1inline, [vx, vy, vz]))
    # print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)

    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
