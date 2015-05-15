#!/usr/bin/env python
""" Tests calculator magnitude function on synthetic vector data...
If numexpr or cython are not installed, the test fails
The test also fails if the two results aren't almost exactly equal """

from __future__ import print_function
import sys
import os
from time import time
import argparse
# import resource

import numpy as np
import matplotlib.pyplot as plt

_viscid_root = os.path.realpath(os.path.dirname(__file__) + '/../viscid/')
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)

import viscid
from viscid import logger
from viscid import vutil
from viscid.calculator import calc
from viscid.plot import mpl

def run_mag_test(fld, show=False):
    vx, vy, vz = fld.component_views() #pylint: disable=W0612
    vx, vy, vz = fld.component_fields()

    t0 = time()
    mag_ne = calc.magnitude(fld, preferred="numexpr", only=True)
    t1 = time()
    logger.info("numexpr mag runtime: %g", t1 - t0)
    t0 = time()

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

    if show:
        mpl.mplshow()

def main():
    parser = argparse.ArgumentParser(description="Test calc")
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)
    dtype = 'float32'

    x = np.array(np.linspace(-5, 5, 512), dtype=dtype)
    y = np.array(np.linspace(-5, 5, 256), dtype=dtype)
    z = np.array(np.linspace(-5, 5, 256), dtype=dtype)
    v = viscid.empty([z, y, x], name="V", nr_comps=3, center='node',
                         layout='interlaced')
    Z, Y, X = v.get_crds_nc(shaped=True)
    v['x'] = 0.5 * X**2 +       Y    + 0.0 * Z
    v['y'] = 0.0 * X    + 0.5 * Y**2 + 0.0 * Z
    v['z'] = 0.0 * X    + 0.0 * Y    + 0.5 * Z**2

    logger.info("Testing node centered magnitudes")
    run_mag_test(v, show=args.show)

    logger.info("Testing cell centered magnitudes")
    v = v.as_centered('cell')
    run_mag_test(v, show=args.show)

    #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    # print("ne: ", timereps(10, Div1ne, [vx, vy, vz]))
    #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    # print("inline: ", timereps(10, Div1inline, [vx, vy, vz]))
    #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)

if __name__ == "__main__":
    main()

##
## EOF
##
