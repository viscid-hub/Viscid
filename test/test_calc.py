#!/usr/bin/env python
""" Tests calculator magnitude function on synthetic vector data...
If numexpr or cython are not installed, the test fails
The test also fails if the two results aren't almost exactly equal """

from __future__ import print_function
import sys
import os
from time import time
import logging
import argparse
# import resource

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm

_viscid_root = os.path.realpath(os.path.dirname(__file__) + '/../src/viscid/')
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)

from viscid import vutil
from viscid import field
from viscid import coordinate
from viscid.calculator import calc
from viscid.plot import mpl

def run_mag_test(fld, show=False):
    vx, vy, vz = fld.component_views() #pylint: disable=W0612
    fld_vx, fld_vy, fld_vz = fld.component_fields()

    t0 = time()
    mag_ne = calc.magnitude(fld, preferred="numexpr", only=True)
    t1 = time()
    logging.info("numexpr mag runtime: {0}".format(t1 - t0))
    t0 = time()
    # cython magnitude doesn't work since switch to only 4d arrays
    # mag_cy = calc.magnitude(fld, preferred="cython", only=True)
    # t1 = time()
    # logging.info("cython mag runtime: {0}".format(t1 - t0))

    # diff1 = calc.diff(mag_ne, mag_cy)
    # absdiff1 = calc.abs_val(diff1)
    # if not (absdiff1.data < 1e-14).all():
    #     logging.warn("numexpr result not exactly cython result")
    # logging.info("min/max(numexpr - cython): {0} / {1}".format(
    #              np.min(absdiff1.data), np.max(absdiff1.data)))

    planes = ["z=0", "y=0"]
    nrows = 4
    ncols = len(planes)
    ax = plt.subplot2grid((nrows, ncols), (0, 0))
    ax.axis("equal")

    for ind, p in enumerate(planes):
        plt.subplot2grid((nrows, ncols), (0, ind), sharex=ax, sharey=ax)
        mpl.plot(fld_vx, p, show=False)
        plt.subplot2grid((nrows, ncols), (1, ind), sharex=ax, sharey=ax)
        mpl.plot(fld_vy, p, show=False)
        plt.subplot2grid((nrows, ncols), (2, ind), sharex=ax, sharey=ax)
        mpl.plot(fld_vz, p, show=False)
        plt.subplot2grid((nrows, ncols), (3, ind), sharex=ax, sharey=ax)
        mpl.plot(mag_ne, p, show=False)
        # plt.subplot2grid((nrows, ncols), (4, ind), sharex=ax, sharey=ax)
        # mpl.plot(diff1, p, show=False)

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

    crds = coordinate.wrap_crds("Rectilinear", (('z', z), ('y', y),
                                                ('x', x)))

    logging.info("Testing node centered magnitudes")
    Z, Y, X = crds.get_crds_nc(shaped=True)

    vx = 0.5 * X**2 +       Y    + 0.0 * Z
    vy = 0.0 * X    + 0.5 * Y**2 + 0.0 * Z
    vz = 0.0 * X    + 0.0 * Y    + 0.5 * Z**2

    fld_v = field.VectorField("v", crds, [vx, vy, vz],
                              center="Node", forget_source=True,
                              deep_meta={"force_layout": field.LAYOUT_INTERLACED},
                             )
    run_mag_test(fld_v, show=args.show)

    logging.info("Testing cell centered magnitudes")
    Z, Y, X = crds.get_crds_cc(shaped=True)

    vx = 0.5 * X**2 + Y + 0.0 * Z
    vy = 0.5 * Y**2
    vz = 0.5 * Z**2

    fld_v = field.VectorField("v", crds, [vx, vy, vz],
                              center="Cell", forget_source=True,
                              deep_meta={"force_layout": field.LAYOUT_INTERLACED},
                             )
    run_mag_test(fld_v, show=args.show)

    #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    # print("ne: ", timereps(10, Div1ne, [fld_vx, fld_vy, fld_vz]))
    #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    # print("inline: ", timereps(10, Div1inline, [fld_vx, fld_vy, fld_vz]))
    #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)

if __name__ == "__main__":
    main()

##
## EOF
##
