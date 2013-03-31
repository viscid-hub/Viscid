#!/usr/bin/env python
# Tests calculator magnitude function on synthetic vector data...
# If numexpr or cython are not installed, the test fails
# The test also fails if the two results aren't almost exactly equal

from __future__ import print_function
import sys
import os
from time import time
# import resource

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm

_viscid_root = os.path.realpath(os.path.dirname(__file__) + '/../src/viscid/')
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)

from viscid import vutil
from viscid import readers
from viscid import field
from viscid import coordinate
from viscid.calculator import calc
from viscid.plot import mpl

verb = 0

def run_mag_test(fld, show=False):
    vx, vy, vz = fld.component_views() #pylint: disable=W0612
    fld_vx, fld_vy, fld_vz = fld.component_fields()

    t0 = time()
    mag_ne = calc.magnitude(fld, backends="numexpr", only=True)
    t1 = time()
    if verb:
        print("numexpr mag runtime: ", t1 - t0)
    t0 = time()
    mag_cy = calc.magnitude(fld, backends="cython", only=True)
    t1 = time()
    if verb:
        print("cython mag runtime: ", t1 - t0)

    diff1 = calc.difference(mag_ne, mag_cy)
    absdiff1 = calc.abs_val(diff1)
    if not (absdiff1.data < 1e-14).all():
        vutil.warn("numexpr result not exactly cython result")
    if verb:
        print("min/max(numexpr - cython): ", np.min(absdiff1.data), "/",
              np.max(absdiff1.data))

    planes = ["z=0", "y=0"]
    nrows = 5
    ncols = len(planes)
    ax = plt.subplot2grid((nrows, ncols), (0, 0))
    ax.axis("equal")

    for ind, p in enumerate(planes):
        plt.subplot2grid((nrows, ncols), (0, ind), sharex=ax, sharey=ax)
        mpl.plot(fld_vx, p, show=False, verb=verb)
        plt.subplot2grid((nrows, ncols), (1, ind), sharex=ax, sharey=ax)
        mpl.plot(fld_vy, p, show=False, verb=verb)
        plt.subplot2grid((nrows, ncols), (2, ind), sharex=ax, sharey=ax)
        mpl.plot(fld_vz, p, show=False, verb=verb)
        plt.subplot2grid((nrows, ncols), (3, ind), sharex=ax, sharey=ax)
        mpl.plot(mag_ne, p, show=False, verb=verb)
        plt.subplot2grid((nrows, ncols), (4, ind), sharex=ax, sharey=ax)
        mpl.plot(diff1, p, show=False, verb=verb)

    if show:
        mpl.mplshow()

def main():
    dtype = 'float32'
    show = "--plot" in sys.argv or "--show" in sys.argv

    x = np.array(np.linspace(-5, 5, 512), dtype=dtype)
    y = np.array(np.linspace(-5, 5, 256), dtype=dtype)
    z = np.array(np.linspace(-5, 5, 256), dtype=dtype)

    crds = coordinate.wrap_crds("Rectilinear", (('z', z), ('y', y),
                                                ('x', x)))

    if verb:
        print("Testing Node centered magnitudes")
    Z, Y, X = crds.get_nc(shaped=True)

    vx = 0.5 * X**2 + Y
    vy = 0.5 * Y**2
    vz = 0.5 * Z**2

    fld_v = field.VectorField("v", crds, [vx, vy, vz],
                              force_layout=field.LAYOUT_INTERLACED,
                              center="Node", forget_source=True)
    run_mag_test(fld_v, show=show)

    if verb:
        print("Testing Cell centered magnitudes")
    Z, Y, X = crds.get_cc(shaped=True)

    vx = 0.5 * X**2 + Y
    vy = 0.5 * Y**2
    vz = 0.5 * Z**2

    fld_v = field.VectorField("v", crds, [vx, vy, vz],
                              force_layout=field.LAYOUT_INTERLACED,
                              center="Cell", forget_source=True)
    run_mag_test(fld_v, show=show)

    #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    # print("ne: ", timereps(10, Div1ne, [fld_vx, fld_vy, fld_vz]))
    #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    # print("inline: ", timereps(10, Div1inline, [fld_vx, fld_vy, fld_vz]))
    #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)

if __name__ == "__main__":
    if "-v" in sys.argv:
        verb += 1
    main()

##
## EOF
##
