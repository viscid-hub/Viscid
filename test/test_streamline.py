#!/usr/bin/env python

from __future__ import print_function
import argparse
import logging

import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt

from viscid import vutil
from viscid import field
from viscid import coordinate
from viscid import readers
from viscid.calculator import calc
from viscid.calculator import cycalc
from viscid.calculator import streamline
from viscid.calculator import seed
from viscid.plot import mpl

def get_dipole(m=None, twod=False):
    dtype = 'float32'
    x = np.array(np.linspace(-5, 5, 256), dtype=dtype)
    y = np.array(np.linspace(-5, 5, 256), dtype=dtype)
    z = np.array(np.linspace(-5, 5, 256), dtype=dtype)
    if twod:
        y = np.array(np.linspace(-0.1, 0.1, 2), dtype=dtype)
    crds = coordinate.wrap_crds("Rectilinear", (('z', z), ('y', y), ('x', x)))

    one = np.array([1.0], dtype=dtype) #pylint: disable=W0612
    three = np.array([3.0], dtype=dtype) #pylint: disable=W0612
    if not m:
        m = [0.0, 0.0, -1.0]
    m = np.array(m, dtype=dtype)
    mx, my, mz = m #pylint: disable=W0612

    Zcc, Ycc, Xcc = crds.get_crd(shaped=True, center="Cell") #pylint: disable=W0612

    rsq = ne.evaluate("Xcc**2 + Ycc**2 + Zcc**2") #pylint: disable=W0612
    mdotr = ne.evaluate("mx * Xcc + my * Ycc + mz * Zcc") #pylint: disable=W0612
    Bx = ne.evaluate("((three * Xcc * mdotr / rsq) - mx) / rsq**1.5")
    By = ne.evaluate("((three * Ycc * mdotr / rsq) - my) / rsq**1.5")
    Bz = ne.evaluate("((three * Zcc * mdotr / rsq) - mz) / rsq**1.5")

    fld = field.VectorField("B_cc", crds, [Bx, By, Bz],
                            center="Cell", forget_source=True,
                            info={"force_layout": field.LAYOUT_INTERLACED},
                           )
    # fld_rsq = field.ScalarField("r", crds, hmm,
    #                             center="Cell", forget_source=True)
    return fld  # , fld_rsq

def main():
    parser = argparse.ArgumentParser(description="Load some data files")
    parser.add_argument('files', nargs="*", help='input files')
    parser.add_argument("--show", "--plot", action="store_true")    
    args = vutil.common_argparse(parser)
    args = parser.parse_args()

    logging.info("Testing field lines on 2d field...")
    B = get_dipole(twod=True)
    obound0 = np.array([-10, -10, -10], dtype=B.data.dtype)
    obound1 = np.array([10, 10, 10], dtype=B.data.dtype)
    lines = streamline.streamlines(B,
                                   seed.Line((0.0, 0.0, -1.0),
                                             (0.0, 0.0, 1.0),
                                             20),
                                   ds0=0.01, ibound=0.05, maxit=10000,
                                   obound0=obound0, obound1=obound1)
    mpl.plot_field_lines(lines, show=args.show)

    logging.info("Testing field lines on 3d field...")
    B = get_dipole(m=[0.2, 0.3, -0.9])
    lines = streamline.streamlines(B,
                                   seed.Sphere((0.0, 0.0, 0.0),
                                               2.0, 5, 10),
                                   ds0=0.01, ibound=0.05, maxit=10000)
    mpl.plot_field_lines(lines, show=args.show)

    logging.info("Testing trilinear interpolation...")

    # plane = seed.Plane((1, 1, 1), (1, 1, 1), (0, 0, 1), 2, 2)
    sphere = seed.Sphere((0.0, 0.0, 0.0), 2.0)

    # doing trilin interp on a vector field
    interp_vals = cycalc.trilin_interp(B, sphere)
    # make a 3d scatter plot of bz
    mpl.scatter_3d(sphere.points, interp_vals[:, 2], show=args.show)

    # doing trilin interp on a scalar field
    bmag = calc.magnitude(B)
    interp_vals = cycalc.trilin_interp(bmag, sphere)
    # interp_vals is now a 1d array of interpolated values
    # interp_vals[i] is located at sphere.points[i]
    mpl.scatter_3d(sphere.points, interp_vals, show=args.show)

    val = cycalc.trilin_interp(bmag, seed.Point((1.0, 1.0, 1.0)))
    logging.info("bmag value at point (1, 1, 1) is {0}".format(val))

if __name__ == "__main__":
    main()

##
## EOF
##
