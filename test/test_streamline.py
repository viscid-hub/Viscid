#!/usr/bin/env python
""" test making and plotting streamlines """

from __future__ import print_function
from timeit import default_timer as time
import argparse
import logging

import numpy as np

from viscid import vutil
from viscid import vlab
from viscid.calculator import calc
from viscid.calculator import cycalc
from viscid.calculator import streamline
from viscid.calculator import seed
from viscid.plot import mpl

def main():
    parser = argparse.ArgumentParser(description="Load some data files")
    parser.add_argument('files', nargs="*", help='input files')
    parser.add_argument("--show", "--plot", action="store_true")
    parser.add_argument("--old", action="store", type=int)
    args = vutil.common_argparse(parser)
    args = parser.parse_args()

    logging.info("Testing field lines on 2d field...")
    B = vlab.get_dipole(twod=True)
    obound0 = np.array([-4, -4, -4], dtype=B.data.dtype)
    obound1 = np.array([4, 4, 4], dtype=B.data.dtype)
    t0 = time()
    lines, topo = streamline.streamlines(B,
                                         seed.Line((0.0, 0.0, 0.2),
                                                   (0.0, 0.0, 1.0),
                                                   10),
                                         ds0=0.01, ibound=0.1, maxit=10000,
                                         obound0=obound0, obound1=obound1,
                                         method=streamline.EULER1,
                                         stream_dir=streamline.DIR_BOTH,
                                         output=streamline.OUTPUT_BOTH,
                                         tol_lo=5e-3, tol_hi=2e-1,
                                         fac_refine=0.75, fac_coarsen=1.5)
    t1 = time()
    logging.info("streamlines took {0:.3e}s to compute.".format(t1 - t0))
    mpl.plot_streamlines(lines, show=args.show)

    logging.info("Testing field lines on 3d field...")
    B = vlab.get_dipole(m=[0.2, 0.3, -0.9])
    t0 = time()
    lines, topo = streamline.streamlines(B,
                                         seed.Sphere((0.0, 0.0, 0.0),
                                                     2.0, 20, 10),
                                         ds0=0.01, ibound=0.1, maxit=10000,
                                         method=streamline.RK12,
                                         output=streamline.OUTPUT_STREAMLINES,
                                         tol_lo=1e-3, tol_hi=1e-2,
                                         fac_refine=0.75, fac_coarsen=2.0)
    t1 = time()
    logging.info("streamlines took {0:.3e}s to compute.".format(t1 - t0))
    mpl.plot_streamlines(lines, show=args.show)

    # assert(0)
    logging.info("Testing trilinear interpolation...")

    bmag = calc.magnitude(B)

    plane = seed.Plane((1., 1., 1.), (1., 1., 1.), (0., 0., 1.), 2., 2., 50, 50)
    t0 = time()
    interp_vals = cycalc.interp_trilin(bmag, plane)
    t1 = time()
    logging.info("plane interp took {0:.3e}s to compute.".format(t1 - t0))
    # interp_vals is now a 1d array of interpolated values
    # interp_vals[i] is located at sphere.points[i]
    mpl.scatter_3d(plane.points(), interp_vals, show=args.show)

    vol = bmag.crds.slice("x=::32,y=::32,z=::32")
    t0 = time()
    interp_vals = cycalc.interp_trilin(bmag, vol)
    t1 = time()
    logging.info("volume interp took {0:.3e}s to compute.".format(t1 - t0))
    # interp_vals is now a 1d array of interpolated values
    # interp_vals[i] is located at sphere.points[i]
    mpl.scatter_3d(vol.points(), interp_vals, show=args.show)

    sphere = seed.Sphere((0.0, 0.0, 0.0), 2.0, 200, 200)

    # doing trilin interp on a scalar field
    t0 = time()
    interp_vals = cycalc.interp_trilin(bmag, sphere)
    t1 = time()
    logging.info("sphere interp took {0:.3e}s to compute.".format(t1 - t0))
    # interp_vals is now a 1d array of interpolated values
    # interp_vals[i] is located at sphere.points[i]
    mpl.scatter_3d(sphere.points(), interp_vals, show=args.show)

    # doing trilin interp on a vector field
    t0 = time()
    interp_vals = cycalc.interp_trilin(B, sphere)
    t1 = time()
    logging.info("vector interp took {0:.3e}s to compute.".format(t1 - t0))
    # make a 3d scatter plot of bz
    mpl.scatter_3d(sphere.points(), interp_vals[:, 2], show=args.show)

    # val = cycalc.interp_trilin(bmag, seed.Point((1.0, 1.0, 1.0)))
    # logging.info("bmag value at point (1, 1, 1) is {0}".format(val))

if __name__ == "__main__":
    main()

##
## EOF
##
