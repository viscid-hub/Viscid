#!/usr/bin/env python
""" test using seeds to calc streamlines / interpolation """

from __future__ import print_function
from timeit import default_timer as time
import argparse

import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt

from viscid import logger
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
    n = 128
    x = np.array(np.linspace(-10, 10, n), dtype=dtype)
    y = np.array(np.linspace(-10, 10, n), dtype=dtype)
    z = np.array(np.linspace(-10, 10, n), dtype=dtype)
    if twod:
        y = np.array(np.linspace(-0.1, 0.1, 2), dtype=dtype)
    crds = coordinate.wrap_crds("nonuniform_cartesian", (('z', z), ('y', y), ('x', x)))

    one = np.array([1.0], dtype=dtype) #pylint: disable=W0612
    three = np.array([3.0], dtype=dtype) #pylint: disable=W0612
    if not m:
        m = [0.0, 0.0, -1.0]
    m = np.array(m, dtype=dtype)
    mx, my, mz = m #pylint: disable=W0612

    Zcc, Ycc, Xcc = crds.get_crds_cc(shaped=True) #pylint: disable=W0612

    rsq = ne.evaluate("Xcc**2 + Ycc**2 + Zcc**2") #pylint: disable=W0612
    mdotr = ne.evaluate("mx * Xcc + my * Ycc + mz * Zcc") #pylint: disable=W0612
    Bx = ne.evaluate("((three * Xcc * mdotr / rsq) - mx) / rsq**1.5")
    By = ne.evaluate("((three * Ycc * mdotr / rsq) - my) / rsq**1.5")
    Bz = ne.evaluate("((three * Zcc * mdotr / rsq) - mz) / rsq**1.5")

    fld = field.VectorField("B_cc", crds, [Bx, By, Bz],
                            center="Cell", forget_source=True,
                            _force_layout=field.LAYOUT_INTERLACED,
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

    logger.info("Testing field lines on 3d field...")
    B = get_dipole(m=[0.0, 0.0, -1.0])

    mygrid = B.crds.slice_keep("z=1.0:3.0,y=1:3,x=0.0", cc=True)

    # print(B.crds.get_crd(center="Node"))
    # print(mygrid.get_crd(center="Cell"))
    # for z, y, x in mygrid.iter_points(center="Cell"):
    #     print(z, y, x)
    # print("======")
    # for z, y, x in mygrid.points(center="Cell"):
    #     print(z, y, x)
    # print("======")
    # print(mygrid.points(center="Cell")[:][0])

    logger.info("testing streamlines")
    t0 = time()
    lines, topo = streamline.streamlines(B, mygrid, ds0=0.01, ibound=0.05,
                                         maxit=10000)
    t1 = time()
    logger.info("streamlines took {0:.3e}s to compute.".format(t1 - t0))
    mpl.plot_streamlines(lines, show=args.show)

    logger.info("testing interp")
    bmag = calc.magnitude(B)
    t0 = time()
    interp_vals = cycalc.interp_trilin(bmag, mygrid)
    t1 = time()
    logger.info("interp took {0:.3e}s to compute.".format(t1 - t0))
    # interp_vals is now a 1d array of interpolated values
    # interp_vals[i] is located at sphere.points[i]
    mpl.scatter_3d(mygrid.points(center="Cell"), interp_vals, show=args.show)

if __name__ == "__main__":
    main()

##
## EOF
##
