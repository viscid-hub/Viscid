#!/usr/bin/env python
""" test using seeds to calc streamlines / interpolation """

from __future__ import print_function
from timeit import default_timer as time
import argparse

from viscid import logger
from viscid import vutil
from viscid.vlab import get_dipole
from viscid.calculator import calc
from viscid.calculator import cycalc
from viscid.calculator import streamline
from viscid.plot import mpl

def main():
    parser = argparse.ArgumentParser(description="Load some data files")
    parser.add_argument('files', nargs="*", help='input files')
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)
    args = parser.parse_args()

    logger.info("Testing field lines on 3d field...")
    B = get_dipole(l=[-10] * 3, h=[10] * 3, n=[128] * 3, m=[0.0, 0.0, -1.0])

    mygrid = B.crds.slice_keep("z=1.0f:3.0f,y=1:3,x=0.0f", cc=True)

    logger.info("testing streamlines")
    t0 = time()
    lines, _ = streamline.streamlines(B, mygrid, ds0=0.01, ibound=0.05,
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
