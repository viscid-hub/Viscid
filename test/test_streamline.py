#!/usr/bin/env python
""" test making and plotting streamlines """

from __future__ import print_function
from timeit import default_timer as time
import argparse

import numpy as np

import viscid
from viscid.calculator import streamline
from viscid.plot import mpl


def run_test(_fld, _seeds, show=False, try_mvi=True, **kwargs):
    lines, topo = streamline.streamlines(_fld, _seeds, **kwargs)
    topo_fld = _seeds.wrap_field(topo)
    try:
        if not try_mvi:
            raise ImportError
        from viscid.plot import mvi
        mvi.plot_lines(lines)
        if show:
            mvi.mlab.show()
    except ImportError:
        mpl.plot_streamlines(lines, show=show)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mayavi", action="store_true")
    parser.add_argument("--show", "--plot", action="store_true")
    args = viscid.vutil.common_argparse(parser)
    args = parser.parse_args()

    viscid.logger.info("Testing field lines on 2d field...")
    B = viscid.vlab.get_dipole(twod=True)
    line = viscid.seed.Line((0.0, 0.0, 0.2), (0.0, 0.0, 1.0), 10)
    obound0 = np.array([-4, -4, -4], dtype=B.data.dtype)
    obound1 = np.array([4, 4, 4], dtype=B.data.dtype)
    run_test(B, line, show=args.show, try_mvi=args.mayavi,
             obound0=obound0, obound1=obound1)

    viscid.logger.info("Testing field lines on 3d field...")
    B = viscid.vlab.get_dipole(m=[0.2, 0.3, -0.9])
    sphere = viscid.seed.Sphere((0.0, 0.0, 0.0), 2.0, 20, 10)
    obound0 = np.array([-4, -4, -4], dtype=B.data.dtype)
    obound1 = np.array([4, 4, 4], dtype=B.data.dtype)
    run_test(B, sphere, show=args.show, try_mvi=args.mayavi,
             obound0=obound0, obound1=obound1, method=streamline.RK12)

if __name__ == "__main__":
    main()

##
## EOF
##
