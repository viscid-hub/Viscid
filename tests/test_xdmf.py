#!/usr/bin/env python
""" test reading 2d / 3d xdmf files """

from __future__ import print_function
import argparse

import matplotlib.pyplot as plt

from viscid_test_common import sample_dir, next_plot_fname

import viscid
from viscid import vutil
from viscid.plot import mpl

def main():
    parser = argparse.ArgumentParser(description="Test xdmf")
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)

    f2d = viscid.load_file(sample_dir + '/sample_xdmf.py_0.xdmf')
    b2d = viscid.scalar_fields_to_vector([f2d['bx'], f2d['by'], f2d['bz']],
                                         name="b")
    bx2d, by2d, bz2d = b2d.component_fields()  # pylint: disable=W0612

    f3d = viscid.load_file(sample_dir + '/sample_xdmf.3d.xdmf')
    b3d = f3d['b']
    bx, by, bz = b3d.component_fields()  # pylint: disable=W0612

    nrows = 4
    ncols = 2

    plt.subplot2grid((nrows, ncols), (0, 0))
    mpl.plot(bx, "z=0,x=:30", earth=True, show=False)
    plt.subplot2grid((nrows, ncols), (1, 0))
    mpl.plot(bx, "z=0.5f,x=0:2,y=-100.0f:100.0f", earth=True, show=False)
    plt.subplot2grid((nrows, ncols), (2, 0))
    mpl.plot(bx, "z=-1,x=-10.0f:,y=-100.0f:100.0f", earth=True, show=False)
    plt.subplot2grid((nrows, ncols), (3, 0))
    mpl.plot(bx, "x=0.0f,y=-5.0f:5.0f,z=-5.0f:5.0f", earth=True, show=False)

    plt.subplot2grid((nrows, ncols), (0, 1))
    mpl.plot(bx2d, "y=20.0f,z=-100.0f:100.0f", earth=True, show=False)
    plt.subplot2grid((nrows, ncols), (1, 1))
    mpl.plot(bx2d, "x=0f:20:-1,y=0.0f,z=0.0f", show=False)
    plt.subplot2grid((nrows, ncols), (2, 1))
    mpl.plot(bx2d, earth=True, show=False)
    plt.subplot2grid((nrows, ncols), (3, 1))
    mpl.plot(bx2d, "z=0.0f,x=-20.0f:0.0f", show=False)

    mpl.plt.savefig(next_plot_fname(__file__))
    if args.show:
        mpl.mplshow()

if __name__ == "__main__":
    main()

##
## EOF
##
