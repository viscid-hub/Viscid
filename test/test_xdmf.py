#!/usr/bin/env python
# Try to use matplot lib to plot a bunch of 2d and 1d slices of 3d
# xdmf sample data.
# This is really a test of the slicing routines

from __future__ import print_function
import sys
import os

import pylab as pl

_viscid_root = os.path.realpath(os.path.dirname(__file__) + '/../src/viscid/')
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)

from viscid import readers
from viscid import field
from viscid.plot import mpl

verb = 0

def main():
    show = "--plot" in sys.argv or "--show" in sys.argv

    f2d = readers.load(_viscid_root + '/../../sample/sample.py_0.xdmf')
    b2d = field.scalar_fields_to_vector("b", [f2d['bx'], f2d['by'], f2d['bz']])
    bx2d, by2d, bz2d = b2d.component_fields() #pylint: disable=W0612

    f3d = readers.load(_viscid_root + '/../../sample/sample.3df.xdmf')
    b3d = f3d['b']
    bx, by, bz = b3d.component_fields() #pylint: disable=W0612

    nrows = 4
    ncols = 2


    pl.subplot2grid((nrows, ncols), (0, 0))
    mpl.plot(bx, "z=0i,x=:30i", earth=True, show=False)
    pl.subplot2grid((nrows, ncols), (1, 0))
    mpl.plot(bx, "z=0.5,x=0i:2i,y=-100:100", earth=True, show=False)
    pl.subplot2grid((nrows, ncols), (2, 0))
    mpl.plot(bx, "z=-1i,x=-10:,y=-100:100", earth=True, show=False)
    pl.subplot2grid((nrows, ncols), (3, 0))
    mpl.plot(bx, "x=0,y=-5:5,z=-5:5", earth=True, show=False)

    pl.subplot2grid((nrows, ncols), (0, 1))
    mpl.plot(bx2d, "y=20,z=-100:100", earth=True, show=False)
    pl.subplot2grid((nrows, ncols), (1, 1))
    mpl.plot(bx2d, "x=0i:20i,y=0,z=0", show=False)
    pl.subplot2grid((nrows, ncols), (2, 1))
    mpl.plot(bx2d, earth=True, show=False)
    pl.subplot2grid((nrows, ncols), (3, 1))
    mpl.plot(bx2d, "z=0,x=-20:0", show=False)

    if show:
        mpl.mplshow()

if __name__ == "__main__":
    if "-v" in sys.argv:
        verb += 1
    main()

##
## EOF
##
