#!/usr/bin/env python
from __future__ import print_function
import sys
import os

_viscid_root = os.path.realpath(os.path.dirname(__file__)+'/../src/viscid/')
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)

import pylab as pl

from viscid import readers
from viscid import field
from viscid.plot import mpl

if __name__ == '__main__':
    f2d = readers.load(_viscid_root + '/../../sample/sample.py_0.xdmf')
    b2d = field.scalar_fields_to_vector("b", [f2d['bx'], f2d['by'], f2d['bz']])
    rr2d = f2d["rr"]

    f3d = readers.load(_viscid_root + '/../../sample/sample.3df.xdmf')
    b3d = f3d['b']
    rr3d = f3d["rr"]

    # mpl.plot(b2d, show=True)
    # mpl.plot(rr3d, "y=0", show=True)

    # bx, by, bz = b3d.component_fields()
    # pl.subplot2grid((4, 1), (0, 0))
    # mpl.plot(bx, "z=0i,x=:30i", show=False)
    # pl.subplot2grid((4, 1), (1, 0))
    # mpl.plot(bx, "z=0,x=0i:2i", show=False)
    # pl.subplot2grid((4, 1), (2, 0))
    # mpl.plot(bx, "z=-1i,x=-10:", show=False)
    # pl.subplot2grid((4, 1), (3, 0))
    # mpl.plot(bx, "z=0,x=-20:-4.5", show=True)

    pl.subplot2grid((4, 1), (0, 0))
    mpl.plot(f2d['bx'], "y=20", show=False)
    pl.subplot2grid((4, 1), (1, 0))
    mpl.plot(f2d['bz'], "x=0i:20i,y=0,z=0", show=False)
    pl.subplot2grid((4, 1), (2, 0))
    mpl.plot(f2d['bz'], show=False)
    pl.subplot2grid((4, 1), (3, 0))
    mpl.plot(f2d['bx'], "z=0,x=-20:0", show=False)

    pl.show()

##
## EOF
##
