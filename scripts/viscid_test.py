#!/usr/bin/env python
from __future__ import print_function
import sys
import os

import pylab as pl

_viscid_root = os.path.realpath(os.path.dirname(__file__)+'/../src/viscid/')
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)

from viscid import data_pool

class asdf(object):
    def __init__(self):
        pass

if __name__=='__main__':
    dp = data_pool.get_data_pool()
    #f = dp.load(_viscid_root + '../sample/local_0001.py_0.xdmf')
    f = dp.load(_viscid_root + '/../../sample/test.asc')
    def a(asdf):
        return None
    print(f)

    print(f.grids)
    g = f.get_grid(0)

    print(g.get_coords())
    print(g["y0"])

    pl.plot(g.get_coords()[0], g["y0"])
    pl.show()

##
## EOF
##

