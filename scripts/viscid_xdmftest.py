#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import code

# import pylab as pl

_viscid_root = os.path.realpath(os.path.dirname(__file__) + '/../src/viscid/')
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)

from viscid import vlab
# from viscid.vfile_bucket import load_vfile

if __name__ == '__main__':
    #f = dp.load(_viscid_root + '../sample/local_0001.py_0.xdmf')
    f1name = _viscid_root + "/../../sample/local_0001.py_0.004200.xdmf"
    f1 = vlab.load_vfile(f1name)
    f2name = _viscid_root + "/../../sample/local_0001.py_0.004260.xdmf"
    f2 = vlab.load_vfile(f2name)
    f3 = vlab.load_vfile(_viscid_root + "/../../sample/local_0001.py_0.xdmf")
    f4 = vlab.load_vfile(_viscid_root + "/../../sample/amr.xdmf")
    print("f1 = ", f1)
    print("f2 = ", f2)
    print("f3 = ", f3)
    code.interact("===", local=locals())

##
## EOF
##
