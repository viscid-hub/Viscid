#!/usr/bin/env python
""" test loading a gnuplot styled 1d ascii datafile """
from __future__ import print_function
import sys
import os
import argparse

_viscid_root = os.path.realpath(os.path.dirname(__file__) + '/../viscid/')
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)

from viscid import vutil
from viscid import readers
from viscid.plot import mpl

def main():
    parser = argparse.ArgumentParser(description="Test xdmf")
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)

    f = readers.load_file(_viscid_root + '/../sample/test.asc')
    mpl.plot(f['0'], show=args.show)

if __name__ == "__main__":
    main()

##
## EOF
##
