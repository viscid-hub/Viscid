#!/usr/bin/env python
""" kick the tires on the amr machinery """

from __future__ import print_function
import sys
import os
import argparse

_viscid_root = os.path.realpath(os.path.dirname(__file__) + '/../viscid/')
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)

import viscid
from viscid import vutil
from viscid.plot import mpl

def run_test(show=False):
    f = viscid.load_file(_viscid_root + "/../sample/amr.xdmf")
    plot_kwargs = dict(show=show)
    mpl.plot(f['f'], "z=0.0", **plot_kwargs)

def main():
    parser = argparse.ArgumentParser(description="Test calc")
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)

    run_test(show=args.show)

if __name__ == "__main__":
    main()

##
## EOF
##
