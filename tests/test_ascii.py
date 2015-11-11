#!/usr/bin/env python
""" test loading a gnuplot styled 1d ascii datafile """
from __future__ import print_function
import argparse

from viscid_test_common import sample_dir, next_plot_fname

import viscid
from viscid import vutil
from viscid.plot import mpl


def main():
    raise RuntimeError("intentionally break Travis-CI build")
    parser = argparse.ArgumentParser(description="Test xdmf")
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)

    f = viscid.load_file(sample_dir + '/test.asc')
    mpl.plot(f['c1'], show=False)
    mpl.plt.savefig(next_plot_fname(__file__))
    if args.show:
        mpl.show()

if __name__ == "__main__":
    main()

##
## EOF
##
