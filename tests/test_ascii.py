#!/usr/bin/env python
"""Load and plot a 1d ascii datafile (gnuplot format)"""
from __future__ import print_function
import argparse
import sys
import os

from viscid_test_common import next_plot_fname

import viscid
from viscid import vutil
from viscid.plot import vpyplot as vlt
import matplotlib.pyplot as plt


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)

    f = viscid.load_file(os.path.join(viscid.sample_dir, "test.asc"))
    vlt.plot(f['c1'], show=False)
    plt.savefig(next_plot_fname(__file__))
    if args.show:
        vlt.show()

    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
