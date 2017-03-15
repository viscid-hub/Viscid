#!/usr/bin/env python
"""Kick the tires on the amr machinery"""

from __future__ import print_function
import argparse
import sys
import os

from viscid_test_common import next_plot_fname

import viscid
from viscid import vutil
from viscid.plot import mpl


def run_test(show=False):
    f = viscid.load_file(os.path.join(viscid.sample_dir, "amr.xdmf"))
    plot_kwargs = dict(patchec='y')
    mpl.plot(f['f'], "z=0.0f", **plot_kwargs)

    mpl.plt.savefig(next_plot_fname(__file__))
    if show:
        mpl.show()

def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)

    run_test(show=args.show)

    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
