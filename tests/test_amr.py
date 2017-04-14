#!/usr/bin/env python
"""Kick the tires on the amr machinery"""

from __future__ import print_function
import argparse
import sys
import os

import matplotlib.pyplot as plt

from viscid_test_common import next_plot_fname

import viscid
from viscid import vutil
from viscid.plot import vpyplot as vlt


def run_test(show=False):
    f = viscid.load_file(os.path.join(viscid.sample_dir, "amr.xdmf"))
    plot_kwargs = dict(patchec='y')
    vlt.plot(f['f'], "z=0.0f", **plot_kwargs)

    plt.savefig(next_plot_fname(__file__))
    if show:
        vlt.show()

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
