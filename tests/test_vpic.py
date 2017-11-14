#!/usr/bin/env python
"""Test the Athena bin and tab readers"""

from __future__ import print_function
import argparse
import sys
import os

from viscid_test_common import next_plot_fname

import viscid
from viscid import sample_dir
from viscid import vutil
from viscid.plot import vpyplot as vlt

import matplotlib.pyplot as plt


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)

    f = viscid.load_file(os.path.join(sample_dir, 'vpic_sample', 'global.vpc'))

    _, axes = vlt.subplots(2, 2, figsize=(8, 4))

    for i, ti in enumerate([0, -1]):
        f.activate_time(ti)
        vlt.plot(f['n_e']['y=0f'], symmetric=False, ax=axes[0, i])
        vlt.plot(f['bx']['y=0f'], symmetric=True, ax=axes[1, i])
        axes[0, i].set_title(f.get_grid().time)

    vlt.auto_adjust_subplots()

    plt.savefig(next_plot_fname(__file__))
    if args.show:
        vlt.show()
    plt.close()

    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
