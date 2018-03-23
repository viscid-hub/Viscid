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

    ####### test 5-moment uniform grids
    gk_uniform = viscid.load_file(os.path.join(sample_dir,
                                               'sample_gkeyll_uniform_q_*.h5'))

    _, axes = plt.subplots(1, 2, figsize=(9, 3))
    for i, grid in enumerate(gk_uniform.iter_times(":")):
        vlt.plot(grid['rho_i'], logscale=True, style='contourf', levels=128,
                 ax=axes[i])
        seeds = viscid.Line((-1.2, 0, 0), (1.4, 0, 0), 8)
        b_lines, _ = viscid.calc_streamlines(grid['b'], seeds, method='euler1',
                                             max_length=20.0)
        vlt.plot2d_lines(b_lines, scalars='#000000', symdir='z', linewidth=1.0)
        plt.title(grid.format_time('.02f'))
    vlt.auto_adjust_subplots()
    plt.suptitle("Uniform Gkeyll Dataset")

    plt.savefig(next_plot_fname(__file__))
    if args.show:
        plt.show()
    plt.clf()

    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
