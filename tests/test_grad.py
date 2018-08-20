#!/usr/bin/env python
"""Test gradient and curvature calculation

To verify that the results are accurate, one should compare the results
with the reference plots.
"""

from __future__ import print_function
import argparse
import sys

import numpy as np

from viscid_test_common import next_plot_fname, xfail

import viscid
from viscid import vutil
from viscid.plot import vpyplot as vlt
import matplotlib.pyplot as plt


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prof", action="store_true")
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)

    b = viscid.make_dipole(l=(-5, -5, -5), h=(5, 5, 5), n=(255, 255, 127),
                           m=(0, 0, -1))
    b2 = np.sum(b * b, axis=b.nr_comp)

    if args.prof:
        print("Without boundaries")
        viscid.timeit(viscid.grad, b2, bnd=False, timeit_repeat=10,
                      timeit_print_stats=True)
        print("With boundaries")
        viscid.timeit(viscid.grad, b2, bnd=True, timeit_repeat=10,
                      timeit_print_stats=True)

    grad_b2 = viscid.grad(b2)
    grad_b2.pretty_name = r"$\nabla$ B$^2$"
    conv = viscid.convective_deriv(b)
    conv.pretty_name = r"(B $\cdot \nabla$) B"

    _ = plt.figure(figsize=(9, 4.2))

    ax1 = vlt.subplot(231)
    vlt.plot(b2['z=0j'], logscale=True)
    vlt.plot(b2['z=0j'], logscale=True, style='contour', levels=10, colors='grey')
    # vlt.plot2d_quiver(viscid.normalize(b['z=0j']), step=16, pivot='mid')
    ax2 = vlt.subplot(234)
    vlt.plot(b2['y=0j'], logscale=True)
    vlt.plot(b2['y=0j'], logscale=True, style='contour', levels=10, colors='grey')
    vlt.plot2d_quiver(viscid.normalize(b['y=0j'], preferred='numpy'),
                      step=16, pivot='mid')

    vlt.subplot(232, sharex=ax1, sharey=ax1)
    vlt.plot(1e-4 + viscid.magnitude(grad_b2['z=0j']), logscale=True)
    vlt.plot(1e-4 + viscid.magnitude(grad_b2['z=0j']), logscale=True,
             style='contour', levels=10, colors='grey')
    vlt.plot2d_quiver(viscid.normalize(grad_b2['z=0j']), step=16, pivot='mid')
    vlt.subplot(235, sharex=ax2, sharey=ax2)
    vlt.plot(1e-4 + viscid.magnitude(grad_b2['y=0j']), logscale=True)
    vlt.plot(1e-4 + viscid.magnitude(grad_b2['y=0j']), logscale=True,
             style='contour', levels=10, colors='grey')
    vlt.plot2d_quiver(viscid.normalize(grad_b2['y=0j']), step=16, pivot='mid')

    vlt.subplot(233, sharex=ax1, sharey=ax1)
    vlt.plot(viscid.magnitude(conv['z=0j']), logscale=True)
    vlt.plot(viscid.magnitude(conv['z=0j']), logscale=True,
             style='contour', levels=10, colors='grey')
    vlt.plot2d_quiver(viscid.normalize(conv['z=0j']), step=16, pivot='mid')
    vlt.subplot(236, sharex=ax2, sharey=ax2)
    vlt.plot(viscid.magnitude(conv['y=0j']), logscale=True)
    vlt.plot(viscid.magnitude(conv['y=0j']), logscale=True,
             style='contour', levels=10, colors='grey')
    vlt.plot2d_quiver(viscid.normalize(conv['y=0j']), step=16, pivot='mid')

    vlt.auto_adjust_subplots()

    plt.savefig(next_plot_fname(__file__))
    if args.show:
        vlt.show()

    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
