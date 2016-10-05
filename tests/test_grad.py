#!/usr/bin/env python
""" Tests gradient && curvature """

from __future__ import print_function
import argparse

import numpy as np

from viscid_test_common import next_plot_fname, xfail

import viscid
from viscid import vutil
from viscid.plot import mpl

try:
    import numexpr as ne  # pylint: disable=unused-import,wrong-import-order
except ImportError:
    xfail("Numexpr is not installed")


def main():
    parser = argparse.ArgumentParser(description="Test grad")
    parser.add_argument("--prof", action="store_true")
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)

    b = viscid.make_dipole(l=(-5, -5, -5), h=(5, 5, 5), n=(256, 256, 128),
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

    _ = mpl.plt.figure(figsize=(9, 4.2))

    ax1 = mpl.subplot(231)
    mpl.plot(b2['z=0f'], logscale=True)
    mpl.plot(b2['z=0f'], logscale=True, style='contour', levels=10, colors='grey')
    # mpl.plot2d_quiver(viscid.normalize(b['z=0f']), step=16, pivot='mid')
    ax2 = mpl.subplot(234)
    mpl.plot(b2['y=0f'], logscale=True)
    mpl.plot(b2['y=0f'], logscale=True, style='contour', levels=10, colors='grey')
    mpl.plot2d_quiver(viscid.normalize(b['y=0f'], only='numpy'), step=16, pivot='mid')

    mpl.subplot(232, sharex=ax1, sharey=ax1)
    mpl.plot(1e-4 + viscid.magnitude(grad_b2['z=0f']), logscale=True)
    mpl.plot(1e-4 + viscid.magnitude(grad_b2['z=0f']), logscale=True,
             style='contour', levels=10, colors='grey')
    mpl.plot2d_quiver(viscid.normalize(grad_b2['z=0f']), step=16, pivot='mid')
    mpl.subplot(235, sharex=ax2, sharey=ax2)
    mpl.plot(1e-4 + viscid.magnitude(grad_b2['y=0f']), logscale=True)
    mpl.plot(1e-4 + viscid.magnitude(grad_b2['y=0f']), logscale=True,
             style='contour', levels=10, colors='grey')
    mpl.plot2d_quiver(viscid.normalize(grad_b2['y=0f']), step=16, pivot='mid')

    mpl.subplot(233, sharex=ax1, sharey=ax1)
    mpl.plot(viscid.magnitude(conv['z=0f']), logscale=True)
    mpl.plot(viscid.magnitude(conv['z=0f']), logscale=True,
             style='contour', levels=10, colors='grey')
    mpl.plot2d_quiver(viscid.normalize(conv['z=0f']), step=16, pivot='mid')
    mpl.subplot(236, sharex=ax2, sharey=ax2)
    mpl.plot(viscid.magnitude(conv['y=0f']), logscale=True)
    mpl.plot(viscid.magnitude(conv['y=0f']), logscale=True,
             style='contour', levels=10, colors='grey')
    mpl.plot2d_quiver(viscid.normalize(conv['y=0f']), step=16, pivot='mid')

    mpl.auto_adjust_subplots()

    mpl.plt.savefig(next_plot_fname(__file__))
    if args.show:
        mpl.show()

if __name__ == "__main__":
    main()

##
## EOF
##
