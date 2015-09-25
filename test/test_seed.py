#!/usr/bin/env python

from __future__ import division, print_function
import argparse
import os
import sys

import numpy as np
import viscid

_viscid_root = os.path.realpath(os.path.dirname(__file__) + '/../viscid/')
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)


def run_test(fld, seeds, plot2d=True, plot3d=True, show=False):
    interpolated_fld = viscid.interp_trilin(fld, seeds)

    try:
        if not plot2d:
            raise ImportError
        from viscid.plot import mpl
        mpl.plt.clf()
        # mpl.plt.plot(seeds.get_points()[2, :], fld)
        mpl.plot(seeds.wrap_field(interpolated_fld))
        if show:
            mpl.plt.show()
    except ImportError:
        pass

    try:
        if not plot3d:
            raise ImportError
        from viscid.plot import mvi
        mvi.clf()
        pts = seeds.get_points()
        p = mvi.mlab.points3d(pts[0], pts[1], pts[2], interpolated_fld,
                              scale_mode='none', scale_factor=0.02)
        mvi.mlab.axes(p)
        if show:
            mvi.show()
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--two", dest='two', action="store_false")
    parser.add_argument("--three", dest='three', action="store_false")
    parser.add_argument("--show", "--plot", action="store_true")
    args = viscid.vutil.common_argparse(parser, default_verb=0)

    plot2d = args.two
    plot3d = args.three

    # plot2d = True
    # plot3d = True
    # args.show = False

    img = np.load(_viscid_root + "/../sample/logo.npy")
    x = np.linspace(-1, 1, img.shape[0])
    y = np.linspace(-1, 1, img.shape[1])
    z = np.linspace(-1, 1, img.shape[2])
    logo = viscid.arrays2field(img, [x, y, z])

    if True:
        viscid.logger.info('Testing Line...')
        seeds = viscid.Line([-1, -1, 0], [1, 1, 2], n=5)
        run_test(logo, seeds, plot2d=plot2d, plot3d=plot3d, show=args.show)

    if True:
        viscid.logger.info('Testing Plane...')
        seeds = viscid.Plane([0.0, 0.0, 0.0], [1, 1, 1], [1, 0, 0], 2, 2,
                             nl=160, nm=160, NL_are_vectors=True)
        run_test(logo, seeds, plot2d=plot2d, plot3d=plot3d, show=args.show)

    if True:
        viscid.logger.info('Testing Volume...')
        seeds = viscid.Volume([-0.8, -0.8, -0.8], [0.8, 0.8, 0.8],
                              n=[64, 64, 3])
        # note: can't make a 2d plot of the volume w/o a slice
        run_test(logo, seeds, plot2d=False, plot3d=plot3d, show=args.show)

    if True:
        viscid.logger.info('Testing Spherical Sphere...')
        seeds = viscid.Sphere([0, 0, 0], r=1.0, ntheta=160, nphi=160,
                              pole=[-1, -1, -1], theta_phi=False)
        run_test(logo, seeds, plot2d=plot2d, plot3d=plot3d, show=args.show)

    if True:
        viscid.logger.info('Testing Spherical Cap...')
        seeds = viscid.SphericalCap([0, 0, 0], r=1.0, ntheta=64, nphi=80,
                                    pole=[-1, -1, -1], theta_phi=False)
        run_test(logo, seeds, plot2d=plot2d, plot3d=plot3d, show=args.show)

    if True:
        viscid.logger.info('Testing Spherical Patch...')
        seeds = viscid.SphericalPatch([0, 0, 0], [0, -0, -1], 30.0, 59.9,
                                      nalpha=65, nbeta=80, r=1.0)
        run_test(logo, seeds, plot2d=plot2d, plot3d=plot3d, show=args.show)

    return 0

if __name__ == "__main__":
    sys.exit(main())

##
## EOF
##
