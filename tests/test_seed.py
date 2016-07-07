#!/usr/bin/env python

from __future__ import division, print_function
import argparse
import sys

from viscid_test_common import sample_dir, next_plot_fname

import numpy as np
import viscid


def run_test(fld, seeds, plot2d=True, plot3d=True, add_title="",
             view_kwargs=None, show=False):
    interpolated_fld = viscid.interp_trilin(fld, seeds)
    seed_name = seeds.__class__.__name__
    if add_title:
        seed_name += " " + add_title

    try:
        if not plot2d:
            raise ImportError
        from viscid.plot import mpl
        mpl.plt.clf()
        # mpl.plt.plot(seeds.get_points()[2, :], fld)
        mpl.plot(interpolated_fld)
        mpl.plt.title(seed_name)

        mpl.plt.savefig(next_plot_fname(__file__, series='2d'))
        if show:
            mpl.plt.show()
    except ImportError:
        pass

    try:
        if not plot3d:
            raise ImportError
        from viscid.plot import mvi
        mvi.clf()

        try:
            mesh = mvi.mesh_from_seeds(seeds, scalars=interpolated_fld)
            mesh.actor.property.backface_culling = True
        except RuntimeError:
            pass

        pts = seeds.get_points()
        p = mvi.points3d(pts[0], pts[1], pts[2], interpolated_fld.flat_data,
                         scale_mode='none', scale_factor=0.02)
        mvi.axes(p)
        mvi.title(seed_name)
        if view_kwargs:
            mvi.view(**view_kwargs)

        mvi.savefig(next_plot_fname(__file__, series='3d'))
        if show:
            mvi.show()
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notwo", dest='notwo', action="store_true")
    parser.add_argument("--nothree", dest='nothree', action="store_true")
    parser.add_argument("--show", "--plot", action="store_true")
    args = viscid.vutil.common_argparse(parser, default_verb=0)

    plot2d = not args.notwo
    plot3d = not args.nothree

    # plot2d = True
    # plot3d = True
    # args.show = True

    img = np.load(sample_dir + "/logo.npy")
    x = np.linspace(-1, 1, img.shape[0])
    y = np.linspace(-1, 1, img.shape[1])
    z = np.linspace(-1, 1, img.shape[2])
    logo = viscid.arrays2field(img, [x, y, z])

    if 1:
        viscid.logger.info('Testing Line...')
        seeds = viscid.Line([-1, -1, 0], [1, 1, 2], n=5)
        run_test(logo, seeds, plot2d=plot2d, plot3d=plot3d, show=args.show)

    if 1:
        viscid.logger.info('Testing Plane...')
        seeds = viscid.Plane([0.0, 0.0, 0.0], [1, 1, 1], [1, 0, 0], 2, 2,
                             nl=160, nm=170, NL_are_vectors=True)
        run_test(logo, seeds, plot2d=plot2d, plot3d=plot3d, show=args.show)

    if 1:
        viscid.logger.info('Testing Volume...')
        seeds = viscid.Volume([-0.8, -0.8, -0.8], [0.8, 0.8, 0.8],
                              n=[64, 64, 3])
        # note: can't make a 2d plot of the volume w/o a slice
        run_test(logo, seeds, plot2d=False, plot3d=plot3d, add_title="3d",
                 show=args.show)

    if 1:
        viscid.logger.info('Testing Volume (with ignorable dim)...')
        seeds = viscid.Volume([-0.8, -0.8, 0.0], [0.8, 0.8, 0.0],
                              n=[64, 64, 1])
        run_test(logo, seeds, plot2d=plot2d, plot3d=plot3d, add_title="2d",
                 show=args.show)

    if 1:
        viscid.logger.info('Testing Spherical Sphere (phi, theta)...')
        seeds = viscid.Sphere([0, 0, 0], r=1.0, ntheta=160, nphi=170,
                              pole=[-1, -1, -1], theta_phi=False)
        run_test(logo, seeds, plot2d=plot2d, plot3d=plot3d, add_title="PT",
                 show=args.show)

    if 1:
        viscid.logger.info('Testing Spherical Sphere (theta, phi)...')
        seeds = viscid.Sphere([0, 0, 0], r=1.0, ntheta=160, nphi=170,
                              pole=[-1, -1, -1], theta_phi=True)
        run_test(logo, seeds, plot2d=plot2d, plot3d=plot3d, add_title="TP",
                 show=args.show)

    if 1:
        viscid.logger.info('Testing Spherical Cap (phi, theta)...')
        seeds = viscid.SphericalCap(p0=[0, 0, 0], r=1.0, ntheta=64, nphi=80,
                                    pole=[-1, -1, -1], theta_phi=False)
        run_test(logo, seeds, plot2d=plot2d, plot3d=plot3d, add_title="PT",
                 view_kwargs=dict(azimuth=180, elevation=180), show=args.show)

    if 1:
        viscid.logger.info('Testing Spherical Cap (theta, phi)...')
        seeds = viscid.SphericalCap(p0=[0, 0, 0], r=1.0, ntheta=64, nphi=80,
                                    pole=[-1, -1, -1], theta_phi=True)
        run_test(logo, seeds, plot2d=plot2d, plot3d=plot3d, add_title="TP",
                 view_kwargs=dict(azimuth=180, elevation=180), show=args.show)

    if 1:
        viscid.logger.info('Testing Spherical Patch...')
        seeds = viscid.SphericalPatch(p0=[0, 0, 0], p1=[0, -0, -1],
                                      max_alpha=30.0, max_beta=59.9,
                                      nalpha=65, nbeta=80, r=0.5, roll=45.0)
        run_test(logo, seeds, plot2d=plot2d, plot3d=plot3d, show=args.show)

    return 0

if __name__ == "__main__":
    sys.exit(main())

##
## EOF
##
