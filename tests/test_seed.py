#!/usr/bin/env python
"""Test interpolation with the gamut of seed generators"""

from __future__ import division, print_function
import argparse
import os
import sys

from viscid_test_common import next_plot_fname

import numpy as np
import viscid
from viscid import sample_dir


_global_ns = dict()


# The reference plots are made in GSE crds (for the RectilinearMeshPoints
# test). Usually these flags should be set in your rc file. See the
# corresponding page in the tutorial for more information.
viscid.readers.openggcm.GGCMFile.read_log_file = True
viscid.readers.openggcm.GGCMGrid.mhd_to_gse_on_read = "auto"


def run_test(fld, seeds, plot2d=True, plot3d=True, add_title="",
             view_kwargs=None, show=False):
    interpolated_fld = viscid.interp_trilin(fld, seeds)
    seed_name = seeds.__class__.__name__
    if add_title:
        seed_name += " " + add_title

    try:
        if not plot2d:
            raise ImportError
        from matplotlib import pyplot as plt
        from viscid.plot import vpyplot as vlt
        plt.clf()
        # plt.plot(seeds.get_points()[2, :], fld)
        mpl_plot_kwargs = dict()
        if interpolated_fld.is_spherical():
            mpl_plot_kwargs['hemisphere'] = 'north'
        vlt.plot(interpolated_fld, **mpl_plot_kwargs)
        plt.title(seed_name)

        plt.savefig(next_plot_fname(__file__, series='2d'))
        if show:
            plt.show()
    except ImportError:
        pass

    try:
        if not plot3d:
            raise ImportError
        from viscid.plot import vlab

        try:
            fig = _global_ns['figure']
            vlab.clf()
        except KeyError:
            fig = vlab.figure(size=[1200, 800], offscreen=not show)
            _global_ns['figure'] = fig

        try:
            mesh = vlab.mesh_from_seeds(seeds, scalars=interpolated_fld)
            mesh.actor.property.backface_culling = True
        except RuntimeError:
            pass

        pts = seeds.get_points()
        p = vlab.points3d(pts[0], pts[1], pts[2], interpolated_fld.flat_data,
                          scale_mode='none', scale_factor=0.02)
        vlab.axes(p)
        vlab.title(seed_name)
        if view_kwargs:
            vlab.view(**view_kwargs)

        vlab.savefig(next_plot_fname(__file__, series='3d'))
        if show:
            vlab.show()
    except ImportError:
        pass


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notwo", dest='notwo', action="store_true")
    parser.add_argument("--nothree", dest='nothree', action="store_true")
    parser.add_argument("--show", "--plot", action="store_true")
    args = viscid.vutil.common_argparse(parser, default_verb=0)

    plot2d = not args.notwo
    plot3d = not args.nothree

    # plot2d = True
    # plot3d = True
    # args.show = True

    img = np.load(os.path.join(sample_dir, "logo.npy"))
    x = np.linspace(-1, 1, img.shape[0])
    y = np.linspace(-1, 1, img.shape[1])
    z = np.linspace(-1, 1, img.shape[2])
    logo = viscid.arrays2field([x, y, z], img)

    if 1:
        viscid.logger.info('Testing Point with user-specified local coordinates...')
        pts = np.vstack([[-1, -0.5, 0, 0.5, 1], [-1, -0.5, 0, 0.5, 1], [0, 0.5, 1, 1.5, 2]])
        local_crds = viscid.asarray_datetime64([0, 60, 120, 180, 240], conservative=True)
        seeds = viscid.Point(pts, local_crds=local_crds)
        run_test(logo, seeds, plot2d=plot2d, plot3d=plot3d, show=args.show)

    if 1:
        viscid.logger.info('Testing Line...')
        seeds = viscid.Line([-1, -1, 0], [1, 1, 2], n=5)
        run_test(logo, seeds, plot2d=plot2d, plot3d=plot3d, show=args.show)

    if 1:
        viscid.logger.info('Testing Spline (two knots, i.e., straight line)...')
        seeds = viscid.Spline([[-1, 1], [-1, 1], [0, 2]], n=5)
        run_test(logo, seeds, plot2d=plot2d, plot3d=plot3d, show=args.show)

    if 1:
        viscid.logger.info('Testing Spline (three knots)...')
        seeds = viscid.Spline([[-1, 1, 0], [-1, 1, 0], [0, 2, 0]], n=5)
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

    if 1:
        viscid.logger.info('Testing RectilinearMeshPoints...')
        f = viscid.load_file(os.path.join(sample_dir, 'sample_xdmf.3d.[-1].xdmf'))
        slc = 'x=-40f:12f, y=-10f:10f, z=-10f:10f'
        b = f['b'][slc]
        z = b.get_crd('z')
        sheet_iz = np.argmin(b['x']**2, axis=2)
        sheet_pts = b['z=0:1'].get_points()
        sheet_pts[2, :] = z[sheet_iz].reshape(-1)
        isphere_mask = np.sum(sheet_pts[:2, :]**2, axis=0) < 5**2
        day_mask = sheet_pts[0:1, :] > -1.0
        sheet_pts[2, :] = np.choose(isphere_mask, [sheet_pts[2, :], 0])
        sheet_pts[2, :] = np.choose(day_mask, [sheet_pts[2, :], 0])
        nx, ny, _ = b.sshape
        sheet_seed = viscid.RectilinearMeshPoints(sheet_pts.reshape(3, nx, ny))
        vx_sheet = viscid.interp_nearest(f['vx'], sheet_seed)

        try:
            if not plot2d:
                raise ImportError
            from matplotlib import pyplot as plt
            from viscid.plot import vpyplot as vlt
            vlt.clf()
            vlt.plot(vx_sheet, symmetric=True)
            plt.savefig(next_plot_fname(__file__, series='2d'))
            if args.show:
                vlt.show()
        except ImportError:
            pass

        try:
            if not plot3d:
                raise ImportError
            from viscid.plot import vlab
            vlab.clf()
            mesh = vlab.mesh_from_seeds(sheet_seed, scalars=vx_sheet,
                                        clim=(-400, 400))
            vlab.plot_earth_3d(crd_system=b)
            vlab.view(azimuth=+90.0 + 45.0, elevation=90.0 - 25.0,
                      distance=30.0, focalpoint=(-10.0, +1.0, +1.0))

            vlab.title("RectilinearMeshPoints")
            vlab.savefig(next_plot_fname(__file__, series='3d'))
            if args.show:
                vlab.show()

        except ImportError:
            pass

    # prevent weird xorg bad-instructions on tear down
    if 'figure' in _global_ns and _global_ns['figure'] is not None:
        from viscid.plot import vlab
        vlab.mlab.close(_global_ns['figure'])

    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
