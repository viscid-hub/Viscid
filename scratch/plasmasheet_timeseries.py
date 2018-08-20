#!/usr/bin/env python
#
# Numpy slicing reference
# https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html
#
# Numpy broadcasting reference
# https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html

from __future__ import print_function
import sys

import numpy as np
import viscid
from viscid.plot import vpyplot as vlt


viscid.readers.openggcm.GGCMFile.read_log_file = True
viscid.readers.openggcm.GGCMGrid.mhd_to_gse_on_read = "auto"


def _main():
    f = viscid.load_file('~/dev/work/xi_fte_001/*.3d.*.xdmf')
    time_slice = ':'
    times = np.array([grid.time for grid in f.iter_times(time_slice)])

    # XYZ coordinates of virtual satelites in warped "plasma sheet coords"
    x_sat_psc = np.linspace(-30, 0, 31)  # X (GSE == PSC)
    y_sat_psc = np.linspace(-10, 10, 21)  # Y (GSE == PSC)
    z_sat_psc = np.linspace(-2, 2, 5)  # Z in PSC (z=0 is the plasma sheet)

    # the GSE z location of the virtual satelites in the warped plasma sheet
    # coordinates, so sat_z_gse_ts['x=5j, y=1j, z=0j'] would give the
    # plasma sheet location at x=5.0, y=1.0
    # These fields depend on time because the plasma sheet moves in time
    sat_z_gse_ts = viscid.zeros([times, x_sat_psc, y_sat_psc, z_sat_psc],
                                crd_names='txyz', center='node',
                                name='PlasmaSheetZ_GSE')
    vx_ts = viscid.zeros_like(sat_z_gse_ts)
    bz_ts = viscid.zeros_like(sat_z_gse_ts)

    for itime, grid in enumerate(f.iter_times(time_slice)):
        print("Processing time slice", itime, grid.time)

        gse_slice = 'x=-35j:0j, y=-15j:15j, z=-6j:6j'
        bx = grid['bx'][gse_slice]
        bx_argmin = np.argmin(bx**2, axis=2)
        z_gse = bx.get_crd('z')
        # ps_zloc_gse is the plasma sheet z location along the GGCM grid x/y
        ps_z_gse = viscid.zeros_like(bx[:, :, 0:1])
        ps_z_gse[...] = z_gse[bx_argmin]

        # Note: Here you could apply a gaussian filter to
        #       ps_z_gse[:, :, 0].data in order to smooth the surface
        #       if desired. Scipy / Scikit-Image have some functions
        #       that do this

        # ok, we found the plasma sheet z GSE location on the actual GGCM
        # grid, but we just want a subset of that grid for our virtual
        # satelites, so just interpolate the ps z location to our subset
        ps_z_gse_subset = viscid.interp_trilin(ps_z_gse,
                                               sat_z_gse_ts[itime, :, :, 0:1],
                                               wrap=True)
        # now we know the plasma sheet z location in GSE, and how far
        # apart we want the satelites in z, so put those two things together
        # to get a bunch of satelite locations
        sat_z_gse_ts[itime] = ps_z_gse_subset.data + z_sat_psc.reshape(1, 1, -1)

        # make a seed generator that we can use to fill the vx and bz
        # time series for this instant in time
        sat_loc_gse = sat_z_gse_ts[itime].get_points()
        sat_loc_gse[2, :] = sat_z_gse_ts[itime].data.reshape(-1)

        # slicing the field before doing the interpolation makes this
        # faster for hdf5 data, but probably for other data too
        vx_ts[itime] = viscid.interp_trilin(grid['vx'][gse_slice],
                                            sat_loc_gse,
                                            wrap=False
                                            ).reshape(vx_ts.shape[1:])
        bz_ts[itime] = viscid.interp_trilin(grid['bz'][gse_slice],
                                            sat_loc_gse,
                                            wrap=False
                                            ).reshape(bz_ts.shape[1:])

        # 2d plots of the plasma sheet z location to make sure we did the
        # interpolation correctly
        if False:  # pylint: disable=using-constant-test
            from viscid.plot import vpyplot as vlt
            fig, (ax0, ax1) = vlt.subplots(2, 1)  # pylint: disable=unused-variable
            vlt.plot(ps_z_gse, ax=ax0, clim=(-5, 5))
            vlt.plot(ps_z_gse_subset, ax=ax1, clim=(-5, 5))
            vlt.auto_adjust_subplots()
            vlt.show()

        # make a 3d plot of the plasma sheet surface to verify that it
        # makes sense
        if True:  # pylint: disable=using-constant-test
            from viscid.plot import vlab
            fig = vlab.figure(size=(1280, 800), bgcolor=(1, 1, 1),
                              fgcolor=(0, 0, 0))
            vlab.clf()
            # plot the plasma sheet coloured by vx
            # Note: points closer to x = 0 are unsightly since the plasma
            #       sheet criteria starts to fall apart on the flanks, so
            #       just remove the first few rows
            ps_z_gse_tail = ps_z_gse['x=:-2.25j']
            ps_mesh_shape = [3, ps_z_gse_tail.shape[0], ps_z_gse_tail.shape[1]]
            ps_pts = ps_z_gse_tail.get_points().reshape(ps_mesh_shape)
            ps_pts[2, :, :] = ps_z_gse_tail[:, :, 0]
            plasma_sheet = viscid.RectilinearMeshPoints(ps_pts)
            ps_vx = viscid.interp_trilin(grid['vx'][gse_slice], plasma_sheet)
            _ = vlab.mesh_from_seeds(plasma_sheet, scalars=ps_vx)
            vx_clim = (-1400, 1400)
            vx_cmap = 'viridis'
            vlab.colorbar(title='Vx', clim=vx_clim, cmap=vx_cmap,
                          nb_labels=5)
            # plot satelite locations as dots colored by Vx with the same
            # limits and color as the plasma sheet mesh
            sat3d = vlab.points3d(sat_loc_gse[0], sat_loc_gse[1], sat_loc_gse[2],
                                  vx_ts[itime].data.reshape(-1),
                                  scale_mode='none', scale_factor=0.2)
            vlab.apply_cmap(sat3d, clim=vx_clim, cmap=vx_cmap)

            # plot Earth for reference
            cotr = viscid.Cotr(dip_tilt=0.0)  # pylint: disable=not-callable
            vlab.plot_blue_marble(r=1.0, lines=False, ntheta=64, nphi=128,
                                  rotate=cotr, crd_system='mhd')
            vlab.plot_earth_3d(radius=1.01, night_only=True, opacity=0.5,
                               crd_system='gse')
            vlab.view(azimuth=45, elevation=70, distance=35.0,
                      focalpoint=[-9, 3, -1])
            vlab.savefig('plasma_sheet_3d_{0:02d}.png'.format(itime))
            vlab.show()
            try:
                vlab.mlab.close(fig)
            except TypeError:
                pass  # this happens if the figure is already closed

    # now do what we will with the time series... this is not a good
    # presentation of this data, but you get the idea
    from viscid.plot import vpyplot as vlt
    fig, axes = vlt.subplots(4, 4, figsize=(12, 12))
    for ax_row, yloc in zip(axes, np.linspace(-5, 5, len(axes))[::-1]):
        for ax, xloc in zip(ax_row, np.linspace(4, 7, len(ax_row))):
            vlt.plot(vx_ts['x={0}j, y={1}j, z=0j'.format(xloc, yloc)], ax=ax)
            ax.set_ylabel('')
            vlt.plt.title('x = {0:g}, y = {1:g}'.format(xloc, yloc))
    vlt.plt.suptitle('Vx [km/s]')
    vlt.auto_adjust_subplots()
    vlt.show()

    return 0

if __name__ == "__main__":
    _main()
