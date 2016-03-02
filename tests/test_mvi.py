#!/usr/bin/env python
""" Try to convert a Field to a mayavi type and plot
streamlines or something """

from __future__ import print_function
import argparse
import sys

from viscid_test_common import sample_dir, next_plot_fname, xfail

try:
    from mayavi import mlab
except ImportError:
    xfail("Mayavi not installed")

import numpy as np
import viscid
from viscid import vutil
from viscid.plot import mvi

def main():
    parser = argparse.ArgumentParser(description="Test calc")
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)

    f3d = viscid.load_file(sample_dir + '/sample_xdmf.3d.[0].xdmf')
    f_iono = viscid.load_file(sample_dir + "/sample_xdmf.iof.[0].xdmf")

    b = f3d["b"]
    pp = f3d["pp"]
    e = f3d["e_cc"]

    # plot a scalar cut plane of pressure
    pp_src = mvi.field2source(pp, center='node')
    scp = mlab.pipeline.scalar_cut_plane(pp_src, plane_orientation='z_axes',
                                         transparent=True, opacity=0.5,
                                         view_controls=False)
    scp.implicit_plane.normal = [0, 0, -1]
    scp.implicit_plane.origin = [0, 0, 0]
    # i don't know why this log10 doesn't seem to work
    cbar = mlab.colorbar(scp, title=pp.name, orientation='vertical')
    cbar.lut.scale = 'log10'
    mvi.apply_cmap(cbar, 'Reds_r')

    # calculate B field lines && topology in viscid and plot them
    seeds = viscid.SphericalPatch([0, 0, 0], [2, 0, 1], 30, 15, r=5.0,
                                  nalpha=5, nbeta=5)
    b_lines, topo = viscid.calc_streamlines(b, seeds, ibound=3.5,
                                            obound0=[-25, -20, -20],
                                            obound1=[15, 20, 20], wrap=True)
    mvi.plot_lines(b_lines, scalars=viscid.topology2color(topo))

    # plot a random circle with scalars colored by the Matplotlib viridis
    # color map, just because we can
    circle = viscid.Circle([0, 0, 0], r=4.0, n=128, endpoint=True)
    scalar = np.sin(circle.as_local_coordinates().get_crd('x'))
    mvi.plot_line(circle.get_points(), scalars=scalar, cmap='viridis')

    # Use Mayavi (VTK) to calculate field lines using an interactive seed
    # These field lines are colored by E parallel, and while the syntax used
    # to add Epar to the B field is hacky looking, it works.
    b_src = mvi.field2source(b, center='node')

    # add e parallel to the b_src so we can use it to color the lines
    epar = viscid.project(e, b)
    epar.name = "Epar"
    epar0_src = mvi.add_field(epar, center='node')
    b_src._point_scalars_list.append(epar0_src.name)  # pylint: disable=protected-access
    b_src.data.point_data.scalars = epar0_src.data.point_data.scalars
    b_src.point_scalars_name = epar0_src.name

    # now add the streamlines
    bsl2 = mlab.pipeline.streamline(b_src, seedtype='sphere',
                                    integration_direction='both',
                                    seed_resolution=4, vmin=-0.1, vmax=0.1)
    # apply the default matplotlib colormap
    mvi.apply_cmap(bsl2)

    bsl2.stream_tracer.maximum_propagation = 20.
    bsl2.seed.widget.center = [-11, 0, 0]
    bsl2.seed.widget.radius = 1.0
    bsl2.streamline_type = 'tube'
    bsl2.tube_filter.radius = 0.03
    bsl2.stop()  # this stop/start was a hack to get something to work?
    bsl2.start()
    bsl2.seed.widget.enabled = False

    cbar = mlab.colorbar(bsl2, title=epar.name, orientation='horizontal')
    cbar.scalar_bar_representation.position = (0.2, 0.01)
    cbar.scalar_bar_representation.position2 = (0.6, 0.14)

    # Plot the ionosphere too
    fac_tot = 1e9 * f_iono['fac_tot']

    crd_system = 'gse'
    m = mvi.plot_ionosphere(fac_tot, crd_system=crd_system, bounding_lat=30.0,
                            vmin=-300, vmax=300, opacity=0.75)

    mvi.plot_blue_marble(r=1.0, orientation=(0, 21.5, -45.0))
    # now shade the night side with a transparent black hemisphere
    mvi.plot_earth_3d(radius=1.01, crd_system="gse", night_only=True,
                      opacity=0.5)

    # mlab.axes(pp_src, nb_labels=5)
    oa = mlab.orientation_axes()
    oa.marker.set_viewport(0.75, 0.75, 1.0, 1.0)

    mvi.resize([1200, 800])
    mlab.view(azimuth=45, elevation=70, distance=35.0, focalpoint=[-2, 0, 0])

    # # Save Figure
    # print("saving png")
    # mvi.mlab.savefig('mayavi_msphere_sample.png')
    # print("saving x3d")
    # # x3d files can be turned into COLLADA files with meshlab, and
    # # COLLADA (.dae) files can be opened in OS X's preview
    # #
    # # IMPORTANT: for some reason, using bounding_lat in mvi.plot_ionosphere
    # #            causes a segfault when saving x3d files
    # #
    # mvi.mlab.savefig('mayavi_msphere_sample.x3d')
    # print("done")

    mvi.savefig(next_plot_fname(__file__))
    if args.show:
        mlab.show()

if __name__ == "__main__":
    main()

##
## EOF
##
