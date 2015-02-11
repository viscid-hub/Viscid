#!/usr/bin/env python
# this script is for GEM 2012 plots... this is planned to be obsolete

from __future__ import print_function
from os import path
import sys

import numpy as np
import numexpr as ne
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mayavi import mlab
# from tvtk.api import tvtk
# from mayavi.sources.vtk_data_source import VTKDataSource
# from mayavi.modules.iso_surface import IsoSurface
# from mayavi.modules.streamline import Streamline

from viscid import logger
from viscid import vlab, field, coordinate
from viscid.tools import topology
from viscid.calculator import calc
from viscid.plot import mvi
from viscid.plot import mpl

SKIP = 10

safety = 0.5

#def meshgrid3(x, y, z):
#    Xd = np.tile(x.reshape((-1, 1, 1)), (1, len(y), len(z))).flatten()
#    Yd = np.tile(y.reshape((1, -1, 1)), (len(x), 1, len(z))).flatten()
#    Zd = np.tile(z.reshape((1, 1, -1)), (len(x), len(y), 1)).flatten()
#    return Xd, Yd, Zd

def resample_field(sample_fld, src_fld, name):
    """ returns a field """
    if not isinstance(sample_fld, field.ScalarField):
        raise TypeError("Sample is not a scalar field!")
    data = np.ones_like(sample_fld.data) * np.nan

    x, y, z = sample_fld.crds[('xcc', 'ycc', 'zcc')]
    src_x, src_y, src_z = src_fld.crds[('x', 'y', 'z')]
    xminind = closest1d_ind(x, src_x[-1]); xmaxind = closest1d_ind(x, src_x[0])  # REVERSED FOR GSE!!!
    yminind = closest1d_ind(y, src_y[-1]); ymaxind = closest1d_ind(y, src_y[0])  # REVERSED FOR GSE!!!
    zminind = closest1d_ind(z, src_z[0]); zmaxind = closest1d_ind(z, src_z[-1])
    #print("({0}, {1}, {2})  ({3}, {4}, {5})".format(xmin, ymin, zmin, xmax, ymax, zmax))
    #print("({0}, {1}, {2})  ({3}, {4}, {5})".format(xminind, yminind, zminind, xmaxind, ymaxind, zmaxind))
    src_dat_shaped = src_dat.reshape((len(src_z), len(src_y), len(src_x)))
    for i in range(xminind, xmaxind + 1):
        for j in range(yminind, ymaxind + 1):
            for k in range(zminind, zmaxind + 1):
                nval = nearest_val(src_dat_shaped, src_fld, (x[i], y[j], z[k]))
                #if j==128 and k==128:
                #    print(i, x[i], y[j], z[k], nval)
                data[k, j, i] = nval
    print("data fill done")

    return field.wrap_field(name, sample_fld.center, sample_fld.crds,
                            data, sample_fld.time, "Scalar")

def plot_line_file(fname):
    with open(fname, 'r') as f:
        mode = "start"
        for line in f:
            #print(line)
            if mode == "start":
                col = tuple([float(d) for d in line.split()])
                mode = "readnpts"

            elif mode == "readnpts":
                n = int(line.split()[0])
                i = 0
                x, y, z = [np.empty([n / SKIP]) for d in range(3)]
                #pts = []
                mode = "readpt"

            elif mode == "readpt":
                sp = line.split()
                ind = int(sp[0])

                if ind % 10 == 0:
                    ptx, pty, ptz = [float(d) for d in sp[1:4]]
                    x[i] = -ptx
                    y[i] = -pty
                    z[i] = ptz
                    #print((ptx, pty, ptz))
                    i += 1

                #pt = (ptx, pty, ptz)
                #if pt in pts:
                #    print("totally")
                #pts.append(pt)

                if int(ind) == n:
                    mlab.plot3d(x, y, z, color=col, tube_radius=0.015)
                    mode = "start"

def plot_marker_file(fname):
    try:
        x, y, z = np.loadtxt(fname, usecols=[0,1,2], unpack=True)
        mlab.points3d(-x, -y, z, color=(.2, .2, .2), scale_factor=0.15)
    except ValueError:
        print("no markers to plot")

def plot_meeting_file(fname):
    try:
        x, y, z = np.loadtxt(fname, usecols=[0,1,2], unpack=True)
        mlab.points3d(-x, -y, z, color=(.0, .5, .0), scale_factor=0.05)
    except ValueError:
        print("no meeting points to plot")

if __name__=='__main__':
    dir = sys.argv[1]
    run = sys.argv[2]
    time = int(sys.argv[3])
    res = int(sys.argv[4])
    try:
        stream_color = sys.argv[5]
    except IndexError:
        stream_color = "magnitude"

    strtime = "{0:06d}".format(time)
    strres = "{0}".format(res)

    lines_file = "{0}/li.recon-lines.{1}".format(dir, strtime)
    if path.exists(lines_file):
        pass
        plot_line_file(lines_file)
    else:
        logger.warn("Lines file not found".format(lines_file))

    markers_file = "{0}/li.recon-markers.{1}".format(dir, strtime)
    if path.exists(markers_file):
        plot_marker_file(markers_file)
    else:
        logger.warn("Lines file not found".format(markers_file))

    itopfile = "{0}/{1}.3df{2}.itop.{3}.txt".format(dir, run, strtime, strres)
    ihemfile = "{0}/{1}.3df{2}.ihem.{3}.txt".format(dir, run, strtime, strres)
    topo_arr = None
    if path.exists(itopfile):
        topo_fld = topology.load_topo(itopfile, ihemfile)
        topo_src = mvi.field_to_point_source(topo_fld)
        e=mlab.get_engine()
        e.add_source(topo_src)
        mlab.axes(topo_src)

        # p1 = mlab.pipeline.scalar_cut_plane(topo_src, plane_orientation='x_axes',
        #                                     transparent=True, opacity=1.0, view_controls=True)
        p2 = mlab.pipeline.scalar_cut_plane(topo_src, plane_orientation='y_axes',
                                            transparent=True, opacity=0.5, view_controls=True)
        # p2.module_manager.scalar_lut_manager.reverse_lut = True
        # p2.module_manager.scalar_lut_manager.show_scalar_bar = True
        # p3 = mlab.pipeline.scalar_cut_plane(topo_src, plane_orientation='z_axes',
        #                                     transparent=True, opacity=0.5, view_controls=True)
        # p3.module_manager.scalar_lut_manager.reverse_lut = True

        #iso = mlab.pipeline.iso_surface(topo_src, opacity=0.9)
        #iso.module_manager.scalar_lut_manager.reverse_lut = True
        # mlab.colorbar(orientation='vertical')
        # mlab.outline()
    else:
        logger.warn("topology file {0} does not exist".format(itopfile))

    shear_msx = None
    xdmf_file = "{0}/{1}.3df.{2}.xdmf".format(dir, run, strtime)
    if path.exists(xdmf_file):
        f = vlab.load_vfile(xdmf_file)
        b = f['b']
        v = f['v']
        xj = f['xj']

        resampled_topo_fld = None
        if stream_color == "itop" and topo_fld is not None:
            resampled_topo_fld = resample_field(f['rr'], topo_fld, "Topology")
            #itop_src = mvi.field_to_point_source(itop_fld)
            #e.add_source(itop_src)
            #ptop = mlab.pipeline.scalar_cut_plane(itop_src, plane_orientation='y_axes',
            #                                 transparent=True, opacity=1.0, view_controls=False)
            #ptop.module_manager.scalar_lut_manager.show_scalar_bar = True

        if topo_fld is not None:
            shear_mpx, shear_msx, shear_y, shear_z, shear, shear_array, ix, iy, iz = topology.shear_angle(topo_fld, b, safety)
            #Y, Z = np.ix_(shear_y, shear_z)
            # mlab.points3d(shear_x, shear_y, shear_z, color=(.8, 0, 0), scale_factor=0.15)
            shear_cubes = mlab.points3d(shear_mpx, shear_y, shear_z, shear, scale_factor=0.0005,
                                        mode="cube", colormap='spectral', vmin=0, vmax=180)
            shear_cubes.module_manager.scalar_lut_manager.show_scalar_bar = True
            #sheath = mlab.points3d(shear_msx, shear_y, shear_z, color=(0,0,.8), scale_factor=0.05,
            #                       mode="sphere", colormap='spectral')

            xpt_x, xpt_y, xpt_z = topology.find_xpoint_fast(topo_fld, ix, iy, iz)
            mlab.points3d(xpt_x, xpt_y, xpt_z, color=(.8, 0, 0), scale_factor=0.15)

        bsrc = mvi.field_to_point_source(b)
        vsrc = mvi.field_to_point_source(v)
        xjsrc = mvi.field_to_point_source(xj)

        # xjmag = mlab.pipeline.extract_vector_norm(xjsrc)
        # xjcut = mlab.pipeline.scalar_cut_plane(xjmag, plane_orientation='y_axes',
        #                              transparent=True, opacity=0.5, view_controls=False)
        # xjcut.module_manager.scalar_lut_manager.show_scalar_bar = True

        if resampled_topo_fld is not None:
            bstreamsrc = bsrc
            bstreamsrc.data.point_data.scalars = np.reshape(resampled_topo_fld.data, (-1,))
            #bstreamsrc.data.point_data.scalars.name = "Topology"
        elif stream_color == "magnitude":
            bstreamsrc = mlab.pipeline.extract_vector_norm(bsrc)
        else:
            fld = f[stream_color]
            if isinstance(fld, field.ScalarField):
                scalarfld = fld
            elif isinstance(fld, field.VectorField):
                scalarfld = calc.magnitude(fld)
            else:
                raise TypeError("must be vector or scalar")
            bstreamsrc = bsrc
            #print(scalarfld.data[128,128,:])
            arr = np.reshape(scalarfld.data, (-1,))
            if str(arr.dtype).startswith(">"):
                arr = arr.byteswap().newbyteorder()
            #print("TTT ", arr.dtype, max(arr), min(arr))
            #print("NNN ", arr[:,128,128])
            bstreamsrc.data.point_data.scalars = np.reshape(arr, (-1,))
            #bstreamsrc.data.point_data.scalars.name = stream_color

        # b fields in a line
        # bsl1 = mlab.pipeline.streamline(bstreamsrc, seedtype='line', integration_direction='both')
        # bsl1.stream_tracer.maximum_propagation = 100.
        # bsl1.seed.widget.resolution = 50
        # bsl1.seed.widget.point1 = [-10, 0, 0]
        # bsl1.seed.widget.point2 = [-7.5, 0, 0]
        # bsl1.streamline_type = 'tube'
        # bsl1.tube_filter.radius = 0.03
        # bsl1.stop()
        # bsl1.start()
        # bsl1.seed.widget.enabled = True

        # b fields in a sphere
        # bsl2 = mlab.pipeline.streamline(bstreamsrc, seedtype='sphere', integration_direction='both',
        #     seed_resolution=4)
        # bsl2.stream_tracer.maximum_propagation = 20.
        # bsl2.seed.widget.center = [-8, 0, 0]
        # bsl2.seed.widget.radius = 1.0
        # bsl2.streamline_type = 'tube'
        # bsl2.tube_filter.radius = 0.03
        # bsl2.stop()
        # bsl2.start()
        # bsl2.seed.widget.enabled = True

        # b fields around earth
        # bsl3 = mlab.pipeline.streamline(bstreamsrc, seedtype='sphere', integration_direction='both',
        #     seed_resolution=20)
        # bsl3.stream_tracer.maximum_propagation = 20.
        # bsl3.seed.widget.center = [0, 0, 0]
        # bsl3.seed.widget.radius = 4.0
        # bsl3.streamline_type = 'tube'
        # bsl3.tube_filter.radius = 0.01
        # bsl3.module_manager.scalar_lut_manager.show_scalar_bar = True
        # bsl3.stop()
        # bsl3.start()
        # bsl3.seed.widget.enabled = True

        # velocity streamlines
        # vmag = mlab.pipeline.extract_vector_norm(vsrc)
        # vsl = mlab.pipeline.streamline(vmag, seedtype='sphere', integration_direction='both',
        #     seed_resolution=5)
        # vsl.stream_tracer.maximum_propagation = 20.
        # vsl.seed.widget.center = [-8, 0, 0]
        # vsl.seed.widget.radius = 0.5
        # vsl.streamline_type = 'tube'
        # vsl.tube_filter.radius = 0.03
        # vsl.module_manager.scalar_lut_manager.show_scalar_bar = False
        # vsl.stop()
        # vsl.start()
        # vsl.seed.widget.enabled = True

        # if shear_msx is not None:
        #     for x, y, z in zip(shear_msx, shear_y, shear_z):
        #         if np.abs(z) > 0.1 or np.abs(y) > 0.1:
        #             continue
        #         bslmp = mlab.pipeline.streamline(bstreamsrc, seedtype='line', integration_direction='both',
        #             seed_resolution=3)
        #         bslmp.stream_tracer.maximum_propagation = 100.
        #         bslmp.seed.widget.point1 = [x - safety, y, z]
        #         bslmp.seed.widget.point2 = [x, y, z]
        #         bslmp.streamline_type = 'tube'
        #         bslmp.tube_filter.radius = 0.03
        #         bslmp.stop()
        #         bslmp.start()
        #         bslmp.seed.widget.enabled = True

    else:
        logger.warn("xdmf file {0} not found".format(xdmf_file))

    # mlab.outline()

    meeting_file = "{0}/{1}.3df{2}.meeting.{3}.txt".format(dir, run, strtime, strres)
    if path.exists(meeting_file):
        pass
        # plot_meeting_file(meeting_file)
    else:
        logger.warn("Meeting file {0} not found".format(meeting_file))

    #mlab.colorbar(orientation='vertical')
    mvi.mlab_earth(mlab.pipeline)
    mlab.view(focalpoint=(0, 0, 0))
    mlab.show(stop=False)

##
## EOF
##
