#!/usr/bin/env python

from __future__ import print_function
import code

import numpy as np
from mayavi import mlab
from mayavi.sources.vtk_data_source import VTKDataSource
from mayavi.sources.builtin_surface import BuiltinSurface
from tvtk.api import tvtk

from .. import field
from ..calculator.topology import color_from_topology

# def data_source(fld):
#     """ get a data source from a scalar field """
#     dat = tvtk.RectilinearGrid()
#
#     suffix = "cc" if fld.iscentered("Cell") else ""
#     dat.x_coordinates = fld.crds['x' + suffix]
#     dat.y_coordinates = fld.crds['y' + suffix]
#     dat.z_coordinates = fld.crds['z' + suffix]
#
#     if isinstance(fld, field.ScalarField):
#         dat.point_data.scalars = np.ravel(fld.dat)
#         dat.point_data.scalars.name = fld.name
#     elif isinstance(fld, field.VectorField):
#         pass
#
#     return VTKDataSource(data=dat)
#
#     rr = tvtk.RectilinearGrid()
#
#     rr.point_data.scalars.name = 'rr'
#     rr.dimensions = collection[0]['rr'].shape[::-1]
#     #print(collection[0]['coords'][:])
#     rr_src = VTKDataSource(data=rr)
#
#     b = tvtk.RectilinearGrid()
#     bdata = np.empty((np.prod(collection[0]['bx'].shape), 3))
#     for i in range(3):
#         bdata[:,i] = np.asarray(collection[0]['b'][:,:,:,i],
#                                 dtype='f').ravel()
#     b.point_data.vectors = bdata
#     b.dimensions = collection[0]['b'].shape[:-1][::-1]
#
#     b_src = VTKDataSource(data=b)
#     collection.unload()

def field_to_source(fld):
    if fld.iscentered("Node"):
        return field_to_point_source(fld)
    elif fld.iscentered("Cell"):
        return field_to_cell_source(fld)
    else:
        raise NotImplementedError("cell / node only for now")

def field_to_point_source(fld):
    grid, arr = _prep_field(fld)
    dat_target = grid.point_data

    if fld.iscentered("Cell"):
        # grid.dimensions is x, y, z not z, y, x
        grid.dimensions = tuple(fld.crds.shape_cc[::-1])
        grid.x_coordinates = fld.get_crd_cc('x')
        grid.y_coordinates = fld.get_crd_cc('y')
        grid.z_coordinates = fld.get_crd_cc('z')
    elif fld.iscentered("Node"):
        # grid.dimensions is x, y, z not z, y, x
        grid.dimensions = tuple(fld.crds.shape_nc[::-1])
        grid.x_coordinates = fld.get_crd_nc('x')
        grid.y_coordinates = fld.get_crd_nc('y')
        grid.z_coordinates = fld.get_crd_nc('z')
    else:
        raise ValueError("cell or node only please")

    return _finalize_source(fld, arr, grid, dat_target)

def field_to_cell_source(fld):
    grid, arr = _prep_field(fld)
    dat_target = grid.cell_data

    if fld.iscentered("Cell"):
        # grid.dimensions is x, y, z not z, y, x
        grid.dimensions = tuple(fld.crds.shape_cc[::-1])
        grid.x_coordinates = fld.get_crd_nc('x')
        grid.y_coordinates = fld.get_crd_nc('y')
        grid.z_coordinates = fld.get_crd_nc('z')
    elif fld.iscentered("Node"):
        raise NotImplementedError("can't do lossless cell data from nodes yet")
    else:
        raise ValueError("cell or node only please")

    return _finalize_source(fld, arr, grid, dat_target)

def _prep_field(fld):
    grid = tvtk.RectilinearGrid()

    if isinstance(fld, field.ScalarField):
        arr = np.reshape(fld.data, (-1,))
    elif isinstance(fld, field.VectorField):
        arr = np.reshape(fld.data, (-1, 3))
    else:
        raise ValueError("Unexpected fld type: {0}".format(type(fld)))
    # swap endian if needed
    if str(arr.dtype).startswith(">"):
        arr = arr.byteswap().newbyteorder()
    return grid, arr

def _finalize_source(fld, arr, grid, dat_target):
    if isinstance(fld, field.ScalarField):
        dat_target.scalars = arr
        dat_target.scalars.name = fld.name
    elif isinstance(fld, field.VectorField):
        dat_target.vectors = arr
        dat_target.vectors.name = fld.name
    return VTKDataSource(data=grid)

def plot_lines(lines, topology=None, **kwargs):
    if "color" not in kwargs and topology is not None:
        if isinstance(topology, field.Field):
            topology = topology.data.reshape(-1)
        topo_color = True
    else:
        topo_color = False

    for i, line in enumerate(lines):
        if topo_color:
            kwargs["color"] = color_from_topology(topology[i])
        mlab.plot3d(line[2], line[1], line[0], **kwargs)

def mlab_earth(pipeline, daycol=(1, 1, 1), nightcol=(0, 0, 0), res=15,
               crd_system="mhd"):
    crd_system = crd_system.lower()
    if crd_system == "mhd":
        theta_dusk, theta_dawn = 270, 90
    elif crd_system == "gse":
        theta_dusk, theta_dawn = 90, 270

    night = BuiltinSurface(source='sphere', name='night')
    night.data_source.set(center=(0, 0, 0), radius=1.0, start_theta=theta_dusk,
                          end_theta=theta_dawn, theta_resolution=res,
                          phi_resolution=res)
    pipeline.surface(night, color=nightcol)

    day = BuiltinSurface(source='sphere', name='day')
    day.data_source.set(center=(0, 0, 0), radius=1.0, start_theta=theta_dawn,
                        end_theta=theta_dusk, theta_resolution=res,
                        phi_resolution=res)
    pipeline.surface(day, color=daycol)

def show(stop=True):
    mlab.show(stop=stop)

def interact(local=None):
    """ you probably want to use interact(local=locals()) """
    banner = """Have some fun with mayavi :)
    hints:
        - use locals() to explore the namespace
        - mlab.show(stop=True) or mvi.show() to interact with the plot/gui
        - mayavi objects all have a trait_names() method
        - Use Ctrl-D (eof) to end interaction """
    code.interact(banner, local=local)

##
## EOF
##
