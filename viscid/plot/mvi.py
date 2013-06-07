#!/usr/bin/env python

from __future__ import print_function

import numpy as np
from mayavi.sources.vtk_data_source import VTKDataSource
from tvtk.api import tvtk

from .. import field

# def data_source(fld):
#     """ get a data source from a scalar field """
#     dat = tvtk.RectilinearGrid()
#
#     suffix = "cc" if fld.center == "Cell" else ""
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

    # at the moment, everything is point data
    # TODO: fix up for cell data
    if fld.center == "Cell":
        dat_center = grid.point_data
        # grid.dimensions is x, y, z not z, y, x
        grid.dimensions = tuple(fld.crds.shape_cc[::-1])
        grid.x_coordinates = fld.crds['xcc']
        grid.y_coordinates = fld.crds['ycc']
        grid.z_coordinates = fld.crds['zcc']
    elif fld.center == "Node":
        dat_center = grid.point_data
        # grid.dimensions is x, y, z not z, y, x
        grid.dimensions = tuple(fld.crds.shape_nc[::-1])
        grid.x_coordinates = fld.crds['x']
        grid.y_coordinates = fld.crds['y']
        grid.z_coordinates = fld.crds['z']
    else:
        raise ValueError("Cell or Node only please")

    if isinstance(fld, field.ScalarField):
        dat_center.scalars = arr
        dat_center.scalars.name = fld.name
    elif isinstance(fld, field.VectorField):
        dat_center.vectors = arr
        dat_center.vectors.name = fld.name

    src = VTKDataSource(data=grid)

    return src

def mlab_earth(pipeline, daycol=(1, 1, 1), nightcol=(0, 0, 0), res=15):
    from mayavi.sources.builtin_surface import BuiltinSurface

    night = BuiltinSurface(source='sphere', name='night')
    night.data_source.set(center=(0, 0, 0), radius=1.0, start_theta=270,
                          end_theta=90, theta_resolution=res,
                          phi_resolution=res)
    pipeline.surface(night, color=nightcol)

    day = BuiltinSurface(source='sphere', name='day')
    day.data_source.set(center=(0, 0, 0), radius=1.0, start_theta=90,
                        end_theta=270, theta_resolution=res, phi_resolution=res)
    pipeline.surface(day, color=daycol)

##
## EOF
##
