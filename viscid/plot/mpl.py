#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import pylab as pl

from .. import field
from .. import vutil

def plot(fld, selection=None, **kwargs):
    """ just plot... should generically dispatch to gen the right
    matplotlib plot given the field. returns the mpl plot and color bar
    as a tuple """
    if isinstance(fld, field.ScalarField):
        fld = fld.slice(selection, rm_len1_dims=True)
        # print(selection)
        # print(fld.crds.shape, fld.shape)

        if fld.dim == 1:
            return plot1d_field(fld, **kwargs)
        elif fld.dim == 2:
            return pcolor_field(fld, **kwargs)
        else:
            raise ValueError("mpl can only do 1-D or 2-D fields")
    else:
        raise TypeError("I can only do scalar fields right now")

# def parse_opts(plot_opts):
#     """ opts string looks like 'log,x_-20_10', output is
#     [['log'], ['x', '-20', '10']] """
#     if isinstance(plot_opts, str):
#         plot_opts = plot_opts.split(",")
#     elif plot_opts is None:
#         plot_opts = []

#     for i, opt in plot_opts:
#         if isinstance(opt, str):
#             plot_opts[i] = opt.split("_")

#     return plot_opts

# def do_plot_opts(axis, plot_opts):
#     plot_opts = parse_opts(plot_opts)
#     for opt in plot_opts:
#         if opt == "log":
#             pass

def contour2d(fld, selection=None, **kwargs):
    fld = fld.slice(selection, rm_len1_dims=True)

    if fld.dim != 2:
        raise ValueError("I will only contour a 2d field")
    else:
        return contour_field(fld, **kwargs)


def pcolor_field(fld, ax=None, equalaxis=True, earth=None,
                 show=False, mask_nan=False, mod=None, **kwargs):
    #print(slcrds[0][0], slcrds[0][1].shape)
    #print(slcrds[1][0], slcrds[1][1].shape)
    #print(dat.shape)
    cbar = None

    if not ax:
        ax = pl.gca()

    # THIS IS BACKWARD, on account of the convention for
    # Coordinates where z, y, x is used since that is how
    # xdmf data is
    namey, namex = fld.crds.axes # fld.crds.get_culled_axes()

    # pcolor mesh uses node coords, and cell data, if we have
    # node data, fake it by using cell centered coords and
    # trim the edges of the data... maybe i should just be faking
    # the coord array somehow to show the edges...
    if fld.center == "Node":
        X, Y = fld.crds[(namex.upper()+'cc', namey.upper()+'cc')]
        vutil.warn("pcolormesh on node centered field... trimming the edges")
        dat = fld.data[1:-1, 1:-1]
    elif fld.center == "Cell":
        X, Y = fld.crds[(namex.upper(), namey.upper())]
        dat = fld.data

    if mod:
        X *= mod[0]
        Y *= mod[1]

    # print(x.shape, y.shape, fld.data.shape)
    if mask_nan:
        dat = np.ma.masked_where(np.isnan(dat), dat)
    plt = pl.pcolormesh(X, Y, dat, **kwargs)
    cbar = pl.colorbar()
    pl.xlabel(namex)
    pl.ylabel(namey)

    if equalaxis:
        ax.axis('equal')
    if earth:
        plot_earth(fld)
    if show:
        mplshow()
    return plt, cbar

def contour_field(fld, ax=None, equalaxis=True, earth=None,
                 show=False, mask_nan=False, colorbar=True, mod=None, **kwargs):
    #print(slcrds[0][0], slcrds[0][1].shape)
    #print(slcrds[1][0], slcrds[1][1].shape)
    #print(dat.shape)
    cbar = None
    if not ax:
        ax = pl.gca()

    namey, namex = fld.crds.axes # fld.crds.get_culled_axes()
    if fld.center == "Node":
        X, Y = fld.crds[(namex, namey)]
    elif fld.center == "Cell":
        X, Y = fld.crds[(namex + 'cc', namey + 'cc')]

    if mod:
        X *= mod[0]
        Y *= mod[1]

    # print(x.shape, y.shape, fld.data.shape)
    dat = fld.data
    if mask_nan:
        dat = np.ma.masked_where(np.isnan(dat), dat)
    plt = pl.contour(X, Y, dat, **kwargs)
    if colorbar:
        cbar = pl.colorbar()
    pl.xlabel(namex)
    pl.ylabel(namey)

    if equalaxis:
        ax.axis('equal')
    if earth:
        plot_earth(fld)
    if show:
        mplshow()
    return plt, cbar

def plot1d_field(fld, ax=None, show=False):
    # print(fld.shape, fld.crds.shape)

    if not ax:
        ax = pl.gca()

    namex, = fld.crds.axes
    if fld.center == "Node":
        x = fld.crds[namex]
    elif fld.center == "Cell":
        x = fld.crds[namex + "cc"]

    plt = pl.plot(x, fld.data)

    if show:
        mplshow()
    return plt, None

def mplshow():
    # do i need to do anything special before i show?
    # can't think of anything at this point...
    pl.show()

def plot_earth(fld, axis=None, scale=1.0, rot=0,
               daycol='w', nightcol='k', crds="mhd"):
    """ crds = "mhd" (Jimmy crds) or "gsm" (GSM crds)... gsm is the same
    as rot=180. earth_plane is a string in the format 'y=0.2', this says
    what the 3rd dimension is and sets the radius that the earth should
    be """
    import matplotlib.patches as mpatches

    # take only the 1st reduced dim... this should just work
    try:
        plane, value = fld.info["reduced"][0]
    except KeyError:
        vutil.warn("No reduced dims in the field, i don't know what 2d \n "
                   "plane, we're in and can't figure out the size of earth.")
        return None

    if value**2 >= scale**2:
        return None
    radius = np.sqrt(scale**2 - value**2)

    if not axis:
        axis = pl.gca()

    if crds == "gsm":
        rot = 180

    if plane == 'y' or plane == 'z':
        axis.add_patch(mpatches.Wedge((0, 0), radius, 90 + rot, 270 + rot,
                                      ec=nightcol, fc=daycol))
        axis.add_patch(mpatches.Wedge((0, 0), radius, 270 + rot, 450 + rot,
                                      ec=nightcol, fc=nightcol))
    elif plane == 'x':
        if value < 0:
            axis.add_patch(mpatches.Circle((0, 0), radius, ec=nightcol,
                                           fc=daycol))
        else:
            axis.add_patch(mpatches.Circle((0, 0), radius, ec=nightcol,
                                           fc=nightcol))
    return None

##
## EOF
##
