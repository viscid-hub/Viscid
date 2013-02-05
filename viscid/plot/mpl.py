#!/usr/bin/env python

from __future__ import print_function
import sys

import numpy as np
import pylab as pl

def warn(message):
    sys.stderr.write("WARNING: {0}\n".format(message))

def plot(fld, selection=None, **kwargs):
    if selection:
        fld = fld.slice(selection)
    # print(slcrds)
    # print(dat)
    if fld.dim == 1:
        return plot1d_field(fld, **kwargs)
    elif fld.dim == 2:
        return pcolor_field(fld, **kwargs)
    else:
        raise ValueError("mpl can only do 1-D or 2-D fields")

def contour2d(fld, selection=None, **kwargs):
    if selection:
        fld = fld.slice(selection)

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
    namey, namex = fld.crds.axes

    # pcolor mesh uses node coords, and cell data, if we have
    # node data, fake it by using cell centered coords and
    # trim the edges of the data... maybe i should just be faking
    # the coord array somehow to show the edges...
    if fld.center == "Node":
        X, Y = fld.crds[(namex.upper()+'cc', namey.upper()+'cc')]
        warn("pcolormesh on node centered field... trimming the edges")
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
        pass #plot_earth()
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

    namey, namex = fld.crds.axes
    if fld.center == "Node":
        X, Y = fld.crds[(namex, namey)]
    elif fld.center == "Cell":
        X, Y = fld.crds[(namex+'cc', namey+'cc')]

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
        pass #plot_earth()
    if show:
        mplshow()
    return plt

def plot1d_field(fld, ax=None, show=False):
    if not ax:
        ax = pl.gca()

    namex, = fld.crds.axes
    x = fld.crds[namex]
    plt = pl.plot(x, fld.data)

    if show:
        mplshow()
    return plt, cbar

def mplshow():
    # do i need to do anything special before i show?
    # can't think of anything at this point...
    pl.show()

def plot_earth(pylab, plane, axis=None, rot=0, daycol='w', nightcol='k'):
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    p = plane.split('_')
    p[1] = float(p[1])
    if np.abs(p[1]) >= 1.0:
        return None
    radius = np.sqrt(1.0 - p[1]**2)
    if not axis:
        axis = pylab.gca()

    if p[0] == 'y' or p[0] == 'z':
        axis.add_patch(mpatches.Wedge((0,0), radius, 90+rot, 270+rot,
                                      ec=nightcol, fc=daycol))
        axis.add_patch(mpatches.Wedge((0,0), radius, 270+rot, 450+rot,
                                      ec=nightcol, fc=nightcol))
    elif p[0] == 'x':
        if p[1] < 0:
            axis.add_patch(mpatches.Circle((0,0), radius, ec=nightcol,
                                           fc=daycol))
        else:
            axis.add_patch(mpatches.Circle((0,0), radius, ec=nightcol,
                                           fc=nightcol))
    return None

##
## EOF
##
