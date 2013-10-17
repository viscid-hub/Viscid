#!/usr/bin/env python

from __future__ import print_function
import logging
from distutils.version import LooseVersion

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.mplot3d import Axes3D #pylint: disable=W0611

from .. import field
from ..calculator import calc
from ..calculator.topology import color_from_topology
# from .. import vutil

__mpl_ver__ = matplotlib.__version__
has_colorbar_gridspec = LooseVersion(__mpl_ver__) > LooseVersion("1.1.1")

def plot(fld, selection=None, **kwargs):
    """ just plot... should generically dispatch to gen the right
    matplotlib plot given the field. returns the mpl plot and color bar
    as a tuple """
    if isinstance(fld, field.ScalarField):
        fld = fld.slice_reduce(selection)

        if fld.nr_sdims == 1:
            return plot1d_field(fld, **kwargs)
        elif fld.nr_sdims == 2:
            return plot2d_field(fld, **kwargs)
        else:
            raise ValueError("mpl can only do 1-D or 2-D fields")
    else:
        raise TypeError("I can only do scalar fields right now")

def _parse_str(plot_opts):
    """ opts string looks like 'log,x=-20_10', output is
    [['log'], ['x', '-20', '10']] """

    if isinstance(plot_opts, str):
        plot_opts = plot_opts.split(",")
    elif plot_opts is None:
        plot_opts = []

    for i, opt in enumerate(plot_opts):
        if isinstance(opt, str):
            plot_opts[i] = opt.replace("=", "_").split("_")
        elif not isinstance(plot_opts[i], (list, tuple)):
            plot_opts[i] = [plot_opts[i]]

    return plot_opts

def _apply_parse_opts(plot_opts_str, fld, kwargs, axis=None):
    """ modifies kwargs and returns a list of things to set after the fact
    kwargs are things that get added as arguments to the plot command
    plot_acts are set using plt.set(act[0], act[1], act[2]) """

    plot_opts = _parse_str(plot_opts_str)
    # acts = []

    if not axis:
        axis = plt.gca()

    for opt in plot_opts:
        if opt[0] == "lin":
            opt = [float(o) if i > 0 else o for i, o in enumerate(opt)]
            axis.set_xscale("linear")
            axis.set_yscale("linear")

            # scale will be centered around 0
            if len(opt) == 2 and float(opt[1]) == 0.0:
                absmax = calc.abs_max(fld)
                opt = [opt[0], -1.0 * absmax, 1.0 * absmax]

            if fld.nr_sdims == 1:
                axis.set_ylim(*opt[1:])
            elif fld.nr_sdims == 2:
                # plt.normalize is deprecated
                # kwargs["norm"] = plt.normalize(*opt[1:])
                kwargs["norm"] = Normalize(*opt[1:])

        elif opt[0] == "log":
            opt = [float(o) if i > 0 else o for i, o in enumerate(opt)]
            axis.set_xscale("linear")

            if fld.nr_sdims == 1:
                axis.set_yscale("log")
                axis.set_ylim(*opt[1:])
            elif fld.nr_sdims == 2:
                axis.set_yscale("linear")
                kwargs["norm"] = LogNorm(*opt[1:])

        elif opt[0] == "loglog":
            opt = [float(o) if i > 0 else o for i, o in enumerate(opt)]
            axis.set_xscale("log")
            axis.set_yscale("log")
            if fld.nr_sdims == 2:
                kwargs["norm"] = LogNorm(*opt[1:])

        elif opt[0] in ["g", "grid"]:
            kwargs["edgecolors"] = 'k'
            kwargs["linewidths"] = 0.2
            kwargs["antialiased"] = True

        elif opt[0] == "x":
            opt = [float(o) if i > 0 else o for i, o in enumerate(opt)]
            axis.set_xlim(*opt[1:])
            # acts.append([axis, "xlim", opt[1:]])

        elif opt[0] == "y":
            opt = [float(o) if i > 0 else o for i, o in enumerate(opt)]
            axis.set_ylim(*opt[1:])
            # acts.append([axis, "ylim", opt[1:]])

        elif opt[0] == "earth":
            kwargs["earth"] = True

        elif opt[0] == "own":
            logging.warn("own axis doesn't seem to work yet...")

        elif opt[0] == "ownx":
            logging.warn("own axis doesn't seem to work yet...")

        elif opt[0] == "owny":
            logging.warn("own axis doesn't seem to work yet...")

        else:
            logging.warn("Unknown plot option ({0})".format(opt[0]))

    # things that i just want to be automagic...
    # use seismic cmap if the data looks centered around 0
    if "norm" in kwargs and "cmap" not in kwargs:
        norm = kwargs["norm"]
        if norm.vmin and norm.vmax and np.abs(norm.vmax + 1.0*norm.vmin) < 1e-4:
            kwargs["cmap"] =  plt.get_cmap('seismic')

    # return axis, acts
    return axis

# def _apply_acts(acts):
#     for act in acts:
#         print(act)
#         plt.setp(act[0], act[1], act[2])

def plot2d_field(fld, style="pcolormesh", ax=None, equalaxis=True,
                 show=False, mask_nan=False, mod=None, plot_opts=None,
                 colorbar=True, rotate_plot=False, **kwargs):
    """ Plot a 2D Field using pcolormesh or contour or something like that...

    style: "pcolormesh", "contour", "pcolor", style of 2D plot
    ax: axis to plot in (plt.gca() used if not specified)
    equalaxis: whether or not the plot will enforce equal aspect ratio
    earth: whether or not to plot
    show: call mpl.show() afterward
    mask_nan: mask nan values in fld.data
    mod: list of scaling factors for the coords
    plot_opts: string of options
    colorbar: is evaluated boolean to decide whether to plot a colorbar, if
    it is a dict, it's passed to colorbar as keyword args
    (note: bool({}) == False) """
    if fld.nr_sdims != 2:
        raise RuntimeError("I will only contour a 2d field")

    # THIS IS BACKWARD, on account of the convention for
    # Coordinates where z, y, x is used since that is how
    # xdmf data is
    namey, namex = fld.crds.axes # fld.crds.get_culled_axes()

    # pcolor mesh uses node coords, and cell data, if we have
    # node data, fake it by using cell centered coords and
    # trim the edges of the data... maybe i should just be
    # extapolating the crds and keeping the edges...
    if style in ["pcolormesh", "pcolor"]:
        if fld.iscentered("Node"):
            X, Y = fld.get_crds_cc((namex, namey), shaped=True)
            logging.info("pcolormesh on node centered field... "
                         "trimming the edges")
            # FIXME: this is a little fragile with 2d stuff
            dat = fld.data[1:-1, 1:-1]
        elif fld.iscentered("Cell"):
            X, Y = fld.get_crds_nc((namex, namey), shaped=True)
            dat = fld.data
    else:
        dat = fld.data
        if fld.iscentered("Node"):
            X, Y = fld.get_crds_nc[(namex, namey)]
        elif fld.iscentered("Cell"):
            X, Y = fld.get_crds_cc[(namex, namey)]

    if mod:
        X *= mod[0]
        Y *= mod[1]

    # print(x.shape, y.shape, fld.data.shape)
    if mask_nan:
        dat = np.ma.masked_where(np.isnan(dat), dat)

    if not ax:
        ax = plt.gca()
    if equalaxis:
        ax.axis('equal')
    ax = _apply_parse_opts(plot_opts, fld, kwargs, ax)

    earth = kwargs.pop("earth", False)

    if rotate_plot:
        X, Y = Y.T, X.T
        dat = dat.T
        namex, namey = namey, namex

    if style == "pcolormesh":
        p = ax.pcolormesh(X, Y, dat, **kwargs)
    elif style == "contour":
        p = ax.contour(X, Y, dat, **kwargs)
    elif style == "contourf":
        p = ax.contourf(X, Y, dat, **kwargs)
    elif style == "pcolor":
        p = ax.pcolor(X, Y, dat, **kwargs)
    else:
        raise RuntimeError("I don't understand {0} 2d plot style".format(style))

    # figure out the colorbar...
    if colorbar:
        if not isinstance(colorbar, dict):
            colorbar = {}
        # unless otherwise specified, use_gridspec for
        if has_colorbar_gridspec and not "use_gridspec" in colorbar:
            colorbar["use_gridspec"] = True
        # ok, this way to pass options to colorbar is bad!!!
        # but it's kind of the cleanest way to affect the colorbar?
        cbar = plt.colorbar(p, **colorbar) #pylint: disable=W0142
        cbar.set_label(fld.name)
    else:
        cbar = None

    plt.xlabel(namex)
    plt.ylabel(namey)

    # _apply_acts(acts)

    if earth:
        plot_earth(fld)
    if show:
        mplshow()
    return p, cbar

def plot1d_field(fld, ax=None, plot_opts=None, show=False, **kwargs):
    namex, = fld.crds.axes
    if fld.iscentered("Node"):
        x = fld.get_crd_nc(namex)
    elif fld.iscentered("Cell"):
        x = fld.get_crd_cc(namex)

    ax = _apply_parse_opts(plot_opts, fld, kwargs, ax)
    p = plt.plot(x, fld.data, **kwargs)
    plt.xlabel(namex)
    plt.ylabel(fld.name)
    # _apply_acts(acts)

    if show:
        mplshow()
    return p, None

def plot_streamlines(lines, topology=None, ax=None, show=True, equal=False,
                     **kwargs):
    if not ax:
        ax = plt.gca(projection='3d')

    if topology is not None:
        if isinstance(topology, field.Field):
            topology = topology.data.reshape(-1)
        if not "color" in kwargs:
            topo_color = True
    else:
        topo_color = False

    for i, line in enumerate(lines):
        line = np.array(line)
        z = line[0]
        y = line[1]
        x = line[2]

        if topo_color:
            kwargs["color"] = color_from_topology(topology[i])
        p = ax.plot(x, y, z, **kwargs)
    if equal:
        ax.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    if show:
        plt.show()
    return p, None

def plot_streamlines2d(lines, symmetry_dir, topology=None, ax=None, show=False,
                       equal=False, rotate_plot=False, **kwargs):
    """ print streamlines given as a list of lines which are ndarrays with
    dimension (3, npts). symmetry_dir says which dimension to ignore, so that
    the lines are just parallel projected onto a plane. kwargs are passed to
    plt.plot(...). topology can be an integer array (or field) of
    size = nr_lines to color the lines by topology """
    if not ax:
        ax = plt.gca()
    p = None

    if topology is not None:
        if isinstance(topology, field.Field):
            topology = topology.data.reshape(-1)
        if not "color" in kwargs:
            topo_color = True
    else:
        topo_color = False

    for i, line in enumerate(lines):
        line = np.array(line)
        if symmetry_dir.lower() == "x":
            x = line[1]
            y = line[0]
        elif symmetry_dir.lower() == "y":
            x = line[2]
            y = line[0]
        elif symmetry_dir.lower() == "z":
            x = line[2]
            y = line[1]
        else:
            raise ValueError("symmetry_dir should be x, y, or z")

        if rotate_plot:
            x, y = y, x

        if topo_color:
            kwargs["color"] = color_from_topology(topology[i])
        p = ax.plot(x, y, **kwargs)

    if show:
        plt.show()
    return p, None

def scatter_3d(points, c='b', ax=None, show=True, equal=False, **kwargs):
    """ c should be an array of values to use to color the points,
    a la pyplot.scatter """
    if not ax:
        ax = plt.gca(projection='3d')

    z = points[0]
    y = points[1]
    x = points[2]
    p = ax.scatter(x, y, z, c=c, **kwargs)
    if equal:
        ax.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    if show:
        plt.show()
    return p, None


def mplshow():
    # do i need to do anything special before i show?
    # can't think of anything at this point...
    plt.show()

def tighten():
    """ tightens the layout so that axis labels dont get plotted over """
    try:
        plt.tight_layout()
    except AttributeError:
        logging.warn("No matplotlib tight layout support")

def plot_earth(fld, axis=None, scale=1.0, rot=0,
               daycol='w', nightcol='k', crds="mhd"):
    """ crds = "mhd" (Jimmy crds) or "gsm" (GSM crds)... gsm is the same
    as mhd + rot=180. earth_plane is a string in the format 'y=0.2', this
    says what the 3rd.nr_sdimsension is and sets the radius that the earth should
    be """
    import matplotlib.patches as mpatches

    # take only the 1st reduced.nr_sdims... this should just work
    try:
        plane, value = fld.info["reduced"][0]
    except KeyError:
        logging.error("No reduced dims in the field, i don't know what 2d \n "
                      "plane, we're in and can't figure out the size of earth.")
        return None

    if value**2 >= scale**2:
        return None
    radius = np.sqrt(scale**2 - value**2)

    if not axis:
        axis = plt.gca()

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
