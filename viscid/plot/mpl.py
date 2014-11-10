"""Convenience module for making matplotlib plots

Your best friend in this module is the :meth:`plot` function, but the
best reference for all quirky options is :meth:`plot2d_field`.
"""
from __future__ import print_function
from distutils.version import LooseVersion

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.mplot3d import Axes3D #pylint: disable=W0611
try:
    from mpl_toolkits.basemap import Basemap
    _HAS_BASEMAP = True
except ImportError:
    _HAS_BASEMAP = False

from viscid import logger
from viscid import field
from viscid import coordinate
from viscid.calculator import calc
from viscid.calculator.topology import color_from_topology
from viscid.compat import string_types
# from viscid import vutil

__mpl_ver__ = matplotlib.__version__
has_colorbar_gridspec = LooseVersion(__mpl_ver__) > LooseVersion("1.1.1")

def plot(fld, selection=None, **kwargs):
    """Plot a field by dispatching to the most appropiate funciton

    * If fld has 1 spatial dimensions, call
      :meth:`plot1d_field(fld[selection], **kwargs)`
    * If fld has 2 spatial dimensions, call
      :meth:`plot2d_field(fld[selection], **kwargs)`
    * If fld is 2-D and has spherical coordinates (as is the case for
      ionospheric fields), try to use :meth:`plot2d_mapfield` which
      uses basemap to make its axes.

    Parameters:
        selection (optional): something that describes a field slice
        kwargs: passed as keyword arguments to the actual plotting
            function

    Returns:
        tuple: (plot, colorbar)
            plot: matplotlib plot object
            colorbar: matplotlib colorbar object

    See Also:
        * :meth:`plot1d_field`: target for 1d fields
        * :meth:`plot2d_mapfield`: target for 2d spherical fields
        * :meth:`plot2d_field`: target for 2d fields


    Note:
        Field slices are done using "slice_reduce", meaning extra
        dimensions are reduced out.
    """
    if isinstance(fld, field.ScalarField):
        fld = fld.slice_reduce(selection)

        if not isinstance(fld, field.ScalarField):
            raise ValueError("Selection '{0}' sliced away too many "
                             "dimensions".format(selection))
        if fld.nr_sdims == 1:
            return plot1d_field(fld, **kwargs)
        elif fld.nr_sdims == 2:
            if fld.is_spherical():
                return plot2d_mapfield(fld, **kwargs)
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
    actions = []

    if not axis:
        axis = plt.gca()

    for opt in plot_opts:
        if opt[0] == "lin":
            opt = [float(o) if i > 0 else o for i, o in enumerate(opt)]
            actions.append([axis.set_xscale, ["linear"]])
            actions.append([axis.set_yscale, ["linear"]])

            # scale will be centered around 0
            if len(opt) == 2 and float(opt[1]) == 0.0:
                absmax = calc.abs_max(fld)
                opt = [opt[0], -1.0 * absmax, 1.0 * absmax]

            if fld.nr_sdims == 1:
                actions.append([axis.set_ylim, opt[1:]])
            elif fld.nr_sdims == 2:
                # plt.normalize is deprecated
                # kwargs["norm"] = plt.normalize(*opt[1:])
                kwargs["norm"] = Normalize(*opt[1:])

        elif opt[0] == "log":
            opt = [float(o) if i > 0 else o for i, o in enumerate(opt)]
            actions.append([axis.set_xscale, ["linear"]])

            if fld.nr_sdims == 1:
                actions.append([axis.set_yscale, ["log"]])
                actions.append([axis.set_ylim, opt[1:]])
            elif fld.nr_sdims == 2:
                actions.append([axis.set_yscale, ["linear"]])
                kwargs["norm"] = LogNorm(*opt[1:])

        elif opt[0] == "loglog":
            opt = [float(o) if i > 0 else o for i, o in enumerate(opt)]
            actions.append([axis.set_xscale, ["log"]])
            actions.append([axis.set_yscale, ["log"]])
            if fld.nr_sdims == 2:
                kwargs["norm"] = LogNorm(*opt[1:])

        elif opt[0] == "x":
            opt = [float(o) if i > 0 else o for i, o in enumerate(opt)]
            # axis.set_xlim(*opt[1:])
            actions.append([axis.set_xlim, opt[1:]])

        elif opt[0] == "y":
            opt = [float(o) if i > 0 else o for i, o in enumerate(opt)]
            # axis.set_ylim(*opt[1:])
            actions.append([axis.set_ylim, opt[1:]])

        elif opt[0] == "own":
            logger.warn("own axis doesn't seem to work yet...")

        elif opt[0] == "ownx":
            logger.warn("own axis doesn't seem to work yet...")

        elif opt[0] == "owny":
            logger.warn("own axis doesn't seem to work yet...")

        else:
            val = "_".join(opt[1:])
            if val == "" or val.lower() == "true":
                val = True
            elif val.lower() == "false":
                val = False
            else:
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
            kwargs[opt[0]] = val
            # logger.warn("Unknown plot option ({0}) didn't parse "
            #              "correctly".format(opt[0]))


    # things that i just want to be automagic...
    # use seismic cmap if the data looks centered around 0
    if "norm" in kwargs and "cmap" not in kwargs:
        norm = kwargs["norm"]
        if norm.vmin and norm.vmax and np.abs(norm.vmax + 1.0*norm.vmin) < 1e-4:
            kwargs["cmap"] = plt.get_cmap('seismic')

    return axis, actions

def _apply_actions(acts):
    for act in acts:
        act[0](*act[1])

def plot2d_field(fld, style="pcolormesh", ax=None, plot_opts=None,
                 colorbar=True, do_labels=True, show=False,
                 action_ax=None, scale=None, mod=None, extra_args=None,
                 **kwargs):
    """Plot a 2D Field using pcolormesh, contour, etc.

    Parameters:
        style (str, optional): One of pcolormesh, pcolor, contour, or
            contourf
        ax (matplotlib axis, optional): Plot in a specific axis object
        plot_opts (str, optional): comma separated string of additional
            options with underscore separated argument options. They are
            summarized as follows...

            ================  ======================================
            Option            Description
            ================  ======================================
            lin               Use a linear scale for data with two
                              optional sub-options giving a range.
                              ``lin_0`` has the special meaning to
                              make the range symmetric about 0
            log               Use a log scale, with two sub-options
                              for the range
            loglog            same as log, but also make coordinates
                              log scaled
            x                 Set limits of x axis using 2 manditory
                              sub-options
            y                 Set limits of y axis using 2 manditory
                              sub-options
            grid [#f1]_       plot lines showing grid cells
            earth [#f1]_      plot a black and white Earth
            equalaxis [#f1]_  Use 1:1 aspect ratio for grid cells
                              (True by default)
            flip_plot [#f1]_  flip x and y axes (2d fields only)
            masknan           if true, nan values are masked, so
                              black if pcolor, or missing for
                              countours
            ================  ======================================

            .. [#f1] These options can be given as kwargs

            If a plot_opt is not understood, it is added to kwargs.
            Some plot_opt examples are:
            * ``lin_-300_300,earth``
            * ``log,x_-3_30,y_-10_10,cmap_afmhot``
            * ``lin_0,x_-10_20,grid,earth``
        kwargs: Some other keyword arguments are understood and
            described below, and all others are passed as keyword
            arguments to the matplotlib plotting functions. This way,
            one can pass arguments like cmap and the like.

    Other Parameters:
        colorbar (bool or dict): If dict, the items are passed to
            plt.colorbar as keyword arguments
        levels (int): Number of contours to follow (default: 10)
        do_labels: automatically label x/y axes and color bar
        show_grid (bool): Plot lines showing grid cells
        earth (bool): Plot a black and white circle representing Earth
            showing the sunlit half
        show (bool, optional): Call pyplot.show before returning
        action_ax: axis on which to call matplotlib functions... I
            don't even remember why this was necessary
        scale (float): multiply field by scalar before plotting, useful
            for changing units
        mod (list of floats): DEPRECATED, scale x and y axes by some
            factor
        extra_args (list): DEPRECATED, was used to pass args to
            matplotlib functions, like contour levels, but there
            is probably a better way to give these options
    """
    if fld.nr_sdims != 2:
        raise RuntimeError("I will only contour a 2d field")

    if extra_args is None:
        extra_args = []

    if not ax:
        ax = plt.gca()
    if action_ax is None:
        action_ax = ax

    # parse plot_opts and apply them
    ax, actions = _apply_parse_opts(plot_opts, fld, kwargs, ax)

    colorbar = kwargs.pop("colorbar", colorbar)
    if colorbar:
        if not isinstance(colorbar, dict):
            colorbar = {}
    else:
        colorbar = None

    # make customizing plot type from command line possible
    style = kwargs.pop("style", style)
    earth = kwargs.pop("earth", False)
    equalaxis = kwargs.pop("equalaxis", True)
    flip_plot = kwargs.pop("flip_plot", False)
    masknan = kwargs.pop("masknan", False)

    show_grid = kwargs.pop("show_grid", False)
    show_grid = kwargs.pop("g", show_grid)
    if show_grid:
        kwargs["edgecolors"] = 'k'
        kwargs["linewidths"] = 0.2
        kwargs["antialiased"] = True

    # THIS IS BACKWARD, on account of the convention for
    # Coordinates where z, y, x is used since that is how
    # xdmf data is
    namey, namex = fld.crds.axes # fld.crds.get_culled_axes()

    # pcolor mesh uses node coords, and cell data, if we have
    # node data, fake it by using cell centered coords and
    # trim the edges of the data... maybe i should just be
    # extapolating the crds and keeping the edges...
    if style in ["pcolormesh", "pcolor"]:
        fld = fld.as_cell_centered()
        X, Y = fld.get_crds_nc((namex, namey))
    else:
        if fld.iscentered("Node"):
            X, Y = fld.get_crds_nc((namex, namey))
        elif fld.iscentered("Cell"):
            X, Y = fld.get_crds_cc((namex, namey))

    if kwargs.get('latlon', False):
        # translate latitude from 0..180 to -90..90
        X, Y = np.meshgrid(X, 90 - Y)

    dat = fld.data
    if scale is not None:
        dat *= scale

    if mod:
        X *= mod[0]
        Y *= mod[1]

    # print(x.shape, y.shape, fld.data.shape)
    if masknan:
        dat = np.ma.masked_where(np.isnan(dat), dat)

    if equalaxis:
        ax.axis('equal')
    _apply_actions(actions)

    # ok, here's some raw hackery for contours
    if style in ["contourf", "contour"]:
        if len(extra_args) > 0:
            n = extra_args[0]
        else:
            n = 10
        n = int(kwargs.pop("levels", n))
        extra_args = [n] + extra_args[1:]

        if "norm" in kwargs:
            norm = kwargs["norm"]
            norm.autoscale_None(dat)

            if isinstance(norm, LogNorm):
                extra_args[0] = np.logspace(np.log10(norm.vmin),
                                            np.log10(norm.vmax), n)
                if colorbar is not None and not "ticks" in colorbar:
                    colorbar["ticks"] = matplotlib.ticker.LogLocator()
            elif isinstance(norm, Normalize):
                extra_args[0] = np.linspace(norm.vmin, norm.vmax, n)
            else:
                raise ValueError("I should never be here")

    if flip_plot:
        X, Y = Y.T, X.T
        dat = dat.T
        namex, namey = namey, namex

    if style == "pcolormesh":
        p = action_ax.pcolormesh(X, Y, dat, *extra_args, **kwargs)
    elif style == "contour":
        p = action_ax.contour(X, Y, dat, *extra_args, **kwargs)
        if "colors" in kwargs:
            colorbar = None
    elif style == "contourf":
        p = action_ax.contourf(X, Y, dat, *extra_args, **kwargs)
    elif style == "pcolor":
        p = action_ax.pcolor(X, Y, dat, *extra_args, **kwargs)
    else:
        raise RuntimeError("I don't understand {0} 2d plot style".format(style))

    try:
        if masknan:
            p.get_cmap().set_bad(masknan)
        else:
            raise ValueError()
    except ValueError:
        p.get_cmap().set_bad('k')

    # figure out the colorbar...
    if colorbar is not None:
        # unless otherwise specified, use_gridspec for
        if not "use_gridspec" not in colorbar:
            colorbar["use_gridspec"] = True
        # ok, this way to pass options to colorbar is bad!!!
        # but it's kind of the cleanest way to affect the colorbar?
        if masknan and dat.mask.all():
            cbar = None
        else:
            cbar = plt.colorbar(p, **colorbar)
            if do_labels:
                cbar.set_label(fld.pretty_name)
    else:
        cbar = None

    if do_labels:
        plt.xlabel(namex)
        plt.ylabel(namey)

    # _apply_acts(acts)

    if earth:
        plot_earth(fld, axis=action_ax)
    if show:
        mplshow()
    return p, cbar

def _mlt_labels(longitude):
    return "{0:g}".format(longitude * 24.0 / 360.0)

def plot2d_mapfield(fld, projection="polar", hemisphere="north",
                    drawcoastlines=False, show_grid=True,
                    lon_0=0.0, lat_0=None, bounding_lat=40.0,
                    title=True, label_lat=True, label_mlt=True, **kwargs):
    """Plot data on a map projection of a sphere

    The default projection is polar, but any other basemap projection
    can be used.

    Note:
        Parameters are in degrees, but if the projection is 'polar',
        then the plot is actually made in radians, which is important
        if you want to annotate a plot.

    Parameters:
        fld (Field): field whose crds are spherical
        projection (string): 'polar' or Basemap projection to use
        hemisphere (string): 'north' or 'south'
        drawcoastlines (bool): If projection is a basemap projection,
            then draw coastlines, pretty cool, but not actually useful.
            NOTE: coastlines do NOT reflect UT time; London is always
            at midnight.
        show_grid (bool): draw grid lines
        lon_0 (float): center longitude (basemap projections only)
        lat_0 (fload): center latitude (basemap projections only)
        bounding_lat (float): bounding latitude in degrees from the
            nearest pole (not for all projections)
        title (bool, str): put a specific title on the plot, or with
            if a boolean, use the field's pretty_name.
        label_lat (bool, str): label latitudes at 80, 70, 60 degrees
            with sign indicating northern / southern hemisphere.
            if label_lat is 'from_pole', then the labels are 10, 20,
            30 for both hemispheres. Note that basemap projections
            won't label latitudes unless they hit the edge of the plot.
        label_mlt (bool): label magnetic local time
        kwargs: either mapping keyword arguments, or those that
            should be passed along to `plot2d_field`

    See Also:
        * :meth:`plot2d_field`: `plot2d_mapfield` basically just wraps
          this function setting up a Basemap first
    """
    hemisphere = hemisphere.lower().strip()
    if hemisphere == "north":
        # def_projection = "nplaea"
        # def_boundinglat = 40.0
        latlabel_arr = np.linspace(50.0, 80.0, 4)
    elif hemisphere == "south":
        # def_projection = "splaea"
        # def_boundinglat = -40.0
        # FIXME: should I be doing this?
        if bounding_lat > 0.0:
            bounding_lat *= -1.0
        latlabel_arr = -1.0 * np.linspace(50.0, 80.0, 4)
    else:
        raise ValueError("hemisphere is either 'north' or 'south'")

    # boundinglat = kwargs.pop("boundinglat", def_boundinglat)
    # lon_0 = kwargs.pop("lon_0", 0.0)
    # lat_0 = kwargs.pop("lat_0", None)
    # drawcoastlines = kwargs.pop("drawcoastlines", False)
    ax = kwargs.get("ax", None)

    if projection != "polar" and not _HAS_BASEMAP:
        print("NOTE: install the basemap for the desired spherical "
              "projection; falling back to matplotlib's polar plot.")
        projection = "polar"

    if projection == "polar":
        if LooseVersion(__mpl_ver__) < LooseVersion("1.1"):
            raise RuntimeError("polar plots are annoying for matplotlib < ",
                               "version 1.1. Update your matplotlib and "
                               "profit.")

        absboundinglat = np.abs(bounding_lat)

        if ax is None:
            ax = plt.gca()
        if not hasattr(ax, "set_thetagrids"):
            ax = plt.subplot(*ax.get_geometry(), projection='polar')
            logger.warn("Clobbering axis for subplot {0}; please give a polar "
                        "axis to plot2d_mapfield if you indend to use it "
                        "later.".format(ax.get_geometry()))

        if hemisphere == "north":
            sl_fld = fld["lat=:{0}".format(absboundinglat)]
            maxlat = sl_fld.get_crd_nc('lat')[-1]
        elif hemisphere == "south":
            sl_fld = fld["lat={0}:".format(180.0 - absboundinglat)]["lat=::-1"]
            maxlat = 180.0 - sl_fld.get_crd_nc('lat')[-1]

        lat, lon = sl_fld.get_crds_nc(['lat', 'lon'])
        new_lat = (np.pi / 180.0) * np.linspace(0.0, maxlat, len(lat))
        # FIXME: Matt's code had a - 0.5 * (lon[1] - lon[0]) here...
        # I'm omiting it
        ax.set_theta_offset(-90 * np.pi / 180.0)
        # new_lon = (lon - 90.0) * np.pi / 180.0
        new_lon = lon * np.pi / 180.0
        new_crds = coordinate.wrap_crds("nonuniform_spherical",
                                        [('lat', new_lat), ('lon', new_lon)])
        new_fld = fld.wrap(sl_fld.data, context=dict(crds=new_crds))

        # print(fld.get_crds())

        kwargs['ax'] = ax
        kwargs['action_ax'] = ax
        kwargs['do_labels'] = False
        kwargs['equalaxis'] = False

        ret = plot2d_field(new_fld, show_grid=False, **kwargs)
        if title:
            if not isinstance(title, string_types):
                title = new_fld.pretty_name
            plt.title(title)
        if show_grid:
            ax.grid(True)

            mlt_grid_pos = (0, 45, 90, 135, 180, 225, 270, 315)
            mlt_labels = (24, 3, 6, 9, 12, 15, 18, 21)
            if not label_mlt:
                mlt_labels = []
            ax.set_thetagrids(mlt_grid_pos, mlt_labels)

            abs_grid_dr = 10
            grid_dr = abs_grid_dr * np.sign(bounding_lat)
            lat_grid_pos = np.arange(abs_grid_dr, absboundinglat, abs_grid_dr)
            lat_labels = np.arange(abs_grid_dr, absboundinglat, abs_grid_dr)
            if label_lat == "from_pole":
                lat_labels = ["{0:g}".format(l) for l in lat_labels]
            elif label_lat:
                if hemisphere == 'north':
                    lat_labels = 90 - lat_labels
                else:
                    lat_labels = -90 + lat_labels
                lat_labels = ["{0:g}".format(l) for l in lat_labels]
            else:
                lat_labels = []
            ax.set_rgrids((np.pi / 180.0) * lat_grid_pos, lat_labels)
        else:
            ax.grid(False)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        return ret

    else:
        m = Basemap(projection=projection, lon_0=lon_0, lat_0=lat_0,
                    boundinglat=bounding_lat, ax=ax)
        kwargs['latlon'] = True
        kwargs['action_ax'] = m
        kwargs['do_labels'] = False
        kwargs['equalaxis'] = False
        ret = plot2d_field(fld, **kwargs)
        if show_grid:
            if label_lat:
                lat_lables = [1, 1, 1, 1]
            else:
                lat_lables = [0, 0, 0, 0]
            m.drawparallels(latlabel_arr, labels=lat_lables,
                            linewidth=0.25)

            if label_mlt:
                mlt_labels = [1, 1, 1, 1]
            else:
                mlt_labels = [0, 0, 0, 0]
            m.drawmeridians(np.linspace(360.0, 0.0, 8, endpoint=False),
                            labels=mlt_labels, fmt=_mlt_labels,
                            linewidth=0.25)
        if drawcoastlines:
            m.drawcoastlines(linewidth=0.25)
        return ret


def plot1d_field(fld, ax=None, plot_opts=None, show=False,
                 do_labels=True, action_ax=None, **kwargs):
    """Plot a 1D Field using lines

    Parameters:
        fld: Field to plot

    See Also:
        * :meth:`plot2d_field`: Describes plot_opts, and all other
          keyword arguments
    """
    namex, = fld.crds.axes
    if fld.iscentered("Node"):
        x = fld.get_crd_nc(namex)
    elif fld.iscentered("Cell"):
        x = fld.get_crd_cc(namex)

    ax, actions = _apply_parse_opts(plot_opts, fld, kwargs, ax)
    if action_ax is None:
        action_ax = ax
    masknan = kwargs.pop('masknan', False)

    dat = fld.data
    if masknan:
        dat = np.ma.masked_where(np.isnan(dat), dat)

    if "label" not in kwargs:
        kwargs["label"] = fld.pretty_name

    p = action_ax.plot(x, dat, **kwargs)
    if do_labels:
        plt.xlabel(namex)
        plt.ylabel(fld.pretty_name)
    _apply_actions(actions)

    if show:
        mplshow()
    return p, None

def plot_streamlines(lines, topology=None, ax=None, show=True, equal=False,
                     **kwargs):
    """Plot lines on a matplotlib 3D plot, optionally colored by value

    Parameters:
        lines (list): A set of N lines. Elements should have the shape
            3xP where 3 is the axes zyx (in that order) and P is the
            number of points in the line. As an ndarray, the required
            shape is Nx3xP.
        topology (optional): Value used to color the lines. Should have
            length N.
        ax (matplotlib axis, optional): axis on which to plot
        show (bool, optional): plt.show() before returning
        equal (bool, optional): set 1:1 aspect ratio on axes
    """
    if not ax:
        ax = plt.gca(projection='3d')

    if "color" not in kwargs and topology is not None:
        if isinstance(topology, field.Field):
            topology = topology.data.reshape(-1)
        topo_color = True
    else:
        topo_color = False

    for i, line in enumerate(lines):
        line = np.array(line, copy=False)
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

def plot_streamlines2d(lines, symdir=None, topology=None, ax=None,
                       show=False, flip_plot=False, **kwargs):
    """Project 3D lines onto a 2D plot

    Parameters:
        lines (list): A set of N lines. Elements should have the shape
            3xP where 3 is the axes zyx (in that order) and P is the
            number of points in the line. As an ndarray, the required
            shape is Nx3xP.
        symdir (str, optional): one of xyz for the plane on which to
            orthogonally project lines. Not needed if lines are shaped
            Nx2xP.
        topology (optional): Value used to color the lines. Should have
            length N.
        ax (matplotlib axis, optional): axis on which to plot
        show (bool, optional): plt.show() before returning
        flip_plot (bool, optional): swap plot's x/y axes
    """
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
        line = np.array(line, copy=False)
        if len(line) == 2:
            x = line[1]
            y = line[0]
        elif symdir.lower() == "x":
            x = line[1]
            y = line[0]
        elif symdir.lower() == "y":
            x = line[2]
            y = line[0]
        elif symdir.lower() == "z":
            x = line[2]
            y = line[1]
        else:
            raise ValueError("For 3d lines, symdir should be x, y, or z")

        if flip_plot:
            x, y = y, x

        if topo_color:
            kwargs["color"] = color_from_topology(topology[i])
        p = ax.plot(x, y, **kwargs)

    if show:
        plt.show()
    return p, None

def plot2d_quiver(fld, symdir, downscale=1, **kwargs):
    """Put quivers on a 2D plot

    Parameters:
        fld: Vector field to plot
        symdir (str): One of xyz for direction orthogonal to 2D plane
        downscale (int, optional): only quiver every Nth grid cell
        kwargs: passed along to :meth:`plt.quiver`

    Note:
        There are some edge cases where downscale doesn't work.
    """
    # FIXME: with dowscale != 1, this reveals a problem when slice and
    # downscaling a field; i think this is a prickley one
    vx, vy, vz = fld.component_views()
    x, y = fld.get_crds_cc(shaped=True)
    if symdir.lower() == "x":
        # x, y = ycc, zcc
        pvx, pvy = vy, vz
    elif symdir.lower() == "y":
        # x, y = xcc, zcc
        pvx, pvy = vx, vz
    elif symdir.lower() == "z":
        # x, y = xcc, ycc
        pvx, pvy = vx, vy
    X, Y = np.meshgrid(y, x)
    if downscale != 1:
        X = X[::downscale]
        Y = Y[::downscale]
        pvx = pvx[::downscale]
        pvy = pvy[::downscale]
    # print(X.shape, Y.shape, pvx.shape, pvy.shape)
    return plt.quiver(X, Y, pvx, pvy, **kwargs)

def scatter_3d(points, c='b', ax=None, show=True, equal=False, **kwargs):
    """Plot scattered points on a matplotlib 3d plot

    Parameters:
        points: something shaped 3xN for N points, where 3 are the
            zyx cartesian directions in that order
        c (str, optional): color (in matplotlib format)
        ax (matplotlib Axis, optional): axis on which to plot (should
            be a 3d axis)
        show (bool, optional): show
        kwargs: passed along to :meth:`plt.statter`
    """
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
    """Calls :meth:`matplotlib.pyplot.show()`"""
    # do i need to do anything special before i show?
    # can't think of anything at this point...
    plt.show()

def tighten(**kwargs):
    """Calls `matplotlib.pyplot.tight_layout(**kwargs)`"""
    try:
        plt.tight_layout(**kwargs)
    except AttributeError:
        logger.warn("No matplotlib tight layout support")

def plot_earth(plane_spec, axis=None, scale=1.0, rot=0,
               daycol='w', nightcol='k', crd_system="mhd",
               zorder=1):
    """Plot a black and white Earth to show sunward direction

    Parameters:
        plane_spec: Specifies the plane for determining sunlit portion.
            This can be a :class:`viscid.field.Field` object to try to
            auto-discover the plane and crd_system, or it can be a
            string like "y=0".
        axis (matplotlib Axis): axis on which to plot
        scale (float, optional): scale of earth
        rot (float, optional): Rotation of day/night side... I forget all
            the details :(
        daycol (str, optional): color of dayside (matplotlib format)
        nightcol (str, optional): color of nightside (matplotlib format)
        crd_system (str, optional): 'mhd' or 'gse', can usually be
            deduced from plane_spec if it's a Field instance.
    """
    import matplotlib.patches as mpatches

    # this is kind of a hacky way to
    if isinstance(plane_spec, field.Field):
        crd_system = plane_spec.info.get("crd_system", crd_system)

        # take only the 1st reduced.nr_sdims... this should just work
        try:
            plane, value = plane_spec.deep_meta["reduced"][0]
        except KeyError:
            logger.error("No reduced dims in the field, i don't know what 2d \n "
                          "plane, we're in and can't figure out the size of earth.")
            return None
    else:
        plane, value = [s.strip() for s in plane_spec.split("=")]
        value = float(value)

    if value**2 >= scale**2:
        return None
    radius = np.sqrt(scale**2 - value**2)

    if not axis:
        axis = plt.gca()

    if crd_system == "gse":
        rot = 180

    if plane == 'y' or plane == 'z':
        axis.add_patch(mpatches.Wedge((0, 0), radius, 90 + rot, 270 + rot,
                                      ec=nightcol, fc=daycol, zorder=zorder))
        axis.add_patch(mpatches.Wedge((0, 0), radius, 270 + rot, 450 + rot,
                                      ec=nightcol, fc=nightcol, zorder=zorder))
    elif plane == 'x':
        if value < 0:
            axis.add_patch(mpatches.Circle((0, 0), radius, ec=nightcol,
                                           fc=daycol, zorder=zorder))
        else:
            axis.add_patch(mpatches.Circle((0, 0), radius, ec=nightcol,
                                           fc=nightcol, zorder=zorder))
    return None

##
## EOF
##
