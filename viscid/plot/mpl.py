"""Convenience module for making matplotlib plots

Your best friend in this module is the :meth:`plot` function, but the
best reference for all quirky options is :meth:`plot2d_field`.

Note: you can't set rc parameters for this module
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

from viscid.parsers import pyeval
from viscid import logger
from viscid.compat import string_types
from viscid import field
from viscid import coordinate
from viscid.calculator.topology import color_from_topology
from viscid.plot import vseaborn

__mpl_ver__ = matplotlib.__version__
has_colorbar_gridspec = LooseVersion(__mpl_ver__) > LooseVersion("1.1.1")
vseaborn.activate_from_viscid()


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
    fld = fld.slice_reduce(selection)

    if not hasattr(fld, "blocks"):
        raise ValueError("Selection '{0}' sliced away too many "
                         "dimensions".format(selection))
    block0 = fld.blocks[0]
    nr_sdims = block0.nr_sdims
    if nr_sdims == 1:
        return plot1d_field(fld, **kwargs)
    elif nr_sdims == 2:
        is_spherical = block0.is_spherical()
        if is_spherical:
            return plot2d_mapfield(fld, **kwargs)
        return plot2d_field(fld, **kwargs)
    else:
        raise ValueError("mpl can only do 1-D or 2-D fields")

def plot_opts_to_kwargs(plot_opts, plot_kwargs):
    """Turn plot options from string to items in plot_kwargs

    The Reason for this to to be able to specify arbitrary plotting
    kwargs from the command line

    Args:
        plot_opts (str): plot kwargs as string
        plot_kwargs (dict): kwargs to be popped by hand or passed to
            plotting function

    Returns:
        None, the plot_opts are stuffed into plot_kwargs
    """
    if not plot_opts:
        return

    plot_opts = plot_opts.strip()
    if plot_opts[0] == '{' and plot_opts[-1] == '}':
        try:
            import yaml
            d = yaml.load(plot_opts)
            # if an option is given without a value, Yaml defaults to
            # None, but it was probably a flag, so turn None -> True
            for k in list(d.keys()):
                if d[k] is None:
                    d[k] = True
            plot_kwargs.update(d)
        except ImportError:
            raise ImportError("You gave plot options in YAML syntax, but "
                              "PyYaml is not installed. Either install PyYaml "
                              "or use the old comma/underscore syntax.")
    else:
        plot_opts = plot_opts.split(",")

        for opt in plot_opts:
            opt = opt.replace("=", "_").split("_")
            opt[0] = opt[0].strip()
            opt[1:] = [pyeval.parse(o) for o in opt[1:]]
            if len(opt) == 0 or opt == ['']:
                continue
            elif len(opt) == 1:
                plot_kwargs[opt[0]] = True
            elif len(opt) == 2:
                plot_kwargs[opt[0]] = opt[1]
            else:
                # if opt[1:] are all strings, re-combine them since some legit
                # options have underscores in them, like reversed cmap names
                try:
                    opt[1:] = "_".join(opt[1:])
                except TypeError:
                    pass
                plot_kwargs[opt[0]] = opt[1:]


def _extract_actions_and_norm(axis, plot_kwargs, defaults=None):
    """
    Some plot options will want to call a function after the plot is
    made, like setting xlim and the like. Those are 'actions'.

    Args:
        axis: matplotlib axis for current plot
        plot_kwargs (dict): kwargs dict containing all the options for
            the current plot
        defaults (dict): default values to merge plot_kwargs into

    Returns:
        (actions, norm_dict)

        actions: list of tuples... the first erement of the tuple is
            a function to call, while the 2nd is a list of arguments
            to unpack when calling that function

        norm_dict: will look like {'crdscale': 'lin'|'log',
                                   'vscale': 'lin'|'log',
                                   'clim': [None|number, None|number],
                                   'symetric': True|False}
    """
    for k, v in defaults.items():
        if k not in plot_kwargs:
            plot_kwargs[k] = v

    actions = []
    if not axis:
        axis = plt.gca()

    if "x" in plot_kwargs:
        actions.append((axis.set_xlim, plot_kwargs.pop('x')))
    if "y" in plot_kwargs:
        actions.append((axis.set_xlim, plot_kwargs.pop('y')))
    if "own" in plot_kwargs:
        opt = plot_kwargs.pop('own')
        logger.warn("own axis doesn't seem to work yet...")
    if "ownx" in plot_kwargs:
        opt = plot_kwargs.pop('ownx')
        logger.warn("own axis doesn't seem to work yet...")
    if "owny" in plot_kwargs:
        opt = plot_kwargs.pop('owny')
        logger.warn("own axis doesn't seem to work yet...")
    if "equalaxis" in plot_kwargs:
        if plot_kwargs.pop('equalaxis'):
            actions.append((axis.axis, 'equal'))

    norm_dict = {'crdscale': 'lin',
                 'vscale': 'lin',
                 'clim': [None, None],
                 'symetric': False
                }

    if plot_kwargs.pop('logscale', False):
        norm_dict['vscale'] = 'log'

    if "clim" in plot_kwargs:
        clim = plot_kwargs.pop('clim')
        norm_dict['clim'][:len(clim)] = clim

    sym = plot_kwargs.pop('symetric', False)
    sym = plot_kwargs.pop('sym', False) or sym
    norm_dict['symetric'] = sym

    # parse shorthands for specifying color scale
    if "lin" in plot_kwargs:
        opt = plot_kwargs.pop('lin')
        norm_dict['vscale'] = 'lin'
        if opt == 0:
            norm_dict['symetric'] = True
        elif opt is not True:
            if not isinstance(opt, (list, tuple)):
                opt = [opt]
            norm_dict['clim'][:len(opt)] = opt
    if "log" in plot_kwargs:
        opt = plot_kwargs.pop('log')
        norm_dict['vscale'] = 'log'
        if opt is not True:
            if not isinstance(opt, (list, tuple)):
                opt = [opt]
            norm_dict['clim'][:len(opt)] = opt
    if "loglog" in plot_kwargs:
        opt = plot_kwargs.pop('loglog')
        norm_dict['crdscale'] = 'log'
        norm_dict['vscale'] = 'log'
        if opt is not True:
            if not isinstance(opt, (list, tuple)):
                opt = [opt]
            norm_dict['clim'][:len(opt)] = opt

    # replace 'None' or 'none' with None in clim, this is kinda hacky, non?
    for i in range(len(norm_dict['clim'])):
        if norm_dict['clim'][i] in ["None", "none"]:
            norm_dict['clim'][i] = None

    return actions, norm_dict

def _apply_actions(acts):
    for act in acts:
        act_args = act[1]
        if not isinstance(act_args, (list, tuple)):
            act_args = [act_args]
        act[0](*act_args)

def _plot2d_single(ax, fld, style, namex, namey, mod, scale,
                   masknan, latlon, flip_plot, patchec, patchlw, patchaa,
                   all_masked, extra_args, **kwargs):
    """Make a 2d plot of a single block

    Returns:
        result of the actual matplotlib plotting command
        (pcolormesh, contourf, etc.)
    """
    assert fld.nr_blocks == 1

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
            # this is a hack to get rid or the white space
            # between patches when contouring
            Xnc, Ync = fld.get_crds_nc((namex, namey))
            X[0], X[-1] = Xnc[0], Xnc[-1]
            Y[0], Y[-1] = Ync[0], Ync[-1]

    if latlon:
        # translate latitude from 0..180 to -90..90
        X, Y = np.meshgrid(X, 90 - Y)
    if mod:
        X *= mod[0]
        Y *= mod[1]

    dat = fld.data
    if scale is not None:
        dat *= scale
    # print(x.shape, y.shape, fld.data.shape)
    if masknan:
        dat = np.ma.masked_where(np.isnan(dat), dat)
        all_masked = all_masked and dat.mask.all()

    if flip_plot:
        X, Y = Y.T, X.T
        dat = dat.T
        namex, namey = namey, namex

    if style == "pcolormesh":
        p = ax.pcolormesh(X, Y, dat, *extra_args, **kwargs)
    elif style == "contour":
        p = ax.contour(X, Y, dat, *extra_args, **kwargs)
    elif style == "contourf":
        p = ax.contourf(X, Y, dat, *extra_args, **kwargs)
    elif style == "pcolor":
        p = ax.pcolor(X, Y, dat, *extra_args, **kwargs)
    else:
        raise RuntimeError("I don't understand {0} 2d plot style".format(style))

    try:
        if masknan:
            p.get_cmap().set_bad(masknan)
        else:
            raise ValueError()
    except ValueError:
        p.get_cmap().set_bad('k')

    # show patches?
    if patchec and patchlw:
        _xl = X[0]
        _yl = Y[0]
        _width = X[-1] - _xl
        _height = Y[-1] - _yl
        rect = plt.Rectangle((_xl, _yl), _width, _height,
                             edgecolor=patchec, linewidth=patchlw,
                             fill=False, antialiased=patchaa, zorder=5)
        ax.add_artist(rect)

    return p, all_masked

def plot2d_field(fld, ax=None, plot_opts=None, **plot_kwargs):
    """Plot a 2D Field using pcolormesh, contour, etc.

    Parameters:
        ax (matplotlib axis, optional): Plot in a specific axis object
        plot_opts (str, optional): plot options
        **plot_kwargs (str, optional): plot options

    Returns:
        (plot_object, colorbar_object)

    See Also:
        * :doc:`/plot_options`: Contains a full list of plot options
    """
    block0 = fld.blocks[0]
    if block0.nr_sdims != 2:
        raise RuntimeError("I will only contour a 2d field")

    # raise some deprecation errors
    if "extra_args" in plot_kwargs:
        raise ValueError("extra_args is deprecated and for internal use only")

    # init the plot by figuring out the options to use
    extra_args = []

    if not ax:
        ax = plt.gca()

    # parse plot_opts
    plot_opts_to_kwargs(plot_opts, plot_kwargs)
    actions, norm_dict = _extract_actions_and_norm(ax, plot_kwargs,
                                                   defaults={'equalaxis': True})

    # everywhere options
    scale = plot_kwargs.pop("scale", None)
    masknan = plot_kwargs.pop("masknan", True)
    flip_plot = plot_kwargs.pop("flip_plot", False)
    flip_plot = plot_kwargs.pop("flipplot", flip_plot)
    do_labels = plot_kwargs.pop("do_labels", True)
    do_labels = plot_kwargs.pop("dolabels", do_labels)
    xlabel = plot_kwargs.pop("xlabel", None)
    ylabel = plot_kwargs.pop("ylabel", None)
    show = plot_kwargs.pop("show", False)

    # 2d plot options
    style = plot_kwargs.pop("style", "pcolormesh")
    levels = plot_kwargs.pop("levels", 10)
    show_grid = plot_kwargs.pop("show_grid", False)
    show_grid = plot_kwargs.pop("g", show_grid)
    gridec = plot_kwargs.pop("gridec", None)
    gridlw = plot_kwargs.pop("gridlw", 0.25)
    gridaa = plot_kwargs.pop("gridaa", True)
    show_patches = plot_kwargs.pop("show_patches", False)
    show_patches = plot_kwargs.pop("p", show_patches)
    patchec = plot_kwargs.pop("patchec", None)
    patchlw = plot_kwargs.pop("patchlw", 0.25)
    patchaa = plot_kwargs.pop("patchaa", False)
    mod = plot_kwargs.pop("mod", None)
    colorbar = plot_kwargs.pop("colorbar", True)
    cbarlabel = plot_kwargs.pop("cbarlabel", None)
    earth = plot_kwargs.pop("earth", False)

    # undocumented options
    latlon = plot_kwargs.pop("latlon", None)
    norm = plot_kwargs.pop("norm", None)
    action_ax = plot_kwargs.pop("action_ax", ax)  # for basemap projections

    # some plot_kwargs need a little more info
    if show_grid:
        if not isinstance(show_grid, string_types):
            show_grid = 'k'
        if not gridec:
            gridec = show_grid
    if gridec and gridlw:
        plot_kwargs["edgecolors"] = gridec
        plot_kwargs["linewidths"] = gridlw
        plot_kwargs["antialiased"] = gridaa

    if show_patches:
        if not isinstance(show_patches, string_types):
            show_patches = 'k'
        if not patchec:
            patchec = show_patches

    if colorbar:
        if not isinstance(colorbar, dict):
            colorbar = {}
    else:
        colorbar = None

    #########################
    # figure out the norm...
    if norm is None:
        vscale = norm_dict['vscale']
        vmin, vmax = norm_dict['clim']

        if vmin is None:
            vmin = np.min([np.min(blk) for blk in fld.blocks])
        if vmax is None:
            vmax = np.max([np.max(blk) for blk in fld.blocks])

        if vscale == "lin":
            if norm_dict['symetric']:
                maxval = max(abs(vmin), abs(vmax))
                vmin = -1.0 * maxval
                vmax = +1.0 * maxval
            norm = Normalize(vmin, vmax)
        elif vscale == "log":
            if norm_dict['symetric']:
                raise ValueError("Can't use symetric color bar with logscale")
            if vmax < 0.0:
                print("Warning: Using log scale on a field with no positive "
                      "values")
                vmin, vmax = 1e-20, 1e-20
            elif vmin < 0.0:
                print("Warning: Using log scale on a field with negative "
                      "values. Only plotting 2 decades.")
                vmin, vmax = vmax / 100, vmax
            norm = LogNorm(vmin, vmax)
        else:
            raise ValueError("Unknown norm vscale: {0}".format(vscale))

        if "cmap" not in plot_kwargs and np.isclose(vmax, -1 * vmin):
            plot_kwargs['cmap'] = plt.get_cmap('seismic')
        plot_kwargs['norm'] = norm
    else:
        if isinstance(norm, Normalize):
            vscale = "lin"
        elif isinstance(norm, LogNorm):
            vscale = "log"
        else:
            raise TypeError("Unrecognized norm type: {0}".format(type(norm)))
        vmin, vmax = norm.vmin, norm.vmax

    # ok, here's some hackery for contours
    if style in ["contourf", "contour"]:
        if isinstance(levels, int):
            if vscale == "log":
                levels = np.logspace(np.log10(vmin), np.log10(vmax), levels)
            else:
                levels = np.linspace(vmin, vmax, levels)
        extra_args = [levels]

    ##############################
    # now actually make the plots

    # THIS IS BACKWARD, on account of the convention for
    # Coordinates where z, y, x is used since that is how
    # xdmf data is
    namey, namex = block0.crds.axes # fld.crds.get_culled_axes()

    all_masked = False
    for block in fld.blocks:
        p, all_masked = _plot2d_single(action_ax, block, style,
                                       namex, namey, mod, scale, masknan,
                                       latlon, flip_plot,
                                       patchec, patchlw, patchaa,
                                       all_masked, extra_args, **plot_kwargs)

    # apply option actions... this is for setting xlim / xscale / etc.
    _apply_actions(actions)

    if norm_dict['crdscale'] == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')

    # figure out the colorbar...
    if style == "contour":
        if "colors" in plot_kwargs:
            colorbar = None
    if colorbar is not None:
        # unless otherwise specified, use_gridspec for colorbar
        if "use_gridspec" not in colorbar:
            colorbar["use_gridspec"] = True
        if vscale == "log" and colorbar is not None and "ticks" not in colorbar:
            colorbar["ticks"] = matplotlib.ticker.LogLocator()
        # ok, this way to pass options to colorbar is bad!!!
        # but it's kind of the cleanest way to affect the colorbar?
        if masknan and all_masked:
            cbar = None
        else:
            cbar = plt.colorbar(p, **colorbar)
            if do_labels:
                if not cbarlabel:
                    cbarlabel = block0.pretty_name
                cbar.set_label(cbarlabel)
    else:
        cbar = None

    if do_labels:
        if not xlabel:
            xlabel = namex
        if not ylabel:
            ylabel = namey
        plt.xlabel(namex)
        plt.ylabel(namey)

    if earth:
        plot_earth(fld, axis=ax)
    if show:
        mplshow()
    return p, cbar

def _mlt_labels(longitude):
    return "{0:g}".format(longitude * 24.0 / 360.0)

def plot2d_mapfield(fld, ax=None, plot_opts=None, **plot_kwargs):
    """Plot data on a map projection of a sphere

    The default projection is polar, but any other basemap projection
    can be used.

    Parameters:
        ax (matplotlib axis, optional): Plot in a specific axis object
        plot_opts (str, optional): plot options
        **plot_kwargs (str, optional): plot options

    Returns:
        (plot_object, colorbar_object)

    Note:
        Parameters are in degrees, but if the projection is 'polar',
        then the plot is actually made in radians, which is important
        if you want to annotate a plot.

    See Also:
        * :doc:`/plot_options`: Contains a full list of plot options
    """
    if fld.nr_blocks > 1:
        raise TypeError("plot2d_mapfield doesn't do multi-block fields yet")

    if not ax:
        ax = plt.gca()

    # parse plot_opts
    plot_opts_to_kwargs(plot_opts, plot_kwargs)

    axgridec = plot_kwargs.pop("axgridec", 'grey')
    axgridls = plot_kwargs.pop("axgridls", ':')
    axgridlw = plot_kwargs.pop("axgridlw", 1.0)

    projection = plot_kwargs.pop("projection", "polar")
    hemisphere = plot_kwargs.pop("hemisphere", "north").lower().strip()
    drawcoastlines = plot_kwargs.pop("drawcoastlines", False)
    lon_0 = plot_kwargs.pop("lon_0", 0.0)
    lat_0 = plot_kwargs.pop("lat_0", None)
    bounding_lat = plot_kwargs.pop("bounding_lat", 40.0)
    title = plot_kwargs.pop("title", True)
    label_lat = plot_kwargs.pop("label_lat", True)
    label_mlt = plot_kwargs.pop("label_mlt", True)

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
            logger.warn("Clobbering axis for subplot %s; please give a polar "
                        "axis to plot2d_mapfield if you indend to use it "
                        "later.", ax.get_geometry())

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
        new_crds = coordinate.wrap_crds("uniform_spherical",
                                        [('lat', new_lat), ('lon', new_lon)],
                                        full_arrays=True)
        new_fld = fld.wrap(sl_fld.data, context=dict(crds=new_crds))

        plot_kwargs['do_labels'] = plot_kwargs['dolabels'] = False
        plot_kwargs['equalaxis'] = False
        ret = plot2d_field(new_fld, ax=ax, **plot_kwargs)

        if title:
            if not isinstance(title, string_types):
                title = new_fld.pretty_name
            plt.title(title)
        if axgridec:
            ax.grid(True, color=axgridec, linestyle=axgridls,
                    linewidth=axgridlw)
            ax.set_axisbelow(False)

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
        plot_kwargs['latlon'] = True
        plot_kwargs['do_labels'] = False
        plot_kwargs['equalaxis'] = False
        ret = plot2d_field(fld, ax=ax, action_ax=m, **plot_kwargs)
        if axgridec:
            if label_lat:
                lat_lables = [1, 1, 1, 1]
            else:
                lat_lables = [0, 0, 0, 0]
            m.drawparallels(latlabel_arr, labels=lat_lables,
                            color=axgridec, linestyle=axgridls,
                            linewidth=axgridlw)

            if label_mlt:
                mlt_labels = [1, 1, 1, 1]
            else:
                mlt_labels = [0, 0, 0, 0]
            m.drawmeridians(np.linspace(360.0, 0.0, 8, endpoint=False),
                            labels=mlt_labels, fmt=_mlt_labels,
                            color=axgridec, linestyle=axgridls,
                            linewidth=axgridlw)
        if drawcoastlines:
            m.drawcoastlines(linewidth=0.25)
        return ret

def plot1d_field(fld, ax=None, plot_opts=None, **plot_kwargs):
    """Plot a 1D Field using lines

    Parameters:
        ax (matplotlib axis, optional): Plot in a specific axis object
        plot_opts (str, optional): plot options
        **plot_kwargs (str, optional): plot options

    See Also:
        * :doc:`/plot_options`: Contains a full list of plot options
    """
    block0 = fld.blocks[0]
    if not ax:
        ax = plt.gca()

    # parse plot_opts
    plot_opts_to_kwargs(plot_opts, plot_kwargs)
    actions, norm_dict = _extract_actions_and_norm(ax, plot_kwargs,
                                                   defaults={'equalaxis': False})

    # everywhere options
    scale = plot_kwargs.pop("scale", None)
    masknan = plot_kwargs.pop("masknan", True)
    do_labels = plot_kwargs.pop("do_labels", True)
    do_labels = plot_kwargs.pop("dolabels", do_labels)
    xlabel = plot_kwargs.pop("xlabel", None)
    ylabel = plot_kwargs.pop("ylabel", None)
    show = plot_kwargs.pop("show", False)

    # 1d plot options
    legend = plot_kwargs.pop("legend", False)
    label = plot_kwargs.pop("label", block0.pretty_name)
    mod = plot_kwargs.pop("mod", None)

    plot_kwargs["label"] = label
    namex, = block0.crds.axes

    if block0.iscentered("Node"):
        x = np.concatenate([blk.get_crd_nc(namex) for blk in fld.blocks])
    elif block0.iscentered("Cell"):
        x = np.concatenate([blk.get_crd_cc(namex) for blk in fld.blocks])
    else:
        raise ValueError("1d plots can do node or cell centered data only")

    dat = np.concatenate([blk.data for blk in fld.blocks])

    if mod:
        x *= mod
    if scale:
        dat *= scale
    if masknan:
        dat = np.ma.masked_where(np.isnan(dat), dat)
    p = ax.plot(x, dat, **plot_kwargs)

    _apply_actions(actions)

    ###############################
    # set scale based on norm_dict
    vmin, vmax = norm_dict['clim']
    if norm_dict['crdscale'] == 'log':
        plt.xscale('log')
    if norm_dict['vscale'] == 'log':
        plt.yscale('log')
    if norm_dict['symetric']:
        if norm_dict['vscale'] == 'log':
            raise ValueError("log scale can't be symetric about 0")
        maxval = max(abs(max(dat)), abs(min(dat)))
        vmin, vmax = -maxval, maxval
    plt.ylim((vmin, vmax))

    ########################
    # apply labels and such
    if do_labels:
        if xlabel is None:
            xlabel = namex
        if ylabel is None:
            ylabel = label
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    if legend:
        if isinstance(legend, bool):
            legend = 0
        plt.legend(loc=0)

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
    if fld.nr_blocks > 1:
        raise TypeError("plot2d_quiver doesn't do multi-block fields yet")

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
               zorder=10):
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
    if hasattr(plane_spec, "blocks"):
        # this is for both Fields and AMRFields
        crd_system = plane_spec.blocks[0].meta.get("crd_system", crd_system)
        values = []
        for blk in plane_spec.blocks:
            # take only the 1st reduced.nr_sdims... this should just work
            try:
                plane, _value = blk.deep_meta["reduced"][0]
                values.append(_value)
            except KeyError:
                logger.error("No reduced dims in the field, i don't know what "
                             "2d \nplane, we're in and can't figure out the "
                             "size of earth.")
                return None
        value = np.min(np.abs(values))
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
