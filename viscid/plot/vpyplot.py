"""Convenience module for making matplotlib plots

Your best friend in this module is the :meth:`plot` function, but the
best reference for all quirky options is :meth:`plot2d_field`.

Note:
    You can't set rc parameters for this module!
"""

# FIXME: this module is way too long

from __future__ import print_function
from datetime import datetime
from distutils.version import LooseVersion
from itertools import count

import matplotlib
from matplotlib import rcParams
# hack for graceful fallback of Qt[45]Agg backends
__backend = rcParams.get('backend', None)
if __backend.lower().startswith(('qt4', 'qt5')):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if __backend.lower().startswith('qt4'):
            __new_backend = __backend.lower().replace('qt4', 'qt5')
        else:
            __new_backend = __backend.lower().replace('qt5', 'qt4')
        try:
            matplotlib.use(__new_backend, force=True, warn=False)
            import matplotlib.pyplot as plt
        except ImportError:
            matplotlib.use(__backend, force=True, warn=False)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LogNorm, ListedColormap
import matplotlib.dates as mdates
try:
    from mpl_toolkits.basemap import Basemap  # pylint: disable=no-name-in-module
    _HAS_BASEMAP = True
except ImportError:
    _HAS_BASEMAP = False

import viscid
from viscid import coordinate
from viscid.compat import izip, string_types
from viscid import logger
from viscid import pyeval
from viscid import vutil
from viscid.plot import mpl_style  # pylint: disable=unused-import
from viscid.plot import mpl_extra
from viscid.plot.mpl_direct_label import apply_labels
from viscid.plot.from_seaborn import despine

from viscid.plot import vseaborn

__mpl_ver__ = matplotlib.__version__
vseaborn.activate_from_viscid()


def plot(fld, selection=Ellipsis, force_cartesian=False, **kwargs):
    """Plot a field by dispatching to the most appropiate funciton

    * If fld has 1 spatial dimensions, call
      :meth:`plot1d_field(fld[selection], **kwargs)`
    * If fld has 2 spatial dimensions, call
      :meth:`plot2d_field(fld[selection], **kwargs)`
    * If fld is 2-D and has spherical coordinates (as is the case for
      ionospheric fields), try to use :meth:`plot2d_mapfield` which
      uses basemap to make its axes.

    Parameters:
        fld (Field): Some Field
        selection (optional): something that describes a field slice
        force_cartesian (bool): if false, then spherical plots will use
            plot_mapfield
        **kwargs: Passed on to plotting function

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

    Raises:
        TypeError: Description
        ValueError: Description
    """
    fld = fld.slice_reduce(selection)

    if not hasattr(fld, "patches"):
        raise TypeError("Selection '{0}' sliced away too many "
                        "dimensions".format(selection))
    if fld.nr_comps > 1:
        raise TypeError("Scalar Fields only")

    patch0 = fld.patches[0]
    nr_sdims = patch0.nr_sdims
    if nr_sdims == 1:
        return plot1d_field(fld, **kwargs)
    elif nr_sdims == 2:
        is_spherical = patch0.is_spherical()
        if is_spherical and not force_cartesian:
            return plot2d_mapfield(fld, **kwargs)
        else:
            return plot2d_field(fld, **kwargs)
    else:
        raise ValueError("mpl can only do 1-D or 2-D fields. Either slice the "
                         "field yourself, or use the selection keyword "
                         "argument")

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
    plot_kwargs = dict(plot_kwargs)
    if not plot_opts:
        return plot_kwargs

    plot_opts = plot_opts.strip()
    if plot_opts[0] == '{' and plot_opts[-1] == '}':
        try:
            import yaml
            d = yaml.safe_load(plot_opts)
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
    return plot_kwargs


def _pop_axis_opts(plot_kwargs, default='none'):
    if 'equalaxis' in plot_kwargs:
        viscid.logger.warning("equalaxis option deprecated, please specify "
                              "this axis options explicitly (i.e., axis='equal', "
                              "axis='image', axis='auto', etc.)")
        if 'axis' in plot_kwargs:
            viscid.logger.warning("Clobbering axis option with deprecated "
                                  "equalaxis option :'(")
        if plot_kwargs.pop('equalaxis'):
            plot_kwargs['axis'] = 'equal'
        else:
            plot_kwargs['axis'] = 'none'
    using_default_viscid_axis = 'axis' not in plot_kwargs
    _axis = plot_kwargs.pop("axis", default)
    if _axis is not None:
        _axis = _axis.strip().lower()
        if _axis in ('none', ''):
            _axis = None

    # print('_axis =', _axis)
    return _axis, using_default_viscid_axis, plot_kwargs

def _are_xy_shared(axis):
    x_is_shared = len(axis.get_shared_x_axes().get_siblings(axis)) > 1
    y_is_shared = len(axis.get_shared_y_axes().get_siblings(axis)) > 1
    return x_is_shared, y_is_shared

def _ax_on_edge(axis):
    axes_locator = axis.get_axes_locator()
    if axes_locator is None:
        bottom_most = axis.is_last_row()
        left_most = axis.is_first_col()
    else:
        try:
            rc_spec = axes_locator.get_subplotspec().get_rows_columns()
            nrows, ncols, row_start, row_stop, col_start, col_stop = rc_spec
        except AttributeError:
            subplot_spec = axes_locator.get_subplotspec()
            gridspec = subplot_spec.get_gridspec()
            nrows, ncols = gridspec.get_geometry()
            row_start, col_start = divmod(subplot_spec.num1, ncols)
            if subplot_spec.num2 is not None:
                row_stop, col_stop = divmod(subplot_spec.num2, ncols)
            else:
                row_stop = row_start
                col_stop = col_start

        bottom_most = row_stop >= nrows - 1
        left_most = col_start == 0

    return bottom_most, left_most

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
                                   'vscale': 'lin'|'log|none',
                                   'clim': [None|number, None|number],
                                   'symmetric': True|False}
    """
    for k, v in defaults.items():
        if k not in plot_kwargs:
            plot_kwargs[k] = v

    actions = []
    if not axis:
        axis = plt.gca()

    if 'axis' in plot_kwargs:
        _axis = plot_kwargs.pop('axis')
        if _axis is not None:
            if _axis == 'image':
                # this is a hack to allow image axes even when we're sharing
                # the x/y axes
                actions.append((axis.autoscale_view, [], dict(tight=True)))
                actions.append((axis.set_autoscale_on, False))
                x_is_shared, y_is_shared = _are_xy_shared(axis)
                if x_is_shared ^ y_is_shared:
                    adjustable = 'datalim'
                elif (LooseVersion(__mpl_ver__) < LooseVersion("2.2")
                      and (x_is_shared or y_is_shared)):
                    adjustable = 'box-forced'
                else:
                    adjustable = 'box'
                actions.append((axis.set_aspect, 'equal',
                                dict(adjustable=adjustable, anchor='C')))
            else:
                actions.append((axis.axis, _axis))
    if "x" in plot_kwargs:
        actions.append((axis.set_xlim, plot_kwargs.pop('x')))
    if "y" in plot_kwargs:
        actions.append((axis.set_ylim, plot_kwargs.pop('y')))
    if "own" in plot_kwargs:
        opt = plot_kwargs.pop('own')
        logger.warning("own axis doesn't seem to work yet...")
    if "ownx" in plot_kwargs:
        opt = plot_kwargs.pop('ownx')
        logger.warning("own axis doesn't seem to work yet...")
    if "owny" in plot_kwargs:
        opt = plot_kwargs.pop('owny')
        logger.warning("own axis doesn't seem to work yet...")

    norm_dict = {'crdscale': 'lin',
                 'vscale': 'lin',
                 'clim': [None, None],
                 'symmetric': False
                }

    if plot_kwargs.pop('logscale', False):
        norm_dict['vscale'] = 'log'

    if "clim" in plot_kwargs:
        clim = plot_kwargs.pop('clim')
        norm_dict['clim'][:len(clim)] = clim

    if "vmin" in plot_kwargs:
        norm_dict['clim'][0] = plot_kwargs.pop('vmin')

    if "vmax" in plot_kwargs:
        norm_dict['clim'][1] = plot_kwargs.pop('vmax')

    sym = plot_kwargs.pop('symmetric', False)
    sym = plot_kwargs.pop('sym', False) or sym
    norm_dict['symmetric'] = sym

    # parse shorthands for specifying color scale
    if "lin" in plot_kwargs:
        opt = plot_kwargs.pop('lin')
        norm_dict['vscale'] = 'lin'
        if opt == 0:
            norm_dict['symmetric'] = True
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

    # hack so that the value axis is not rescaled
    if plot_kwargs.pop('norescale', False):
        norm_dict['vscale'] = None

    return actions, norm_dict

def _apply_actions(acts):
    for act in acts:
        act_args = act[1]
        if not isinstance(act_args, (list, tuple)):
            act_args = [act_args]
        try:
            act_kwargs = act[2]
        except IndexError:
            act_kwargs = dict()
        act[0](*act_args, **act_kwargs)

def _prepare_time_axes(ax, ax_arrs, datefmt, timefmt, actions,
                       using_default_viscid_axis):
    new_ax_arrs = [None] * len(ax_arrs)
    datetime_fmt = [None] * len(ax_arrs)

    for i, XI in enumerate(ax_arrs):
        if viscid.is_datetime_like(XI):
            # new_ax_arrs[i] = mdates.date2num(viscid.as_datetime(XI))
            new_ax_arrs[i] = viscid.as_datetime(XI)
            datetime_fmt[i] = datefmt
        elif viscid.is_timedelta_like(XI):
            # new_ax_arrs[i] = mdates.date2num(viscid.as_datetime(XI))
            new_ax_arrs[i] = viscid.as_datetime(XI)
            datetime_fmt[i] = timefmt
        else:
            new_ax_arrs[i] = XI

    # with time axes, you probably don't want imageax, but only override
    # this if the user didn't specify with a plot_opt
    if using_default_viscid_axis and any(datetime_fmt):
        for i in reversed(range(len(actions))):
            if actions[i][0] in (ax.axis, ax.set_aspect):
                actions.pop(i)

    # take x and y plot opts and convert them to datetimes
    for i, setter in enumerate([ax.set_xlim, ax.set_ylim]):
        if datetime_fmt[i]:
            for iact, act in enumerate(actions):
                if act[0] == setter:
                    # print("OVERRIDE::", act)
                    datetimeified = viscid.as_datetime(act[1]).tolist()
                    actions[iact] = (setter, datetimeified)
                    # print("       --", actions[iact])

    return new_ax_arrs, datetime_fmt

def _apply_time_axes(fig, ax, datetime_fmt, autofmt_xdate):
    for fmt, axis_i in zip(datetime_fmt, (ax.xaxis, ax.yaxis)):
        if fmt:
            datetime_formatter = mdates.DateFormatter(fmt)
            axis_i.set_major_formatter(datetime_formatter)
            if axis_i is ax.xaxis and autofmt_xdate:
                fig.autofmt_xdate()

def _apply_axfmt(ax, majorfmt=None, minorfmt=None, majorloc=None, minorloc=None,
                 which_axes="xy"):
    ax_axes = {'x': ax.xaxis, 'y': ax.yaxis}

    if majorfmt == "steve":
        majorfmt = mpl_extra.steve_axfmt
    if minorfmt == "steve":
        minorfmt = mpl_extra.steve_axfmt

    for axis_name in which_axes:
        _axis = ax_axes[axis_name]

        if majorfmt:
            _axis.set_major_formatter(majorfmt)
        if minorfmt:
            _axis.set_minor_formatter(minorfmt)

        if majorloc:
            _axis.set_major_locator(majorloc)
        if minorloc:
            _axis.set_minor_locator(minorloc)

def _plot2d_single(ax, fld, style, namex, namey, mod, scale,
                   masknan, latlon, flip_plot, patchec, patchlw, patchaa,
                   datefmt, timefmt, autofmt_xdate, all_masked, extra_args,
                   actions, using_default_viscid_axis, **kwargs):
    """Make a 2d plot of a single patch

    Returns:
        result of the actual matplotlib plotting command
        (pcolormesh, contourf, etc.)
    """
    assert fld.nr_patches == 1

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

    dat = fld.data.T
    if scale is not None:
        dat *= scale
    if masknan:
        dat = np.ma.masked_where(np.isnan(dat), dat)
        all_masked = all_masked and dat.mask.all()
    if kwargs.pop('logscale_mask_neg', False):
        dat = np.ma.masked_where(dat <= 0.0, dat)
        all_masked = all_masked and dat.mask.all()

    # Field.data is now xyz as are the crds

    if flip_plot:
        X, Y = Y.T, X.T
        dat = dat.T
        namex, namey = namey, namex

    ax_arrs, datetime_fmt = _prepare_time_axes(ax, [X, Y], datefmt, timefmt,
                                               actions, using_default_viscid_axis)
    X, Y = ax_arrs

    # datetime_fmt = [False, False]
    # _XY = [X, Y]
    # for i, XI in enumerate(_XY):
    #     if viscid.is_datetime_like(XI):
    #         _XY[i] = mdates.date2num(viscid.as_datetime(XI))
    #         datetime_fmt[i] = datefmt
    #     elif viscid.is_timedelta_like(XI):
    #         _XY[i] = mdates.date2num(viscid.as_datetime(XI))
    #         datetime_fmt[i] = timefmt
    # X, Y = _XY

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

    for _fmt, axis_i in zip(datetime_fmt, (ax.xaxis, ax.yaxis)):
        if _fmt:
            datetime_formatter = mdates.DateFormatter(_fmt)
            axis_i.set_major_formatter(datetime_formatter)
            if axis_i is ax.xaxis and autofmt_xdate:
                ax.get_figure().autofmt_xdate()

    try:
        if masknan:
            p.get_cmap().set_bad(masknan)
        else:
            raise ValueError()
    except ValueError:
        p.get_cmap().set_bad('y')

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
    patch0 = fld.patches[0]
    if patch0.nr_sdims != 2:
        raise RuntimeError("I will only contour a 2d field")

    # raise some deprecation errors
    if "extra_args" in plot_kwargs:
        raise ValueError("extra_args is deprecated and for internal use only")

    # init the plot by figuring out the options to use
    extra_args = []

    if not ax:
        ax = plt.gca()

    # parse plot_opts
    plot_kwargs = plot_opts_to_kwargs(plot_opts, plot_kwargs)

    # guess a good default value for 'axis'... i am so sorry for this logic
    # but i'm kinda playing the stanly cup in a sandbox here
    _xl = np.array(fld.xl[:2])
    _xh = np.array(fld.xh[:2])
    if 'x' in plot_kwargs:
        _xl[0], _xh[0] = plot_kwargs['x']
    if 'y' in plot_kwargs:
        _xl[1], _xh[1] = plot_kwargs['y']
    _xy_L = np.abs(_xh - _xl)
    if any(viscid.is_time_like(_xli, conservative=True) for _xli in _xy_L):
        _axis_def = 'none'
    else:
        _aspect = float(max(_xy_L)) / min(_xy_L)
        _axis_def = 'image' if _aspect <= 4.0 else 'none'
    # take a step back and go through any shared axes... if any of the
    # shared axes use 'image', 'equal', or 'box', then this axis should too
    shared_axes = set(ax.get_shared_x_axes().get_siblings(ax))
    shared_axes |= set(ax.get_shared_x_axes().get_siblings(ax))
    for sax in shared_axes:
        _asp = sax.get_aspect()
        _adj = sax.get_adjustable()
        _anc = sax.get_anchor()
        _aso = sax.get_autoscale_on()
        if _asp == 'equal' and _adj == 'datalim' and _aso:
            _axis_def = 'equal'
            break
        elif (_asp == 'equal' and _adj in ('box', 'box-forced') and _anc == 'C'
              and not _aso):
            _axis_def = 'image'
            break

    _axis, using_default_viscid_axis, plot_kwargs = _pop_axis_opts(plot_kwargs,
                                                                   default=_axis_def)

    flip_plot = plot_kwargs.pop("flipplot", False)
    flip_plot = plot_kwargs.pop("flip_plot", flip_plot)

    tightlim = plot_kwargs.pop('tightlim', plot_kwargs.pop('tightlims', True))
    _proj = getattr(ax, "transProjection", None)
    if tightlim and (_proj is None or 'identity' in _proj.__class__.__name__.lower()):
        _xl = np.array(fld.xl[:2])
        _xh = np.array(fld.xh[:2])
        if 'x' not in plot_kwargs:
            if flip_plot:
                plot_kwargs['x'] = (fld.xl[1], fld.xh[1])
            else:
                plot_kwargs['x'] = (fld.xl[0], fld.xh[0])
        if 'y' not in plot_kwargs:
            if flip_plot:
                plot_kwargs['y'] = (fld.xl[0], fld.xh[0])
            else:
                plot_kwargs['y'] = (fld.xl[1], fld.xh[1])

    actions, norm_dict = _extract_actions_and_norm(ax, plot_kwargs,
                                                   defaults={'axis': _axis})

    # everywhere options
    scale = plot_kwargs.pop("scale", None)
    masknan = plot_kwargs.pop("masknan", True)
    nolabels = plot_kwargs.pop("nolabels", False)
    xlabel = plot_kwargs.pop("xlabel", None)
    ylabel = plot_kwargs.pop("ylabel", None)
    majorfmt = plot_kwargs.pop("majorfmt", rcParams.get("viscid.majorfmt", None))
    minorfmt = plot_kwargs.pop("minorfmt", rcParams.get("viscid.minorfmt", None))
    majorloc = plot_kwargs.pop("majorloc", rcParams.get("viscid.majorloc", None))
    minorloc = plot_kwargs.pop("minorloc", rcParams.get("viscid.minorloc", None))
    datefmt = plot_kwargs.pop("datefmt", "%Y-%m-%d %H:%M:%S")
    timefmt = plot_kwargs.pop("timefmt", "%H:%M:%S")
    autofmt_xdate = plot_kwargs.pop("autofmtxdate", True)
    autofmt_xdate = plot_kwargs.pop("autofmt_xdate", autofmt_xdate)
    show = plot_kwargs.pop("show", False)

    # 2d plot options
    style = plot_kwargs.pop("style", "pcolormesh")
    levels = plot_kwargs.pop("levels", 10)
    show_grid = plot_kwargs.pop("g", False)
    show_grid = plot_kwargs.pop("show_grid", show_grid)
    gridec = plot_kwargs.pop("gridec", None)
    gridlw = plot_kwargs.pop("gridlw", 0.25)
    gridaa = plot_kwargs.pop("gridaa", True)
    show_patches = plot_kwargs.pop("p", False)
    show_patches = plot_kwargs.pop("show_patches", show_patches)
    patchec = plot_kwargs.pop("patchec", None)
    patchlw = plot_kwargs.pop("patchlw", 0.25)
    patchaa = plot_kwargs.pop("patchaa", True)
    mod = plot_kwargs.pop("mod", None)
    title = plot_kwargs.pop("title", None)
    cax = plot_kwargs.pop("cax", None)
    cbar = plot_kwargs.pop("cbar", True)
    colorbar = plot_kwargs.pop("colorbar", cbar)
    cbar_kwargs = plot_kwargs.pop('colorbar_kwargs', dict())
    cbar_kwargs = plot_kwargs.pop("cbar_kwargs", cbar_kwargs)
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

    if isinstance(colorbar, dict):
        viscid.logger.warning("Deprecation, colorbar options should be passed as "
                              "cbar_kwargs and colorbar should be True/False.")
        if cbar_kwargs:
            viscid.logger.warning("Clobbering cbar_kwargs with colorbar")
        colorbar, cbar_kwargs = True, colorbar

    #########################
    # figure out the norm...
    if norm is None:
        vscale = norm_dict['vscale']
        vmin, vmax = norm_dict['clim']

        if vmin is None:
            vmin = np.nanmin([np.nanmin(blk) for blk in fld.patches])
        if vmax is None:
            vmax = np.nanmax([np.nanmax(blk) for blk in fld.patches])

        # vmin / vmax will only be nan if all values are nan
        if np.isnan(vmin) or np.isnan(vmax):
            logger.warning("All-Nan encountered in Field, {0}"
                           "".format(patch0.name))
            vmin, vmax = 1e38, 1e38
            norm_dict['symmetric'] = False

        if vscale == "lin":
            if norm_dict['symmetric']:
                maxval = max(abs(vmin), abs(vmax))
                vmin = -1.0 * maxval
                vmax = +1.0 * maxval
            norm = Normalize(vmin, vmax)
        elif vscale == "log":
            if norm_dict['symmetric']:
                raise ValueError("Can't use symmetric color bar with logscale")
            if vmax <= 0.0:
                logger.warning("Using log scale on a field with no "
                               "positive values")
                plot_kwargs['logscale_mask_neg'] = True
                vmin, vmax = 1e-20, 1e-20
            elif vmin <= 0.0:
                logger.warning("Using log scale on a field with values "
                               "<= 0. Only plotting 4 decades.")
                plot_kwargs['logscale_mask_neg'] = True
                vmin, vmax = vmax / 1e4, vmax
            norm = LogNorm(vmin, vmax)
        elif vscale is None:
            norm = None
        else:
            raise ValueError("Unknown norm vscale: {0}".format(vscale))

        if norm is not None:
            plot_kwargs['norm'] = norm
    else:
        if isinstance(norm, Normalize):
            vscale = "lin"
        elif isinstance(norm, LogNorm):
            vscale = "log"
        else:
            raise TypeError("Unrecognized norm type: {0}".format(type(norm)))
        vmin, vmax = norm.vmin, norm.vmax

    if "cmap" not in plot_kwargs and np.isclose(vmax, -1 * vmin, atol=0):
        # by default, the symmetric_cmap is seismic (blue->white->red)
        symmetric_cmap = rcParams.get("viscid.symmetric_cmap", None)
        if symmetric_cmap:
            plot_kwargs['cmap'] = plt.get_cmap(symmetric_cmap)
        symmetric_vlims = True
    else:
        symmetric_vlims = False

    # ok, here's some hackery for contours
    if style in ["contourf", "contour"]:
        if plot_kwargs.get("colors", None):
            plot_kwargs['cmap'] = None
        if isinstance(levels, int):
            if vscale == "log":
                levels = np.logspace(np.log10(vmin), np.log10(vmax), levels)
            else:
                levels = np.linspace(vmin, vmax, levels)
        extra_args = [levels]

    ##############################
    # now actually make the plots
    namex, namey = patch0.crds.axes  # fld.crds.get_culled_axes()

    all_masked = False
    for patch in fld.patches:
        p, all_masked = _plot2d_single(action_ax, patch, style,
                                       namex, namey, mod, scale, masknan,
                                       latlon, flip_plot,
                                       patchec, patchlw, patchaa, datefmt, timefmt,
                                       autofmt_xdate, all_masked, extra_args,
                                       actions, using_default_viscid_axis,
                                       **plot_kwargs)

    # apply option actions... this is for setting xlim / xscale / etc.
    _apply_actions(actions)

    if norm_dict['crdscale'] == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')

    # figure out the colorbar...
    if style == "contour" and "colors" in plot_kwargs:
        colorbar = False
    if masknan and all_masked:
        colorbar = False

    if colorbar:
        if 'ax' in cbar_kwargs:
            viscid.logger.warning("ignoring cbar_kwargs['ax']")
            cbar_kwargs.pop('ax')

        if cax is not None:
            if 'cax' in cbar_kwargs:
                viscid.logger.warning("clobbering colorbar['cax']")
            cbar_kwargs['cax'] = cax

        if "cax" in cbar_kwargs or not cbar_kwargs.get('use_grid1', True):
            cax, cbar_kwargs, grid1_kwargs = _make_grid1_cbar_axes(ax, cbar_kwargs,
                                                                   make_cax=False)
            cax = cbar_kwargs.pop('cax', None)
        else:
            cax, cbar_kwargs, grid1_kwargs = _make_grid1_cbar_axes(ax, cbar_kwargs)

        if "ticks" not in cbar_kwargs:
            if vscale == "log":
                cbar_kwargs["ticks"] = matplotlib.ticker.LogLocator()
            elif symmetric_vlims:
                cbar_kwargs["ticks"] = matplotlib.ticker.MaxNLocator()
            else:
                cbar_kwargs["ticks"] = matplotlib.ticker.LinearLocator()

        cbarfmt = cbar_kwargs.pop("format", rcParams.get('viscid.cbarfmt', None))
        if cbarfmt == "steve":
            cbarfmt = mpl_extra.steve_cbarfmt
        if cbarfmt:
            cbar_kwargs["format"] = cbarfmt

        cbar = plt.colorbar(p, cax=cax, **cbar_kwargs)

        _cax_position = grid1_kwargs.get('position', None)
        if cax and _cax_position == 'top':
            cax.get_xaxis().set_ticks_position('top')
            cax.get_xaxis().set_label_position('top')
        # elif _cax_position == 'left':
        #     # cax.get_yaxis() is not a thing?
        #     cax.get_yaxis().set_ticks_position('left')
        #     cax.get_yaxis().set_label_position('left')

        # apply labels... or not
        if not nolabels and (cbarlabel or not title or
                             isinstance(title, string_types)):
            if not cbarlabel:
                cbarlabel = patch0.pretty_name
            cbar.set_label(cbarlabel)
    else:
        cbar = None

    if not nolabels:
        # Field.data is now xyz as are the crds
        if flip_plot:
            namex, namey = namey, namex
        if not xlabel:
            xlabel = namex
        if not ylabel:
            ylabel = namey
        if title:
            if not isinstance(title, string_types):
                title = patch0.pretty_name
            ax.set_title(title)

        x_is_shared, y_is_shared = _are_xy_shared(ax)
        bottom_most, left_most = _ax_on_edge(ax)
        if not x_is_shared or (x_is_shared and bottom_most):
            ax.set_xlabel(namex)
        if not y_is_shared or (y_is_shared and left_most):
            ax.set_ylabel(namey)

    _apply_axfmt(ax, majorfmt=majorfmt, minorfmt=minorfmt,
                 majorloc=majorloc, minorloc=minorloc)

    if earth:
        plot_earth(fld, axis=ax)
    plt.sca(ax)
    if show:
        mplshow()

    return p, cbar

def _make_grid1_cbar_axes(ax, cbar_kwargs, make_cax=True):
    _use_grid1 = cbar_kwargs.pop('use_grid1', make_cax)
    assert make_cax == _use_grid1

    orig_cbar_kwargs = dict(cbar_kwargs)

    position = cbar_kwargs.pop('position', None)
    orientation = cbar_kwargs.pop('orientation', None)

    # figure out consistant position / orientation, and warn if they are
    # inconsistent
    if position is None and orientation is None:
        position = 'right'
        orientation = 'vertical'
    # sanity check orientation given position
    if position in ('top', 'bottom'):
        if orientation not in (None, 'horizontal'):
            viscid.logger.warning("Colorbar position is '{0}', but "
                                  "orientation '{1}' is not horizontal."
                                  "".format(position, orientation))
        orientation = 'horizontal'
    if position in ('left', 'right'):
        if orientation not in (None, 'vertical'):
            viscid.logger.warning("Colorbar position is '{0}', but "
                                  "orientation '{1}' is not vertical."
                                  "".format(position, orientation))
        orientation = 'vertical'
    # sanity check position given orientation
    if orientation == 'vertical':
        if position not in ('left', 'right'):
            if position is not None:
                viscid.logger.warning("Colorbar orientation is horizontal, "
                                      "but position '{0}' is neither left nor "
                                      "right.".format(position))
            position = 'right'
    if orientation == 'horizontal':
        if position not in ('top', 'bottom'):
            if position is not None:
                viscid.logger.warning("Colorbar orientation is horizontal, "
                                      "but position '{0}' is neither left nor "
                                      "right.".format(position))
            position = 'bottom'

    cbar_kwargs['orientation'] = orientation

    grid1_kwargs = dict()
    grid1_kwargs['orientation'] = orientation
    grid1_kwargs['position'] = position
    grid1_kwargs['aspect'] = cbar_kwargs.pop('aspect', 20)
    default_pad = 0.05 if orientation == 'vertical' else 0.15
    grid1_kwargs['pad'] = cbar_kwargs.pop('pad', default_pad)
    grid1_kwargs['fraction'] = cbar_kwargs.pop('fraction', 0.05)
    grid1_kwargs['shrink'] = cbar_kwargs.pop('shrink', 1.0)
    cax = None

    if make_cax:
        try:
            from viscid.plot import _mpl_grid1
            cax = _mpl_grid1.make_grid1_cax(ax, **grid1_kwargs)
        except ImportError:
            viscid.logger.warning("Old matplotlib doesn't have "
                                  "mpl_toolkits.axes_grid1; falling back to "
                                  "awkward default colorbar axis.")
    if cax is None:
        # prepare to fallback to default mechanism, ie, let plt.colorbar
        # take all the kwargs and make its own axis
        cbar_kwargs = orig_cbar_kwargs
        cbar_kwargs.pop('position', None)
        cbar_kwargs['orientation'] = orientation
        if position in ('top', 'left'):
            viscid.logger.warning("Ignoring colorbar position '{0}'"
                                  "".format(position))

    ax.get_figure().sca(ax)
    return cax, cbar_kwargs, grid1_kwargs

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
    if fld.nr_patches > 1:
        raise TypeError("plot2d_mapfield doesn't do multi-patch fields yet")

    # parse plot_opts
    plot_kwargs = plot_opts_to_kwargs(plot_opts, plot_kwargs)

    axgridec = plot_kwargs.pop("axgridec", 'grey')
    axgridls = plot_kwargs.pop("axgridls", ':')
    axgridlw = plot_kwargs.pop("axgridlw", 1.0)

    projection = plot_kwargs.pop("projection", "polar")
    hemisphere = plot_kwargs.pop("hemisphere", "none").lower().strip()
    drawcoastlines = plot_kwargs.pop("drawcoastlines", False)
    lon_0 = plot_kwargs.pop("lon_0", 0.0)
    lat_0 = plot_kwargs.pop("lat_0", None)
    bounding_lat = plot_kwargs.pop("bounding_lat", None)
    bounding_lat_specified = bounding_lat is not None
    title = plot_kwargs.pop("title", True)
    label_lat = plot_kwargs.pop("label_lat", True)
    label_mlt = plot_kwargs.pop("label_mlt", True)

    # try to autodiscover hemisphere if ALL the thetas are either above
    # or below the equator
    lat = viscid.as_mapfield(fld, units='deg').get_crd('lat')
    if hemisphere == "none":
        if np.all(lat >= 0.0):
            hemisphere = "north"
        elif np.all(lat <= 0.0):
            hemisphere = "south"
        else:
            viscid.logger.warning("hemisphere of field {0} is ambiguous, the "
                                  "field contains both. Please specify either "
                                  "north or south (defaulting to North)."
                                  "".format(fld.name))
            hemisphere = 'north'

    # set a sensible default for bounding_lat... full spheres get cut off
    # at 40 deg, but hemispheres or smaller are shown in full
    if bounding_lat is None:
        if abs(lat[-1] - lat[0]) >= 90.01:
            bounding_lat = 40.0
        else:
            bounding_lat = 90.0

    if hemisphere in ("north", 'n'):
        # def_projection = "nplaea"
        pass
    elif hemisphere in ("south", 's'):
        # def_projection = "splaea"
        bounding_lat = -1 * np.abs(bounding_lat)
    else:
        raise ValueError("hemisphere is either 'north' or 'south'")

    # boundinglat = kwargs.pop("boundinglat", def_boundinglat)
    # lon_0 = kwargs.pop("lon_0", 0.0)
    # lat_0 = kwargs.pop("lat_0", None)
    # drawcoastlines = kwargs.pop("drawcoastlines", False)

    make_periodic = plot_kwargs.get('style', None) in ("contour", "contourf")
    new_fld = viscid.as_polar_mapfield(fld, bounding_lat=bounding_lat,
                                       hemisphere=hemisphere,
                                       make_periodic=make_periodic)


    if projection != "polar" and not _HAS_BASEMAP:
        viscid.logger.error("NOTE: install the basemap for the desired "
                            "spherical projection; falling back to "
                            "matplotlib's polar plot.")
        projection = "polar"

    if projection == "polar":
        if LooseVersion(__mpl_ver__) < LooseVersion("1.1"):
            raise RuntimeError("polar plots are annoying for matplotlib < ",
                               "version 1.1. Update your matplotlib and "
                               "profit.")

        ax = _get_polar_axis(ax=ax)

        show = plot_kwargs.pop('show', False)
        plot_kwargs['nolabels'] = True
        plot_kwargs['axis'] = 'none'
        # hack to forceably add padding to colorbar so all labels are visible
        plot_kwargs = _set_default_cbar_pad(plot_kwargs, pad=0.2)

        ret = plot2d_field(new_fld, ax=ax, **plot_kwargs)
        if bounding_lat_specified:
            abslatlim = np.abs(bounding_lat)
        else:
            abslatlim = np.rad2deg(np.abs(new_fld.xh[1]))
        ax.set_theta_offset(-90 * np.pi / 180.0)

        if title:
            if not isinstance(title, string_types):
                title = new_fld.pretty_name
            plt.title(title)
        if axgridec:
            mlt_grid_pos = (0, 45, 90, 135, 180, 225, 270, 315)
            mlt_labels = (24, 3, 6, 9, 12, 15, 18, 21)
            if not label_mlt:
                mlt_labels = ()
            ax.set_thetagrids(mlt_grid_pos, mlt_labels)

            grid_label_origin = 10
            if abslatlim > 50:
                grid_label_dr = 20
            else:
                grid_label_dr = 10
            # grid_dr = abs_grid_dr * np.sign(bounding_lat)
            lat_grid_pos = np.arange(grid_label_origin, abslatlim, grid_label_dr)
            lat_labels = np.arange(grid_label_origin, abslatlim, grid_label_dr)
            if label_lat == "from_pole":
                lat_labels = ["{0:g}".format(l) for l in lat_labels]
            elif label_lat:
                if hemisphere in ('north', 'n'):
                    lat_labels = 90 - lat_labels
                else:
                    lat_labels = -90 + lat_labels
                lat_labels = ["{0:g}".format(l) for l in lat_labels]
            else:
                lat_labels = []
            ax.set_rgrids((np.pi / 180.0) * lat_grid_pos, lat_labels)
            ax.set_rmax(np.deg2rad(abslatlim))
            ax.grid(True, color=axgridec, linestyle=axgridls,
                    linewidth=axgridlw)
            ax.set_axisbelow(False)
        else:
            ax.grid(False)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        plt.sca(ax)
        if show:
            mplshow()
        return ret

    else:
        if not ax:
            ax = plt.gca()
        m = Basemap(projection=projection, lon_0=lon_0, lat_0=lat_0,
                    boundinglat=bounding_lat, ax=ax)
        show = plot_kwargs.pop('show', False)
        plot_kwargs['latlon'] = True
        plot_kwargs['nolabels'] = True
        plot_kwargs['axis'] = 'none'
        ret = plot2d_field(fld, ax=ax, action_ax=m, **plot_kwargs)
        if axgridec:
            if label_lat:
                lat_lables = [1, 1, 1, 1]
            else:
                lat_lables = [0, 0, 0, 0]

            if np.abs(bounding_lat) > 50.0:
                latlabel_arr = np.linspace(20.0, 80.0, 4)
            else:
                latlabel_arr = np.linspace(50.0, 80.0, 4)

            if hemisphere in ("south", 's'):
                latlabel_arr *= -1

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
        plt.sca(ax)
        if show:
            mplshow()
        return ret

def _set_default_cbar_pad(kwargs, pad=0.2, inplace=True):
    if not inplace:
        kwargs = dict(kwargs)

    cbar = kwargs.get('colorbar', True)
    if isinstance(cbar, dict):
        set_key = 'colorbar'
        cbar_kwargs = cbar
    elif 'cbar_kwargs' in kwargs:
        set_key = 'cbar_kwargs'
        cbar_kwargs = kwargs.get('cbar_kwargs')
    elif 'colorbar_kwargs' in kwargs:
        set_key = 'colorbar_kwargs'
        cbar_kwargs = kwargs.get('colorbar_kwargs')
    else:
        set_key = 'cbar_kwargs'
        cbar_kwargs = dict()

    if 'pad' not in cbar_kwargs:
        cbar_kwargs['pad'] = pad
        kwargs[set_key] = cbar_kwargs

    return kwargs

def plot_iono(fld, *args, **kwargs):
    """Wrapper for easier annotated ionosphere plots

    Args:
        fld (Field): Some spherical field
        **kwargs: Consumed, or passed to :py:func:`plot2d_mapfield`

    Keyword Arugments:
        scale (float): scale fld by some scalar
        annotations (str): 'pot' to annotate min/max/cpcp
        units (str): units for annotation values
        hemisphere (str): 'north' or 'south'
        fontsize (int): point of font
        titlescale (float): scale fontsize of title by this much
        title (str): title for the plot

    Returns:
        (plot_object, colorbar_object)

    See Also:
        * :doc:`/plot_options`: Contains a full list of plot options

    Raises:
        ValueError: on bad hemisphere
    """
    kwargs['nolabels'] = True
    scale = kwargs.pop("scale", None)
    annotations = kwargs.pop("annotations", 'pot').strip().lower()
    units = kwargs.pop("units", '').strip()
    _fontsize = kwargs.pop("fontsize", 12)
    _title_scale = kwargs.pop("titlescale", 1.25)
    _title = '\n' + kwargs.get("title", fld.pretty_name).strip() + '\n'
    hem = kwargs.get("hemisphere", "None").strip().lower()

    fld = viscid.as_mapfield(fld, units='deg')
    if scale is not None:
        fld *= scale
    lat = fld.get_crd('lat')

    kwargs = _set_default_cbar_pad(kwargs, pad=0.2)

    if units:
        units = " " + units

    if hem == 'none':
        if np.all(lat >= 0.0):
            hem = 'north'
        elif np.all(lat <= 0.0):
            hem = 'south'
        else:
            hem = 'north'

    if hem in ('north', 'n'):
        fldH = fld["lat=10j:"]
    elif hem in ('south', 's'):
        fldH = fld['theta=:-10j']
    else:
        raise ValueError("Unknown hemisphere: {0}".format(hem))

    # now make the plot
    p, cbar = plot2d_mapfield(fld, *args, **kwargs)

    if annotations == 'pot':
        fldH_min = np.min(fldH)
        fldH_max = np.max(fldH)

        plt.annotate("Min: {0:.1f}{1}".format(fldH_min, units), (-0.04, 1.0),
                     fontsize=_fontsize, xycoords='axes fraction')
        plt.annotate("Max: {0:.1f}{1}".format(fldH_max, units), (0.75, 1.0),
                     fontsize=_fontsize, xycoords='axes fraction')
        plt.annotate("CPCP: {0:.1f}{1}".format(fldH_max - fldH_min, units),
                     (-0.04, -0.04), fontsize=_fontsize, xycoords='axes fraction')
    elif annotations in ('', 'none'):
        pass
    else:
        raise ValueError("unknown annotations type: {0}".format(annotations))

    plt.title(_title, fontsize=int(_title_scale * _fontsize))

    # viscid.interact()
    if cbar:
        cbar.ax.margins(x=0.5)

    return p, cbar

def plot1d_field(fld, ax=None, plot_opts=None, **plot_kwargs):
    """Plot a 1D Field using lines

    Parameters:
        ax (matplotlib axis, optional): Plot in a specific axis object
        plot_opts (str, optional): plot options
        **plot_kwargs (str, optional): plot options

    See Also:
        * :doc:`/plot_options`: Contains a full list of plot options
    """
    patch0 = fld.patches[0]
    if not ax:
        ax = plt.gca()

    # parse plot_opts
    plot_kwargs = plot_opts_to_kwargs(plot_opts, plot_kwargs)
    _axis, using_default_viscid_axis, plot_kwargs = _pop_axis_opts(plot_kwargs)
    actions, norm_dict = _extract_actions_and_norm(ax, plot_kwargs,
                                                   defaults={'axis': _axis})

    # everywhere options
    scale = plot_kwargs.pop("scale", None)
    masknan = plot_kwargs.pop("masknan", True)
    nolabels = plot_kwargs.pop("nolabels", False)
    xlabel = plot_kwargs.pop("xlabel", None)
    ylabel = plot_kwargs.pop("ylabel", None)
    majorfmt = plot_kwargs.pop("majorfmt", rcParams.get("viscid.majorfmt", None))
    minorfmt = plot_kwargs.pop("minorfmt", rcParams.get("viscid.minorfmt", None))
    majorloc = plot_kwargs.pop("majorloc", rcParams.get("viscid.majorloc", None))
    minorloc = plot_kwargs.pop("minorloc", rcParams.get("viscid.minorloc", None))
    datefmt = plot_kwargs.pop("datefmt", "%Y-%m-%d %H:%M:%S")
    timefmt = plot_kwargs.pop("timefmt", "%H:%M:%S")
    autofmt_xdate = plot_kwargs.pop("autofmt_xdate", True)
    autofmt_xdate = plot_kwargs.pop("autofmtxdate", autofmt_xdate)
    show = plot_kwargs.pop("show", False)

    # 1d plot options
    legend = plot_kwargs.pop("legend", False)
    label = plot_kwargs.pop("label", patch0.pretty_name)
    mod = plot_kwargs.pop("mod", None)

    plot_kwargs["label"] = label
    namex, = patch0.crds.axes

    if patch0.iscentered("Node"):
        x = np.concatenate([blk.get_crd_nc(namex) for blk in fld.patches])
    elif patch0.iscentered("Cell"):
        x = np.concatenate([blk.get_crd_cc(namex) for blk in fld.patches])
    else:
        raise ValueError("1d plots can do node or cell centered data only")

    dat = np.concatenate([blk.data for blk in fld.patches])

    ax_arrs, datetime_fmt = _prepare_time_axes(ax, [x, dat], datefmt, timefmt,
                                               actions, using_default_viscid_axis)
    x, dat = ax_arrs

    if mod:
        x *= mod
    if scale:
        dat *= scale
    if masknan:
        dat = np.ma.masked_where(np.isnan(dat), dat)
    p = ax.plot(x, dat, **plot_kwargs)

    _apply_time_axes(ax.get_figure(), ax, datetime_fmt, autofmt_xdate)
    _apply_actions(actions)

    ###############################
    # set scale based on norm_dict
    vmin, vmax = norm_dict['clim']
    if norm_dict['crdscale'] == 'log':
        ax.set_xscale('log')
    if norm_dict['vscale'] == 'log':
        ax.set_yscale('log')
    if norm_dict['symmetric']:
        if norm_dict['vscale'] == 'log':
            raise ValueError("log scale can't be symmetric about 0")
        maxval = max(abs(max(dat)), abs(min(dat))) + 0.05 * (max(dat) - min(dat))
        vmin, vmax = -maxval, maxval
    if norm_dict['vscale'] is not None:
        if vmin is not None or vmax is not None:
            ax.set_ylim((vmin, vmax))

    ########################
    # apply labels and such
    if not nolabels:
        if xlabel is None:
            xlabel = namex
        if ylabel is None:
            ylabel = label
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    _apply_axfmt(ax, majorfmt=majorfmt, minorfmt=minorfmt,
                 majorloc=majorloc, minorloc=minorloc)

    if legend:
        if isinstance(legend, bool):
            legend = 0
        ax.legend(loc=0)

    plt.sca(ax)
    if show:
        mplshow()
    return p, None

def plot2d_line(line, scalars=None, **kwargs):
    if scalars is not None:
        scalars = [scalars]
    return plot2d_lines([line], scalars=scalars, **kwargs)

def plot2d_lines(lines, scalars=None, symdir="", ax=None,
                 show=False, flip_plot=False, subsample=2,
                 pts_interp='linear', scalar_interp='linear',
                 marker=None, colors=None, marker_kwargs=None,
                 axis='none', equal=False, **kwargs):
    """Plot a list of lines in 2D

    Args:
        lines (list): list of 3xN ndarrays describing N xyz points
            along a line
        scalars (list, ndarray): a bunch of floats, rgb tuples, or
            '#0000ff' colors. These can be given as one per line,
            or one per vertex. See
            :py:func:`viscid.vutil.prepare_lines` for more info.
        symdir (str): direction perpendiclar to plane; one of 'xyz'
        ax (matplotlib Axis, optional): axis on which to plot (should
            be a 3d axis)
        show (bool): call plt.show when finished?
        flip_plot (bool): flips x and y axes on the plot
        subsample (int): Factor for resampling the number of vertices.
            If you are plotting a line where you want the color or line
            width to change along the line to show data, you will want
            `subsample > 2` to oversample the number of segments by a
            factor of 2. Otherwise, the colors will be off by half a
            line segment (due to limitations in matplotlib). To
            undersample the lines, you can use `0 < subsample < 1`.
        pts_interp (str): What kind of interpolation to use for
            vertices if subsample > 0. Must be a value recognized by
            :py:func:`scipy.interpolate.interp1d`.
        scalar_interp (str): What kind of interpolation to use for
            scalars if subsample > 0. Must be a value recognized by
            :py:func:`scipy.interpolate.interp1d`.
        marker (str): if given, plot the vertices using plt.scatter
        colors: overrides scalar to color mapping and is passed to
            matplotlib.collections.LineCollection. Note that
            LineCollection only accepts rgba tuples (ie, no generic
            strings). To give colors using one or more hex strings,
            use `scalars='#0f0f0f'` or similar. Use `colors=zloc`
            to color vertices with out-of-plane position.
        marker_kwargs (dict): additional kwargs for plt.scatter
        **kwargs: passed to matplotlib.collections.LineCollection

    Raises:
        ValueError: If a 2D plane can't be determined

    Returns:
        a LineCollection
    """
    if not ax:
        ax = plt.gca()

    if isinstance(scalars, viscid.string_types) and scalars == 'zloc':
        colors = 'zloc'
        scalars = None

    r = _prep_lines(lines, scalars=scalars, subsample=subsample,
                    pts_interp=pts_interp, scalar_interp=scalar_interp)
    verts, segments, vert_scalars, seg_scalars, vert_colors, seg_colors, _ = r
    # alpha = other['alpha']

    xind, yind, zind = _xyzind_from_symdir(segments, symdir)

    if flip_plot:
        xind, yind = yind, xind

    if colors == 'zloc':
        assert zind is not None
        vert_scalars = verts[zind, :]
        seg_scalars = segments[:, 0, zind]
        colors = None
    elif colors is not None:
        if seg_colors is not None:
            viscid.logger.warning("plot2d_lines - overriding seg_colors with "
                                  "explicit colors kwarg")
        seg_colors = colors

    line_collection = LineCollection(segments[:, :, [xind, yind]],
                                     array=seg_scalars, colors=seg_colors,
                                     **kwargs)
    ax.add_collection(line_collection)

    if marker:
        if not marker_kwargs:
            marker_kwargs = dict()

        # if colors are not given,
        if 'c' not in marker_kwargs:
            if vert_colors is not None:
                marker_kwargs['c'] = vert_colors
            elif vert_scalars is not None:
                marker_kwargs['c'] = vert_scalars
        # pass along some kwargs to the scatter plot
        for name in ['cmap', 'norm', 'vmin', 'vmax']:
            if name in kwargs and name not in marker_kwargs:
                marker_kwargs[name] = kwargs[name]
        ax.scatter(verts[xind, :], verts[yind, :], marker=marker,
                   **marker_kwargs)
    else:
        _autolimit_to_vertices(ax, verts[[xind, yind], :])

    if equal:
        axis = 'image'
    if axis.strip().lower() not in ('none', ''):
        ax.axis(axis)

    plt.sca(ax)
    if show:
        plt.show()

    return line_collection

def plot3d_line(line, scalars=None, **kwargs):
    if scalars is not None:
        scalars = [scalars]
    return plot3d_lines([line], scalars=scalars, **kwargs)

def plot3d_lines(lines, scalars=None, ax=None, show=False, subsample=2,
                 pts_interp='linear', scalar_interp='linear',
                 marker='', colors=None, marker_kwargs=None,
                 axis='none', equal=False, **kwargs):
    """Plot a list of lines in 3D

    Args:
        lines (list): list of 3xN ndarrays describing N xyz points
            along a line
        scalars (list, ndarray): a bunch of floats, rgb tuples, or
            '#0000ff' colors. These can be given as one per line,
            or one per vertex. See
            :py:func:`viscid.vutil.prepare_lines` for more info.
        ax (matplotlib Axis, optional): axis on which to plot (should
            be a 3d axis)
        show (bool): call plt.show when finished?
        subsample (int): Factor for resampling the number of vertices.
            If you are plotting a line where you want the color or line
            width to change along the line to show data, you will want
            `subsample > 2` to oversample the number of segments by a
            factor of 2. Otherwise, the colors will be off by half a
            line segment (due to limitations in matplotlib). To
            undersample the lines, you can use `0 < subsample < 1`.
        pts_interp (str): What kind of interpolation to use for
            vertices if subsample > 0. Must be a value recognized by
            :py:func:`scipy.interpolate.interp1d`.
        scalar_interp (str): What kind of interpolation to use for
            scalars if subsample > 0. Must be a value recognized by
            :py:func:`scipy.interpolate.interp1d`.
        marker (str): if given, plot the vertices using plt.scatter
        colors: overrides scalar to color mapping and is passed to
            mpl_toolkits.mplot3d.art3d.Line3DCollection. Note that this
            only accepts rgba tuples (ie, no generic strings). To give
            colors using one or more hex strings, use
            `scalars='#0f0f0f'` or similar.
        marker_kwargs (dict): additional kwargs for plt.scatter
        **kwargs: passed to mpl_toolkits.mplot3d.art3d.Line3DCollection

    Returns:
        TYPE: Line3DCollection
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    ax = _get_3d_axis(ax)

    r = _prep_lines(lines, scalars=scalars, subsample=subsample,
                    pts_interp=pts_interp, scalar_interp=scalar_interp)
    verts, segments, vert_scalars, seg_scalars, vert_colors, seg_colors, _ = r

    if colors is not None:
        if seg_colors is not None:
            viscid.logger.warning("plot3d_lines - overriding seg_colors with "
                                  "explicit colors kwarg")
        seg_colors = colors

    line_collection = Line3DCollection(segments[:, :, [0, 1, 2]],
                                       array=seg_scalars, colors=seg_colors,
                                       **kwargs)
    ax.add_collection3d(line_collection)

    if marker:
        if not marker_kwargs:
            marker_kwargs = dict()

        # if colors are not given,
        if 'c' not in marker_kwargs:
            if vert_colors is not None:
                marker_kwargs['c'] = vert_colors
            elif vert_scalars is not None:
                marker_kwargs['c'] = vert_scalars
        # pass along some kwargs to the scatter plot
        for name in ['cmap', 'norm', 'vmin', 'vmax']:
            if name in kwargs and name not in marker_kwargs:
                marker_kwargs[name] = kwargs[name]
        ax.scatter(verts[0, :], verts[1, :], verts[2, :], marker=marker,
                   **marker_kwargs)
    else:
        _autolimit_to_vertices(ax, verts)

    if equal:
        axis = 'image'
    if axis.strip().lower() not in ('none', ''):
        ax.axis(axis)

    plt.sca(ax)
    if show:
        plt.show()

    return line_collection

def plot2d_quiver(fld, step=1, ax=None, axis='none', equal=False, **kwargs):
    """Put quivers on a 2D plot

    The quivers will be plotted in the 2D plane of fld, so if fld
    is 3D, then one and only one dimenstion must have shape 1.

    Note:
        There are some edge cases where step doesn't work.

    Args:
        fld(VectorField): 2.5-D Vector field to plot
        step (int): only quiver every Nth grid cell. Can also
            be a list of ints to specify x & y downscaling separatly
        **kwargs: passed to :py:func:`matplotlpb.pyplot.quiver`

    Raises:
        TypeError: vector field check
        ValueError: 2d field check

    Returns:
        result of :py:func:`matplotlpb.pyplot.quiver`
    """
    if fld.nr_patches > 1:
        raise TypeError("plot2d_quiver doesn't do multi-patch fields yet")

    fld = fld.slice_reduce(":")

    if fld.patches[0].nr_sdims != 2:
        raise ValueError("2D Fields only for plot2d_quiver")
    if fld.nr_comps != 3:
        raise TypeError("Vector Fields only for plot2d_quiver")

    # get lm axes, ie, the axes in the plane
    l, m = fld.crds.axes
    lm = "".join([l, m])

    # get stepd scalar fields for the vector components in the plane
    if not hasattr(step, "__getitem__") or len(step) < 2:
        step = np.array([step, step]).reshape(-1)
    first_l = (fld.shape[0] % step[0]) // 2
    first_m = (fld.shape[1] % step[1]) // 2
    vl = fld[l][first_l::step[0], first_m::step[1]]
    vm = fld[m][first_l::step[0], first_m::step[1]]

    # get coordinates
    xl, xm = vl.get_crds(lm, shaped=True)
    xl, xm = np.broadcast_arrays(xl, xm)

    if ax is None:
        ax = plt.gca()

    ret = ax.quiver(xl, xm, vl, vm, **kwargs)
    if equal:
        axis = 'image'
    if axis.strip().lower() not in ('none', ''):
        ax.axis(axis)
    plt.sca(ax)
    return ret

def streamplot(fld, ax=None, axis='none', equal=False, **kwargs):
    """Plot 2D streamlines with :py:func:`matplotlib.pyplot.streamplot`

    Args:
        fld (VectorField): Some 2.5-D Vector Field
        **kwargs: passed to :py:func:`matplotlib.pyplot.streamplot`

    Raises:
        TypeError: vector field check
        ValueError: 2d field check

    Returns:
        result of :py:func:`matplotlib.pyplot.streamplot`
    """
    if fld.nr_patches > 1:
        raise TypeError("plot2d_quiver doesn't do multi-patch fields yet")

    fld = fld.slice_reduce(":")

    if fld.patches[0].nr_sdims != 2:
        raise ValueError("2D Fields only for streamplot")
    if fld.nr_comps != 3:
        raise TypeError("Vector Fields only for streamplot")

    # get lm axes, ie, the axes in the plane
    l, m = fld.crds.axes
    lm = "".join([l, m])

    # get scalar fields for the vector components in the plane
    fld = fld.atleast_3d()
    vl, vm = fld[l], fld[m]
    xl, xm = fld.get_crds(lm, shaped=False)

    # matplotlib's streamplot is for uniform grids only, if crds are non
    # uniform, then interpolate onto a new plane with uniform resolution
    # matching the most refined region of fld
    dxl = xl[1:] - xl[:-1]
    dxm = xm[1:] - xm[:-1]
    if not np.allclose(dxl[0], dxl) or not np.allclose(dxm[0], dxm):
        # viscid.logger.warning("Matplotlib's streamplot is for uniform grids only")
        vol = viscid.Volume(fld.xl, fld.xh, fld.sshape)
        vl = viscid.interp_trilin(vl, vol, wrap=True)
        vm = viscid.interp_trilin(vm, vol, wrap=True)

        xl, xm = vl.get_crds(lm)

        # interpolate linewidth and color too if given
        for other in ['linewidth', 'color']:
            try:
                if isinstance(kwargs[other], viscid.field.Field):
                    o_fld = kwargs[other]
                    o_fld = vol.wrap_field(viscid.interp_trilin(o_fld, vol))
                    kwargs[other] = o_fld.slice_reduce(":")
            except KeyError:
                pass

    # streamplot isn't happy if linewidth are color are Fields
    for other in ['linewidth', 'color']:
        try:
            if isinstance(kwargs[other], viscid.field.Field):
                kwargs[other] = kwargs[other].data.T
        except KeyError:
            pass

    vl = vl.slice_reduce(':')
    vm = vm.slice_reduce(':')

    if ax is None:
        ax = plt.gca()

    ret = ax.streamplot(xl, xm, vl.data.T, vm.data.T, **kwargs)
    if equal:
        axis = 'image'
    if axis.strip().lower() not in ('none', ''):
        ax.axis(axis)
    plt.sca(ax)
    return ret

def scatter_2d(points, c='k', symdir='', flip_plot=False, ax=None, show=False,
               axis='none', equal=False, **kwargs):
    """Plot scattered points on a matplotlib 3d plot

    Parameters:
        points: something shaped 3xN for N points, where 3 are the
            xyz cartesian directions in that order
        c (str, optional): color (in matplotlib format)
        ax (matplotlib Axis, optional): axis on which to plot (should
            be a 3d axis)
        show (bool, optional): show
        kwargs: passed along to :meth:`plt.statter`
    """
    if not ax:
        ax = plt.gca()

    xind, yind, zind = _xyzind_from_symdir(points, symdir)

    if flip_plot:
        xind, yind = yind, xind

    x = points[xind, :]
    y = points[yind, :]

    if c == 'zloc':
        if zind is None:
            raise RuntimeError("No 3rd dimension to pull colors from")
        c = points[zind, :]

    p = ax.scatter(x, y, c=c, **kwargs)

    if equal:
        axis = 'image'
    if axis.strip().lower() not in ('none', ''):
        ax.axis(axis)

    ax.set_xlabel("xyz"[xind])
    ax.set_ylabel("xyz"[yind])
    plt.sca(ax)
    if show:
        plt.show()
    return p, None

def scatter_3d(points, c='k', ax=None, show=False, axis='none', equal=False,
               **kwargs):
    """Plot scattered points on a matplotlib 3d plot

    Parameters:
        points: something shaped 3xN for N points, where 3 are the
            xyz cartesian directions in that order
        c (str, optional): color (in matplotlib format)
        ax (matplotlib Axis, optional): axis on which to plot (should
            be a 3d axis)
        show (bool, optional): show
        kwargs: passed along to :meth:`plt.statter`
    """
    import mpl_toolkits.mplot3d.art3d

    if not ax:
        ax = plt.gca(projection='3d')

    x = points[0]
    y = points[1]
    z = points[2]
    if c == 'zloc':
        c = z
    p = ax.scatter(x, y, z, c=c, **kwargs)
    if equal:
        axis = 'image'
    if axis.strip().lower() not in ('none', ''):
        ax.axis(axis)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.sca(ax)
    if show:
        plt.show()
    return p, None

def tighten(**kwargs):
    """Calls `matplotlib.pyplot.tight_layout(**kwargs)`"""
    try:
        plt.tight_layout(**kwargs)
    except AttributeError:
        logger.warning("No matplotlib tight layout support")

def auto_adjust_subplots(fig=None, tight_layout=True, subplot_params=None):
    """Wrapper to adjust subplots w/ tight_layout remembering axes lims

    Args:
        fig (Figure): a matplotlib figure
        tight_layout (bool, dict): flag for whether or not to apply a
            tight layout. If a dict, then it's unpacked into
            `plt.tight_layout(...)`
        subplot_params (dict): unpacked into `fig.subplots_adjust(...)`

    Returns:
        dict: keyword arguments for fig.subplots_adjust that describe
            the current figure after all adjustments are made
    """
    if fig is None:
        fig = plt.gcf()

    # remember the axes' limits before the call to tight_layout
    pre_tighten_xlim = [ax.get_xlim() for ax in fig.axes]
    pre_tighten_ylim = [ax.get_ylim() for ax in fig.axes]

    if tight_layout or isinstance(tight_layout, dict):
        if not isinstance(tight_layout, dict):
            tight_layout = {}
        tighten(**tight_layout)

    # apply specific subplot_params if given; hack for movies that wiggle
    if subplot_params:
        fig.subplots_adjust(**subplot_params)

    # re-apply the old axis limits; hack for movies that wiggle
    for ax, xlim, ylim in zip(fig.axes, pre_tighten_xlim, pre_tighten_ylim):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    p = fig.subplotpars
    ret = {'left': p.left, 'right': p.right, 'top': p.top,
           'bottom': p.bottom, 'hspace': p.hspace, 'wspace': p.wspace}
    return ret

def plot_earth(plane_spec, ax=None, scale=1.0, rot=0,
               daycol='w', nightcol='k', crd_system="gse",
               zorder=10, axis=None):
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
    if hasattr(plane_spec, "patches"):
        # this is for both Fields and AMRFields
        crd_system = viscid.as_crd_system(plane_spec.patches[0], crd_system)
        values = []
        for blk in plane_spec.patches:
            # take only the 1st reduced.nr_sdims... this should just work
            try:
                plane, _value = blk.meta["reduced"][0]
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

    if not ax:
        if axis:
            ax = axis
        else:
            ax = plt.gca()

    if crd_system == "gse":
        rot = 180

    if plane == 'y' or plane == 'z':
        ax.add_patch(mpatches.Wedge((0, 0), radius, 90 + rot, 270 + rot,
                                    ec=nightcol, fc=daycol, zorder=zorder))
        ax.add_patch(mpatches.Wedge((0, 0), radius, 270 + rot, 450 + rot,
                                    ec=nightcol, fc=nightcol, zorder=zorder))
    elif plane == 'x':
        if value < 0:
            ax.add_patch(mpatches.Circle((0, 0), radius, ec=nightcol,
                                         fc=daycol, zorder=zorder))
        else:
            ax.add_patch(mpatches.Circle((0, 0), radius, ec=nightcol,
                                         fc=nightcol, zorder=zorder))
    plt.sca(ax)
    return None

def get_current_colorcycle():
    try:
        cycle = matplotlib.rcParams['axes.prop_cycle']
        return list(c['color'] for c in cycle)
    except KeyError:
        return list(matplotlib.rcParams['axes.color_cycle'])

def show_colorcycle(pal=None, size=1):
    """Plot the values in a color palette as a horizontal array."""
    if not pal:
        pal = get_current_colorcycle()
    n = len(pal)
    _, ax = plt.subplots(1, 1, figsize=(n * size, size))
    ax.imshow(np.arange(n).reshape(1, n), cmap=ListedColormap(list(pal)),
              interpolation="nearest", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.sca(ax)
    plt.show()

def show_cmap(cmap=None, size=1, aspect=8):
    if cmap is None:
        cmap = plt.get_cmap()
    _, ax = plt.subplots(1, 1, figsize=(aspect * size, size))

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    ax.imshow(np.arange(256).reshape(1, 256), cmap=cmap,
              interpolation="nearest", aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.sca(ax)
    plt.show()

def _get_projected_axis(ax=None, projection='polar',
                        check_attr='set_thetagrids'):
    _new_axis = False
    if not ax:
        if len(plt.gcf().axes) == 0:
            _new_axis = True
        ax = plt.gca()
    if not hasattr(ax, check_attr):
        ax = plt.subplot(*ax.get_geometry(), projection=projection,
                         label=str(ax.get_geometry()) + projection)
        if not _new_axis:
            viscid.logger.warning("Clobbering axis for subplot %s; please give a "
                                  "%s axis if you indend to use it later.",
                                  ax.get_geometry(), projection)
    return ax

def _get_polar_axis(ax=None):
    return _get_projected_axis(ax=ax, projection='polar',
                               check_attr='set_thetagrids')

def _get_3d_axis(ax=None):
    from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-variable
    return _get_projected_axis(ax=ax, projection='3d', check_attr='zaxis')

def _autolimit_to_vertices(ax, verts):
    """Set limits on ax so that all verts are visible"""
    xmin, xmax = np.min(verts[0, ...]), np.max(verts[0, ...])
    ymin, ymax = np.min(verts[1, ...]), np.max(verts[1, ...])
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    if xlim[0] < xlim[1]:
        if xmin < xlim[0]:
            ax.set_xlim(left=xmin)
        if xmax > xlim[1]:
            ax.set_xlim(right=xmax)
    else:
        if xmax > xlim[0]:
            ax.set_xlim(left=xmax)
        if xmin < xlim[1]:
            ax.set_xlim(right=xmin)

    if ylim[0] < ylim[1]:
        if ymin < ylim[0]:
            ax.set_ylim(bottom=ymin)
        if ymax > ylim[1]:
            ax.set_ylim(top=ymax)
    else:
        if ymax > ylim[0]:
            ax.set_ylim(bottom=ymax)
        if ymin < ylim[1]:
            ax.set_ylim(top=ymin)

    # maybe z, maybe not
    if verts.shape[0] > 2:
        zlim = ax.get_zlim()
        zmin, zmax = np.min(verts[2, ...]), np.max(verts[2, ...])
        if zlim[0] < zlim[1]:
            if zmin < zlim[0]:
                ax.set_zlim(bottom=zmin)
            if zmax > zlim[1]:
                ax.set_zlim(top=zmax)
        else:
            if zmax > zlim[0]:
                ax.set_zlim(bottom=zmax)
            if zmin < zlim[1]:
                ax.set_zlim(top=zmin)

def _xyzind_from_symdir(arr, symdir):
    symdir = symdir.strip().lower()

    if symdir:
        if arr.shape[0] < 3:
            raise ValueError("specifying symdir assumes arr is 3d")
        if symdir == 'x':
            xind, yind, zind = 1, 2, 0
        elif symdir == 'y':
            xind, yind, zind = 0, 2, 1
        elif symdir == 'z':
            xind, yind, zind = 0, 1, 2
        else:
            raise ValueError("symdir '{0}' not in 'xyz'".format(symdir))
    else:
        xy_inds = list(range(arr.shape[0]))
        i = arr.shape[0] - 1
        while i >= 0 and len(xy_inds) > 2:
            if np.allclose(arr[i, :], arr[i, 0]):
                xy_inds.pop(i)
            i -= 1

        if len(xy_inds) > 2:
            xy_inds = [0, 1]
        elif len(xy_inds) < 2:
            xy_inds = [0, 0]
        xind, yind = xy_inds

        if arr.shape[0] > 2:
            zind = min(set(range(arr.shape[0])) - set(xy_inds))
        else:
            zind = None
    return xind, yind, zind

def _prep_lines(lines, scalars=None, subsample=2, pts_interp='linear',
                scalar_interp='linear', other=None):
    r = viscid.vutil.prepare_lines(lines, scalars, do_connections=True,
                                   other=other)
    verts, scalars, connections, other = r
    nr_sdims = verts.shape[0]
    nverts = verts.shape[1]
    nsegs = connections.shape[0]
    line_start = np.setdiff1d(np.arange(nverts), connections[:, 1],
                              assume_unique=True)
    line_stop = np.concatenate([line_start[1:], [nverts]])
    verts_per_line = line_stop - line_start
    nlines = len(line_start)
    assert nverts == nlines + nsegs

    if scalars is not None:
        scalars = np.atleast_2d(scalars)
    else:
        scalars = np.empty((0, nverts), verts.dtype)

    # Use numpy / scipy to interpolate points and scalars
    if subsample > 0:
        fine_verts = [None] * nlines
        fine_scalars = [None] * nlines
        fine_connections = [None] * nlines

        for i, start, stop in izip(count(), line_start, line_stop):
            n_coarse = stop - start  # number of verts, not segments
            n_fine = int(np.ceil(subsample * (n_coarse - 1)) + 1)
            coarse_verts = verts[:, start:stop]
            coarse_scalars = scalars[:, start:stop]
            fine_verts[i] = np.empty((nr_sdims, n_fine), dtype=verts.dtype)
            fine_scalars[i] = np.empty((scalars.shape[0], n_fine),
                                       dtype=verts.dtype)
            fine_connections[i] = np.empty((n_fine - 1, 2), dtype='i')
            t_coarse = np.linspace(0, 1, n_coarse)
            t_fine = np.linspace(0, 1, n_fine)

            try:
                # raise ImportError
                from scipy.interpolate import interp1d
                for j in range(coarse_verts.shape[0]):
                    fine_verts[i][j, :] = interp1d(t_coarse, coarse_verts[j, :],
                                                   kind=pts_interp)(t_fine)
                for j in range(scalars.shape[0]):
                    fine_scalars[i][j, :] = interp1d(t_coarse, coarse_scalars[j, :],
                                                     kind=scalar_interp)(t_fine)
            except ImportError:
                if pts_interp != 'linear' or scalar_interp != 'linear':
                    viscid.logger.error("Scipy is required to do anything "
                                        "other than linear interpolation")
                    raise

                for j in range(coarse_verts.shape[0]):
                    fine_verts[i][j, :] = np.interp(t_fine, t_coarse,
                                                    coarse_verts[j, :])
                for j in range(scalars.shape[0]):
                    fine_scalars[i][j, :] = np.interp(t_fine, t_coarse,
                                                      coarse_scalars[j, :])
            except ValueError:
                # this happens in scipy's interp1d if this line has exactly 1
                # vertex
                fine_verts[i][j, :] = coarse_verts[j, :]
                if coarse_scalars.shape[0] > 0:
                    fine_scalars[i][j, :] = coarse_scalars[j, :]

            new_start = np.sum(np.ceil((verts_per_line[:i] - 1) * subsample) + 1)

            new_stop = new_start + n_fine
            fine_connections[i][:, 0] = np.arange(new_start, new_stop - 1)
            fine_connections[i][:, 1] = np.arange(new_start + 1, new_stop)

        verts = np.concatenate(fine_verts, axis=1)
        was_uint8 = scalars.dtype == np.dtype('u1')
        scalars = np.concatenate(fine_scalars, axis=1)
        if was_uint8:
            scalars = scalars.round().astype('u1')
        connections = np.concatenate(fine_connections, axis=0)
        nverts = verts.shape[1]
        nsegs = connections.shape[0]
        assert nsegs == nverts - nlines

    # go through and make list of connected segments for the line collection
    segments = np.empty((nsegs, 2, nr_sdims), dtype=verts.dtype)
    segments[:, 0, :] = verts[:, connections[:, 0]].T
    segments[:, 1, :] = verts[:, connections[:, 1]].T

    # # TODO: straighten out array=scalars from color=scalars
    colors, seg_colors = None, None
    if scalars.shape[0] == 0:
        scalars = None
    elif scalars.shape[0] == 1:
        scalars = scalars[0]

    seg_scalars = None
    if scalars is not None:
        if scalars.dtype == np.dtype('u1'):
            colors = scalars.T / 255.0
            seg_colors = colors[connections[:, 0], ...]
            scalars = None
        else:
            seg_scalars = scalars[..., connections[:, 0]]

    return verts, segments, scalars, seg_scalars, colors, seg_colors, other

def interact(stack_depth=0, **kwargs):
    viscid.vutil.interact(stack_depth=stack_depth + 1, mpl_ns=True, **kwargs)

# just explicitly bring in some matplotlib functions
subplot = plt.subplot
subplots = plt.subplots
def subplot2grid(*args, **kwargs):
    viscid.logger.warning("pyplot.subplots should be preferred to "
                          "pyplot.subplot2grid")
    return plt.subplot2grid(*args, **kwargs)
clf = plt.clf
savefig = plt.savefig
show = plt.show
mplshow = show

# man, i was really indecisive about these names... luckily, everything's
# a reference in Python :)
plot_line = plot3d_line
plot_lines = plot3d_lines
plot_line3d = plot3d_line
plot_lines3d = plot3d_lines
plot_line2d = plot2d_line
plot_lines2d = plot2d_lines
plot_streamlines = plot3d_lines
plot_streamlines2d = plot2d_lines

##
## EOF
##
