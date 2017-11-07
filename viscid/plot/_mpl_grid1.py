"""Hacks to matplotlib grid1 system to support projections (polar etc.)"""

# TODO: short circuit these hacks for versions of matplotlib (if any) that
#       have support for projected axes

import matplotlib
import matplotlib.projections as mprojections
from mpl_toolkits.axes_grid1 import axes_divider
from mpl_toolkits.axes_grid1 import axes_size
from mpl_toolkits.axes_grid1 import make_axes_locatable

from viscid import logger


def _get_axes_aspect(ax):
    aspect = ax.get_aspect()
    # when aspec is "auto", consider it as 1.
    if aspect in ('normal', 'auto'):
        aspect = 1.
    elif aspect == "equal":
        aspect = 1
    else:
        aspect = float(aspect)

    return aspect


class _PolarAxesX(axes_size.AxesX):
    def get_size(self, renderer):
        assert isinstance(self._axes, mprojections.polar.PolarAxes)
        r1, r2 = self._axes.get_ylim()
        l1, l2 = -r2, r2

        if self._aspect == "axes":
            ref_aspect = _get_axes_aspect(self._ref_ax)
            aspect = ref_aspect / _get_axes_aspect(self._axes)
        else:
            aspect = self._aspect

        rel_size = abs(l2 - l1) * aspect
        abs_size = 0.
        return rel_size, abs_size


class _PolarAxesY(axes_size.AxesY):
    def get_size(self, renderer):
        assert isinstance(self._axes, mprojections.polar.PolarAxes)
        r1, r2 = self._axes.get_ylim()
        l1, l2 = -r2, r2

        if self._aspect == "axes":
            ref_aspect = _get_axes_aspect(self._ref_ax)
            aspect = _get_axes_aspect(self._axes)
        else:
            aspect = self._aspect

        rel_size = abs(l2 - l1) * aspect
        abs_size = 0.
        return rel_size, abs_size


class _AxesX(axes_size.AxesX):
    pass


class _AxesY(axes_size.AxesY):
    pass


class _UnexpectedProjection(TypeError):
    def __init__(self, ax_type):
        self.ax_type = ax_type


def make_grid1_cax(ax, position='right', orientation='horizontal', aspect=20.0,
                   fraction=0.05, pad=0.05, shrink=1.0):
    # validate position && orientation
    if orientation == 'vertical':
        if position not in ('left', 'right'):
            raise ValueError("bad position '{0}'".format(position))
    elif orientation == 'horizontal':
        if position not in ('bottom', 'top'):
            raise ValueError("bad position '{0}'".format(position))
    else:
        raise ValueError("bad orientation '{0}'".format(orientation))

    cax = None

    try:
        # prepare axes size given orientation and axes type
        if isinstance(ax, mprojections.polar.PolarAxes):
            kls_x, kls_y = _PolarAxesX, _PolarAxesY
        elif type(ax) in (mprojections.axes.Axes, mprojections.axes.Subplot):
            # note this checks type, not isinstance because all axes derive
            # from axes.Axes, and that's not what we're checking here
            kls_x, kls_y = _AxesX, _AxesY
        else:
            raise _UnexpectedProjection(str(type(ax)))
        size_x = kls_x(ax, aspect=1.0 / aspect)
        size_y = kls_y(ax, aspect=1.0 / aspect)

        # scale the long side by shrink
        if orientation == 'vertical':
            size_y = axes_size.Fraction(1.0 / shrink, size_y)
            width = axes_size.Fraction(fraction, size_x)
            pad_size = axes_size.Fraction(pad, size_x)
        else:
            size_x = axes_size.Fraction(1.0 / shrink, size_x)
            width = axes_size.Fraction(fraction, size_y)
            pad_size = axes_size.Fraction(pad, size_y)

        divider = axes_divider.AxesDivider(ax, xref=size_x, yref=size_y)
        locator = divider.new_locator(nx=0, ny=0)
        ax.set_axes_locator(locator)
        cax = divider.append_axes(position, size=width, pad=pad_size,
                                  axes_class=matplotlib.axes.Axes)
    except _UnexpectedProjection as e:
        logger.info("Viscid can't verify that grid1 axes will "
                    "work to colorbar a {0} axis; falling back to "
                    "default.".format(e.ax_type))
    return cax
