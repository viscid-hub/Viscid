"""Tools for converting ndarrays <-> cmaps and plotting them"""

from __future__ import print_function

import numpy as np

try:
    from matplotlib.colors import Colormap, ListedColormap, LinearSegmentedColormap
    from matplotlib.cm import register_cmap as _register_cmap
except ImportError as e:
    from viscid import UnimportedModule
    Colormap = UnimportedModule(e)
    ListedColormap = UnimportedModule(e)
    LinearSegmentedColormap = UnimportedModule(e)
    _register_cmap = UnimportedModule(e)


def to_rgba(colors):
    """Turn colors to an Nx4 ndarray or rgba data

    Args:
        colors (Colormap or ndarray): colors as either a matplotlib
            Colormap or an Nx3 or Nx4 ndarray or rgb(a) data

    Returns:
        Nx4 ndarray of rgba data
    """
    if isinstance(colors, np.ndarray):
        if colors.shape[1] == 3:
            colors = np.hstack([colors, np.ones_like(colors[:, :1])])
        if colors.shape[1] != 4:
            raise ValueError("cmap ndarrays's first axis should have 3 "
                             "elements (rgb) or 4 (rgba)")
        return colors
    else:
        ret = colors(np.linspace(0.0, 1.0, colors.N))
        return ret

def to_linear_cmap(name, colors, reverse=False):
    """Turn colors into a matplotlib LinearSegmentedColormap

    Args:
        name (str): name of the color map
        colors (Colormap or ndarray): colors as either a matplotlib
            Colormap or an Nx3 or Nx4 ndarray or rgb(a) data
        reverse (bool): flip the lower and upper ends of the mapping

    Returns:
        :py:class:`matplotlib.colors.LinearSegmentedColormap` instance
    """
    if isinstance(colors, Colormap):
        if reverse or not isinstance(colors, LinearSegmentedColormap):
            colors = to_rgba(colors)
        else:
            return colors
    rgba = colors
    if reverse:
        rgba = rgba[::-1, :]
    return LinearSegmentedColormap.from_list(name, rgba)

def to_listed_cmap(name, colors, reverse=False):
    """Turn colors into a matplotlib ListedColormap

    Args:
        name (str): name of the color map
        colors (Colormap or ndarray): colors as either a matplotlib
            Colormap or an Nx3 or Nx4 ndarray or rgb(a) data
        reverse (bool): flip the lower and upper ends of the mapping

    Returns:
        :py:class:`matplotlib.colors.ListedColormap` instance
    """
    if isinstance(colors, Colormap):
        if reverse or not isinstance(colors, LinearSegmentedColormap):
            colors = to_rgba(colors)
        else:
            return colors
    rgba = colors
    if reverse:
        rgba = rgba[::-1, :]
    return ListedColormap(name, rgba)

def register_cmap(name, colors, reverse=False):
    """Register a cmap given a name and either a cmap or an Nx3 or Nx4
    ndarray or rgb(a) data.

    Calls to_linear_cmap(colors) before registering.

    Args:
        name (str, None): if None, name = colors.name; this will not
            play nicely with an ndarray
        colors (Colormap or ndarray): colors as either a matplotlib
            Colormap or an Nx3 or Nx4 ndarray or rgb(a) data
        reverse (bool): flip the lower and upper ends of the mapping
    """
    if name is None:
        name = colors.name
    cmap = to_linear_cmap(name, colors, reverse=reverse)
    _register_cmap(name=name, cmap=cmap)

def plot_ab(colors, ax, show=False):
    """Plot the color map in the ab plane of the L*a*b* color space

    Args:
        colors (Colormap or ndarray): colors as either a matplotlib
            Colormap or an Nx3 or Nx4 ndarray or rgb(a) data
        ax: matplotlib axes
        show (bool): call ax.figure.show() before returning
        **kwargs: passed to make_palette
    """
    from skimage.color import rgb2lab
    # cmap =
    rgba = to_rgba(colors)
    rgb = rgba[:, :3]
    lab = rgb2lab(rgb.reshape((-1, 1, 1, 3)))[:, 0, 0, :]

    ax.scatter(lab[:, 1], lab[:, 2], c=rgba, s=25, linewidths=0.0)
    ax.set_xlabel("a$^*$")
    ax.set_ylabel("b$^*$")
    if show:
        ax.figure.show()

    # l0, l1 = lab[0, 0], lab[0, -1]
    # grey_rgb = np.array([np.linspace(intensity0, intensity1, 256)] * 3)
    # ax.scatter(grey[0], grey[1], grey[1], c=grey_rgb.T, edgecolors='none',
    #            depthshade=False)

def plot_lstar(colors, ax, xoffset=0, show=False):
    """Plot the color map in the ab plane of the L*a*b* color space

    Args:
        colors (Colormap or ndarray): colors as either a matplotlib
            Colormap or an Nx3 or Nx4 ndarray or rgb(a) data
        ax: matplotlib axes
        xoffset (int): number to add to the x axis, useful when
            plotting on an axis with other lstar data
        show (bool): call ax.figure.show() before returning
        **kwargs: passed to make_palette
    """
    from skimage.color import rgb2lab
    cmap = to_linear_cmap('null', colors)
    rgba = to_rgba(colors)
    rgb = rgba[:, :3]
    l = rgb2lab(rgb.reshape((-1, 1, 1, 3)))[:, 0, 0, 0]
    x = np.arange(len(l))
    ax.scatter(x + xoffset, l, c=x, s=25, cmap=cmap, linewidths=0.0)
    ax.set_ylabel("L$^*$")

    # # find and plot the density of levels
    # dli = 1.0 / (l[1:] - l[:-1])
    # dli = 100.0 * (dli - np.min(dli)) / np.max(dli - np.min(dli))
    # ax.plot(x[:-1], dli, 'k-')
    if show:
        ax.figure.show()

def plot_rgb(colors, ax_cmap, ax_rgb, show=False):
    """Plot the red green and blue chanels of a color map

    Args:
        colors (Colormap or ndarray): colors as either a matplotlib
            Colormap or an Nx3 or Nx4 ndarray or rgb(a) data
        ax_cmap: matplotlib axis for cmap
        ax_rgb: matplotlib axis for rgb channel plot
        show (bool): call ax.figure.show() before returning
    """
    cmap = to_linear_cmap("", colors)
    rgba = to_rgba(colors)
    rgb = rgba[:, :3]

    if ax_cmap:
        a = np.outer(np.arange(0, 1, 0.01), np.ones(10)).T
        ax_cmap.axis("off")
        ax_cmap.imshow(a, aspect='auto', cmap=cmap, origin="lower")

    if ax_rgb:
        i = np.arange(rgb.shape[0])
        # ax_rgb.plot(i, np.linspace(intensity0, intensity1, len(i)), 'k-')
        ax_rgb.plot(i, rgba[:, 3], 'k-', lw=3)  # alpha channel
        ax_rgb.plot(i, rgba[:, 0], 'r-', lw=3)
        ax_rgb.plot(i, rgba[:, 1], 'g-', lw=3)
        ax_rgb.plot(i, rgba[:, 2], 'b-', lw=3)

    if show:
        if ax_cmap:
            ax_cmap.figure.show()
        elif ax_rgb:
            ax_rgb.figure.show()

##
## EOF
##
