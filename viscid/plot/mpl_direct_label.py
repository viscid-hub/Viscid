#!/usr/bin/env python
"""For matplotlib: Instead of a legend, label the data directly

This module takes care to position labels so not to overlap other
elements already on the plot
"""

# Note, this is not in viscid/plot/__init__.py on purpose since it imports
# pylab - it is instead imported by viscid/plot/vlt.py and made accessable
# in its namespace

from __future__ import print_function, division, unicode_literals
from itertools import count, cycle
import re
import sys
import warnings

from matplotlib import pyplot as plt
from matplotlib.cbook import is_string_like
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.transforms import Bbox
import numpy as np


def estimate_text_size_px(txt, char_aspect_ratio=0.618, line_spacing=1.62,
                          fig=None, size=None, fontproperties=None, dpi=None):
    """Given some text, estimate its extent in pixels when rendered

    Args:
        txt (str): Text to estimate the render size of
        char_aspect_ratio (float): Assume characters have this aspect
            ratio on average. Really, this depends on font and which
            chars are actually in `txt`, but at least it's an estimate.
            Default is 1 / golden ratio.
        line_spacing (float): Assume lines are this fraction > 1 of the
            font height. Default is the golden ratio.
        fig (Figure): Matplotlib figure or None to use plt.gcf(), used
            to get the `dpi` if it's not given
        size (float): font height in points, gets the default from
            matplotlib if not given
        fontproperties (None): Specific fort progerties to get font
            size from if not given
        dpi (int): rendering dots per inch; defaults to `fig.get_dpi()`

    Returns:
        Tuple (width, height) of the text in pixels
    """
    if fig is None:
        fig = plt.gcf()

    if dpi is None:
        dpi = fig.get_dpi()

    if size is None:
        if fontproperties is None:
            fontproperties = FontProperties()
        elif is_string_like(fontproperties):
            fontproperties = FontProperties(fontproperties)
        size = fontproperties.get_size_in_points()

    char_height_px = dpi * size / 72
    # Note: default values of golden ratio and 1/golden ratio give ~13%
    #       overestimate for a single test, which is ok for my purposes
    char_width_px = char_aspect_ratio * char_height_px
    # guestimate the line height as some fraction > 1 of the font height
    line_height_px = line_spacing * char_height_px

    # i don't think i should do an rstrip here b/c the renderer doesn't
    lines = txt.split('\n')
    nlines = len(lines)
    widest_line_nchars = max([len(line) for line in lines])

    return (char_width_px * widest_line_nchars, line_height_px * nlines)

def _xy_as_pixels(xy, coords, ax=None):
    """convert xy from coords to pixels"""
    # make sure xy has two elements, one for x, one for y
    try:
        if len(xy) == 1:
            xy = [xy[0], xy[0]]
        elif len(xy) > 2:
            raise ValueError("only [x pad, y pad] please")
    except TypeError:
        xy = [xy, xy]

    if coords == "offset pixels":
        xy_px = xy
    elif coords == "axes fraction":
        if not ax:
            ax = plt.gca()
        xl, xh = ax.get_xlim()
        yl, yh = ax.get_ylim()
        xy_px = [xl + xy[0] * (xh - xl), yl + xy[1] * (yh - yl)]
    else:
        raise ValueError("coords '{0}' not understood".format(coords))

    return np.array(xy_px, dtype='f')

def estimate_text_size_data(txt, char_aspect_ratio=0.618, line_spacing=1.62,
                            ax=None, size=None, fontproperties=None, dpi=None):
    """Given some text, estimate its extent in data units when rendered

    Note:
        This is not useful if axes are log-scaled.

    Args:
        txt (str): Text to estimate the render size of
        char_aspect_ratio (float): Assume characters have this aspect
            ratio on average. Really, this depends on font and which
            chars are actually in `txt`, but at least it's an estimate.
            Default is 1 / golden ratio.
        line_spacing (float): Assume lines are this fraction > 1 of the
            font height. Default is the golden ratio.
        ax (Axes): Matplotlib axes or None to use plt.gca(), used
            to get the `dpi` if it's not given
        size (float): font height in points, gets the default from
            matplotlib if not given
        fontproperties (None): Specific fort progerties to get font
            size from if not given
        dpi (int): rendering dots per inch; defaults to `fig.get_dpi()`

    Returns:
        Tuple (width, height) of the text in data units.
    """
    if ax is None:
        ax = plt.gca()
    fig = ax.figure

    w_px, h_px = estimate_text_size_px(txt, char_aspect_ratio=char_aspect_ratio,
                                       line_spacing=line_spacing, fig=fig,
                                       size=size, fontproperties=fontproperties,
                                       dpi=dpi)

    ax_width_px = ax.bbox.width
    ax_height_px = ax.bbox.height

    xl, xh = plt.gca().get_xlim()
    yl, yh = plt.gca().get_ylim()

    data_width = w_px * ((xh - xl) / ax_width_px)
    data_height = h_px * ((yh - yl) / ax_height_px)

    return data_width, data_height

def _cycle_colors(colors, n_hands):
    if colors:
        colors = np.asarray(colors)
        if not colors.shape:
            colors = np.asarray([colors])

        if colors.dtype.kind not in ['S', 'U']:
            if len(colors.shape) == 1:
                colors = colors.reshape(1, -1)
            elif len(colors.shape) > 2:
                raise ValueError()
            print("???", colors.shape)
            # turn rgb -> rgba
            if colors.shape[1] == 3:
                colors = np.append(colors, np.ones_like(colors[:, :1]), axis=1)

        colors = [c for i, c in zip(range(n_hands), cycle(colors))]
    else:
        colors = [None] * n_hands
    return colors

def apply_labels(labels=None, colors=None, ax=None, magnet=(0.0, 1.0),
                 magnetcoords="axes fraction", padding=(8, 8),
                 paddingcoords="offset pixels", choices="00:02:20:22",
                 n_candidates=32, alpha=1.0, _debug=False, **kwargs):
    """Apply labels directly to series in liu of a legend

    The `choices` offsets are as follows::

        ---------------------
        |  02  |  12  | 22  |
        |-------------------|
        |  01  |  XX  | 21  |
        |-------------------|
        |  00  |  10  | 20  |
        ---------------------

    Args:
        labels (sequence): Optional sequence of labels to override the
            labels already in the data series
        colors (str, sequence): color as hex string, list of hex
            strings to color each label, or an Nx4 ndarray of rgba
            values for N labels
        ax (matplotlib.axis): axis; defaults to `plt.gca()`
        magnet (tuple): prefer positions that are closer to the magnet
        magnetcoords (str): 'offset pixels' or 'axes fraction'
        padding (tuple): padding for text in the (x, y) directions
        paddingcoords (str): 'offset pixels' or 'axes fraction'
        choices (str): colon separated list of possible label positions
            relative to the data values. The positions are summarized
            above.
        alpha (float): alpha channel (opacity) of label text. Defaults
            to 1.0 to make text visible. Set to `None` to use the
            underlying alpha from the handle's color.
        n_candidates (int): number of potential label locations to
            consider for each data series.
        _debug (bool): Mark up all possible label locations
        **kwargs: passed to plt.annotate

    Returns:
        List: annotation objects
    """
    if not ax:
        ax = plt.gca()

    if isinstance(colors, (list, tuple)):
        pass

    # choices:: "01:02:22" -> [(0, 1), (0, 2), (2, 2)]
    choices = [(int(c[0]), int(c[1])) for c in choices.split(':')]

    magnet_px = _xy_as_pixels(magnet, magnetcoords, ax=ax)
    padding_px = _xy_as_pixels(padding, paddingcoords, ax=ax)

    annotations = []

    cand_map = {}
    for choice in choices:
        cand_map[choice] = np.zeros([n_candidates, 2, 2], dtype='f')

    # these paths are all the paths we can get out hands on so that the text
    # doesn't overlap them. bboxes around labels are added as we go
    # paths = []
    paths_px = []
    # here is a list of bounding boxes around the text boxes as we add them
    bbox_paths_px = []

    ## how many vertices to avoid ?
    # artist
    # collection
    # image
    # line
    # patch
    # table
    # container

    for line in ax.lines:
        paths_px += [ax.transData.transform_path(line.get_path())]
    for collection in ax.collections:
        for pth in collection.get_paths():
            paths_px += [ax.transData.transform_path(pth)]

    hands, hand_labels = ax.get_legend_handles_labels()

    colors = _cycle_colors(colors, len(hands))

    for i, hand, label_i in zip(count(), hands, hand_labels):
        if labels and i < len(labels):
            label = labels[i]
        else:
            label = label_i

        # divine color of label
        if colors[i]:
            color = colors[i]
        else:
            try:
                color = hand.get_color()
            except AttributeError:
                color = hand.get_facecolor()[0]

        # get path vertices to determine candidate label positions
        try:
            verts = hand.get_path().vertices
        except AttributeError:
            verts = [p.vertices for p in hand.get_paths()]
            verts = np.concatenate(verts, axis=0)

        # resample the path of the current handle to get some candidate
        # locations for the label along the path
        s_src = np.linspace(0, 1, verts.shape[0])
        # shape is [i_candidate, value: [x, y]], corner: [lower left, upper right]
        s_dest = np.linspace(0, 1, n_candidates)
        root_dat = np.zeros([n_candidates, 2], dtype='f')
        root_px = np.zeros([n_candidates, 2], dtype='f')

        root_dat[:, 0] = np.interp(s_dest, s_src, verts[:, 0])
        root_dat[:, 1] = np.interp(s_dest, s_src, verts[:, 1])

        root_px[:, :] = ax.transData.transform(root_dat)

        # estimate the width and height of the label's text
        # txt_w, txt_h = estimate_text_size_data(label, ax=ax)
        txt_size = np.array(estimate_text_size_px(label, fig=ax.figure))
        txt_size = txt_size.reshape([1, 2])

        # this initial offset is needed to shift the center of the label
        # to the data point
        offset0 = -txt_size / 2

        # now we can shift the label away from the data point by an amount
        # equal to half the text width/height + the padding
        offset1 = padding_px + txt_size / 2

        for key, abs_px_arr in cand_map.items():
            ioff = np.array(key, dtype='i').reshape(1, 2) - 1
            total_offset = offset0 + ioff * offset1
            # approx lower left corner of the text box in absolute pixels
            abs_px_arr[:, :, 0] = root_px + total_offset
            # approx upper right corner of the text box in absolute pixels
            abs_px_arr[:, :, 1] = abs_px_arr[:, :, 0] + txt_size

        # candidates_abs_px[i] has root @ root_px[i % n_candidates]
        candidates_abs_px = np.concatenate([cand_map[c] for c in choices],
                                           axis=0)

        # find how many other things each candidate overlaps
        n_overlaps = np.zeros_like(candidates_abs_px[:, 0, 0])

        for k, candidate in enumerate(candidates_abs_px):
            cand_bbox = Bbox(candidate.T)

            # penalty for each time a box overlaps a path that's already
            # on the plot
            for _, path in enumerate(paths_px):
                if path.intersects_bbox(cand_bbox):
                    n_overlaps[k] += 1

            # slightly larger penalty if we intersect a text box that we
            # just added to the plot
            for _, path in enumerate(bbox_paths_px):
                if path.intersects_bbox(cand_bbox):
                    n_overlaps[k] += 5

            # big penalty if the candidate is out of the current view
            if not (ax.bbox.contains(*cand_bbox.min) and
                    ax.bbox.contains(*cand_bbox.max)):
                n_overlaps[k] += 100

        # sort candidates by distance between center of text box and magnet
        isorted = np.argsort(np.linalg.norm(0.5 * np.sum(candidates_abs_px,
                                                         axis=2) - magnet_px,
                                            axis=1))
        candidates_abs_px = np.array(candidates_abs_px[isorted, :, :])
        n_overlaps = np.array(n_overlaps[isorted])
        root_dat = np.array(root_dat[isorted % n_candidates, :])
        root_px = np.array(root_px[isorted % n_candidates, :])

        # sort candidates so the ones with the fewest overlaps are first
        sargs = np.argsort(n_overlaps)

        # >>> debug : add all candidates to the plot >>>
        if _debug:
            sm = plt.cm.ScalarMappable(cmap='magma',
                                       norm=plt.Normalize(vmin=0, vmax=4))
            sm._A = []
            for k, candidate in enumerate(candidates_abs_px):
                _croot_px = root_px[k, :]
                _croot_dat = np.array(root_dat[k, :])
                _txt_offset_px = np.array(candidate[:, 0] - _croot_px)
                _color = sm.to_rgba(n_overlaps[k])
                _txt = label

                plt.annotate(_txt, xy=_croot_dat, xycoords='data',
                             xytext=_txt_offset_px, textcoords="offset pixels",
                             color=_color, alpha=alpha,
                             bbox=dict(alpha=0.2, fc='white'))
                plt.plot(_croot_dat[0], _croot_dat[1], '^', color=_color)
            if i == 0:
                plt.colorbar(sm)
        # <<< debug <<<

        # pick winning candidate and add its bounding box to this list of
        # paths to avoid
        winner_abs_px = candidates_abs_px[sargs[0], :, :]
        xy_root_px = root_px[sargs[0], :]
        xy_root_dat = np.array(root_dat[sargs[0], :])
        xy_txt_offset = np.array(winner_abs_px[:, 0] - xy_root_px)

        corners = Bbox(winner_abs_px.T).corners()[(0, 1, 3, 2), :]
        bbox_paths_px += [Path(corners)]

        a = plt.annotate(label, xy=xy_root_dat, xycoords='data',
                         xytext=xy_txt_offset, textcoords="offset pixels",
                         color=color, **kwargs)
        annotations.append(a)
    return annotations

def _test0():
    _x = np.linspace(0, 5, 128)
    plt.plot(_x, np.sin(_x), label="sin wave")

    _txt = "some text"

    print("::", estimate_text_size_px(_txt))
    print("::", estimate_text_size_data(_txt))

    plt.annotate(_txt, estimate_text_size_data(_txt),
                 bbox=dict(alpha=0.2, fc='white'))
    plt.show()
    return 0

def _test1():
    def _gaussian(x, x0, var):
        return (2 * np.pi * var)**(-0.5) * np.exp(-(x - x0)**2 / (2 * var))

    x = np.linspace(0, 100, 512)

    series1 = 1.5 * _gaussian(x, 80, 20) + _gaussian(x, 60, 40)
    series2 = 1.65 * _gaussian(x, 90, 25) + 0.95 * _gaussian(x, 40, 80)
    plt.fill_between(x, series1, 0.0, color='#028482', alpha=0.7, lw=3,
                     label="Series 1")
    plt.fill_between(x, series2, 0.0, color='#7ABA7A', alpha=0.7, lw=3,
                     label="Series 2")
    apply_labels(magnet=(0.5, 1.0), padding=(6, 8), choices="02", alpha=None,
                 _debug=0)
    plt.show()

    return 0

def _main():
    errcode0 = _test0()
    errcode1 = _test1()
    return errcode0 + errcode1

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
