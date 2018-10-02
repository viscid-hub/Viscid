#!/usr/bin/env python
"""For matplotlib: Instead of a legend, label the data directly

This module takes care to position labels so not to overlap other
elements already on the plot. Simply calling :py:func`apply_labels`
should act like a call to :py:func:`matplotlib.pyplot.legend`
"""

# Note, this is not in viscid/plot/__init__.py on purpose since it imports
# pylab. Instead, the entry point of this module is available using
# viscid.plot.vpyplot.apply_labels(...)

from __future__ import print_function, division, unicode_literals
from itertools import count, cycle
import sys

from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.transforms import Bbox
import numpy as np

from matplotlib import pyplot as plt


def apply_labels(labels=None, colors=None, ax=None, magnet=(0.5, 0.75),
                 magnetcoords="axes fraction", padding=None,
                 paddingcoords="offset points", choices="00:02:20:22",
                 n_candidates=32, ignore_filling=False, spacing='linear',
                 _debug=False, **kwargs):
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
        magnetcoords (str): 'offset pixels', 'offset points' or 'axes fraction'
        padding (tuple): padding for text in the (x, y) directions
        paddingcoords (str): 'offset pixels', 'offset points' or 'axes fraction'
        choices (str): colon separated list of possible label positions
            relative to the data values. The positions are summarized
            above.
        alpha (float): alpha channel (opacity) of label text. Defaults
            to 1.0 to make text visible. Set to `None` to use the
            underlying alpha from the handle's color.
        n_candidates (int): number of potential label locations to
            consider for each data series.
        ignore_filling (bool): if True, then assume it's ok to place
            labels inside paths that are filled with color
        spacing (str): one of 'linear' or 'random' to specify how far
            apart candidate locations are spaced along path segments
        _debug (bool): Mark up all possible label locations
        **kwargs: passed to plt.annotate

    Returns:
        List: annotation objects
    """
    if not ax:
        ax = plt.gca()

    if isinstance(colors, (list, tuple)):
        pass

    spacing = spacing.strip().lower()
    if spacing not in ('linear', 'random'):
        raise ValueError("Spacing '{0}' not understood".format(spacing))
    rand_state = np.random.get_state() if spacing == 'random' else None
    if rand_state is not None:
        # save the RNG state to restore it later so that plotting functions
        # don't change the results of scripts that use random numbers
        np.random.seed(1)

    _xl, _xh = ax.get_xlim()
    _yl, _yh = ax.get_ylim()
    axbb0 = np.array([_xl, _yl]).reshape(1, 2)
    axbb1 = np.array([_xh, _yh]).reshape(1, 2)

    # choices:: "01:02:22" -> [(0, 1), (0, 2), (2, 2)]
    choices = [(int(c[0]), int(c[1])) for c in choices.split(':')]

    _size = kwargs.get('fontsize', kwargs.get('size', None))
    _fontproperties = kwargs.get('fontproperties', None)
    font_size_pts = text_size_points(size=_size, fontproperties=_fontproperties)

    # set the default padding equal to the font size
    if paddingcoords == 'offset pixels':
        default_padding = font_size_pts * 72 / ax.figure.dpi
    elif paddingcoords == 'offset points':
        default_padding = font_size_pts
    elif paddingcoords == 'axes fraction':
        default_padding = 0.05
    else:
        raise ValueError("Bad padding coords '{0}'".format(paddingcoords))

    # print("fontsize pt:", font_size_pts,
    #       "fontsize px:", xy_as_pixels([font_size_pts, font_size_pts],
    #                                    'offset points')[0])

    if not isinstance(padding, (list, tuple)):
        padding = [padding, padding]
    padding = [default_padding if pd is None else pd for pd in padding]
    # print("padding::", paddingcoords, padding)

    magnet_px = xy_as_pixels(magnet, magnetcoords, ax=ax)
    padding_px = xy_as_pixels(padding, paddingcoords, ax=ax)
    # print("padding px::", padding_px)

    annotations = []

    cand_map = {}
    for choice in choices:
        cand_map[choice] = np.zeros([n_candidates, 2, 2], dtype='f')

    # these paths are all the paths we can get our hands on so that the text
    # doesn't overlap them. bboxes around labels are added as we go
    paths_px = []
    # here is a list of bounding boxes around the text boxes as we add them
    bbox_paths_px = []
    is_filled = []

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
        is_filled += [False]
    for collection in ax.collections:
        for pth in collection.get_paths():
            paths_px += [ax.transData.transform_path(pth)]
            is_filled += [collection.get_fill()]

    if ignore_filling:
        is_filled = [False] * len(is_filled)

    hands, hand_labels = ax.get_legend_handles_labels()

    colors = _cycle_colors(colors, len(hands))

    # >>> debug >>>
    if _debug:
        import viscid
        from matplotlib import patches as mpatches
        from viscid.plot import vpyplot as vlt

        _fig_width = int(ax.figure.bbox.width)
        _fig_height = int(ax.figure.bbox.height)
        fig_fld = viscid.zeros((_fig_width, _fig_height), dtype='f',
                               center='node')
        _X, _Y = fig_fld.get_crds(shaped=True)

        _axXL, _axYL, _axXH, _axYH = ax.bbox.extents

        _mask = np.bitwise_and(np.bitwise_and(_X >= _axXL, _X <= _axXH),
                               np.bitwise_and(_Y >= _axYL, _Y <= _axYH))
        fig_fld.data[_mask] = 1.0

        dfig, dax = plt.subplots(1, 1, figsize=ax.figure.get_size_inches())

        vlt.plot(fig_fld, ax=dax, cmap='ocean', colorbar=None)
        for _, path in enumerate(paths_px):
            dax.plot(path.vertices[:, 0], path.vertices[:, 1])
        dfig.subplots_adjust(bottom=0.0, left=0.0, top=1.0, right=1.0)
    else:
        dfig, dax = None, None
    # <<< debug <<<


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

        segl_dat = verts[:-1, :]
        segh_dat = verts[1:, :]

        # take out path segments that have one vertex outside the view
        _seg_mask = np.all(np.bitwise_and(segl_dat >= axbb0, segl_dat <= axbb1)
                           & np.bitwise_and(segh_dat >= axbb0, segh_dat <= axbb1),
                           axis=1)
        segl_dat = segl_dat[_seg_mask, :]
        segh_dat = segh_dat[_seg_mask, :]

        if np.prod(segl_dat.shape) == 0:
            print("no full segments are visible, skipping path", i, hand)
            continue

        segl_px = ax.transData.transform(segl_dat)
        segh_px = ax.transData.transform(segh_dat)
        seglen_px = np.linalg.norm(segh_px - segl_px, axis=1)

        # take out path segments that are 0 pixels in length
        _non0_seg_mask = seglen_px > 0
        segl_dat = segl_dat[_non0_seg_mask, :]
        segh_dat = segh_dat[_non0_seg_mask, :]
        segl_px = segl_px[_non0_seg_mask, :]
        segh_px = segh_px[_non0_seg_mask, :]
        seglen_px = seglen_px[_non0_seg_mask]

        if np.prod(segl_dat.shape) == 0:
            print("no non-0 segments are visible, skipping path", i, hand)
            continue

        # i deeply appologize for how convoluted this got, but the punchline
        # is that each line segment gets candidates proportinal to their
        # length in pixels on the figure
        s_src = np.concatenate([[0], np.cumsum(seglen_px)])
        if rand_state is not None:
            s_dest = s_src[-1] * np.sort(np.random.rand(n_candidates))
        else:
            s_dest = np.linspace(0, s_src[-1], n_candidates)
        _diff = s_dest.reshape(1, -1) - s_src.reshape(-1, 1)
        iseg = np.argmin(np.ma.masked_where(_diff <= 0, _diff), axis=0)
        frac = (s_dest - s_src[iseg]) / seglen_px[iseg]

        root_dat = (segl_dat[iseg]
                    + frac.reshape(-1, 1) * (segh_dat[iseg] - segl_dat[iseg]))
        root_px = ax.transData.transform(root_dat)

        # estimate the width and height of the label's text
        txt_size = np.array(estimate_text_size_px(label, fig=ax.figure,
                                                  size=font_size_pts))
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
            for ipth, path in enumerate(paths_px):
                if path.intersects_bbox(cand_bbox, filled=is_filled[ipth]):
                    n_overlaps[k] += 1

            # slightly larger penalty if we intersect a text box that we
            # just added to the plot
            for ipth, path in enumerate(bbox_paths_px):
                if path.intersects_bbox(cand_bbox, filled=is_filled[ipth]):
                    n_overlaps[k] += 5

            # big penalty if the candidate is out of the current view
            if not (ax.bbox.contains(*cand_bbox.min) and
                    ax.bbox.contains(*cand_bbox.max)):
                n_overlaps[k] += 100

        # sort candidates by distance between center of text box and magnet
        magnet_dist = np.linalg.norm(np.mean(candidates_abs_px, axis=-1)
                                     - magnet_px, axis=1)
        isorted = np.argsort(magnet_dist)
        magnet_dist = np.array(magnet_dist[isorted])
        candidates_abs_px = np.array(candidates_abs_px[isorted, :, :])
        n_overlaps = np.array(n_overlaps[isorted])
        root_dat = np.array(root_dat[isorted % n_candidates, :])
        root_px = np.array(root_px[isorted % n_candidates, :])

        # sort candidates so the ones with the fewest overlaps are first
        # but do it with a stable algorithm so among the best candidates,
        # choose the one closest to the magnet
        sargs = np.argsort(n_overlaps, kind='mergesort')

        # >>> debug >>>
        if dax is not None:
            for _candidate, n_overlap in zip(candidates_abs_px, n_overlaps):
                _cand_bbox = Bbox(_candidate.T)
                _x0 = _cand_bbox.get_points()[0]
                _bbox_center = np.mean(_candidate, axis=-1)
                _ray_x = [_bbox_center[0], magnet_px[0]]
                _ray_y = [_bbox_center[1], magnet_px[1]]
                dax.plot(_ray_x, _ray_y, '-', alpha=0.3, color='grey')
                _rect = mpatches.Rectangle(_x0, _cand_bbox.width,
                                           _cand_bbox.height, fill=False)
                dax.add_patch(_rect)
                plt.text(_x0[0], _x0[1], label, color='gray')
                plt.text(_x0[0], _x0[1], '{0}'.format(n_overlap))
        # <<< debug <<<

        # pick winning candidate and add its bounding box to this list of
        # paths to avoid
        winner_abs_px = candidates_abs_px[sargs[0], :, :]
        xy_root_px = root_px[sargs[0], :]
        xy_root_dat = np.array(root_dat[sargs[0], :])
        xy_txt_offset = np.array(winner_abs_px[:, 0] - xy_root_px)

        corners = Bbox(winner_abs_px.T).corners()[(0, 1, 3, 2), :]
        bbox_paths_px += [Path(corners)]

        # a = plt.annotate(label, xy=xy_root_dat, xycoords='data',
        #                  xytext=xy_txt_offset, textcoords="offset pixels",
        #                  color=color, **kwargs)
        a = ax.annotate(label, xy=xy_root_dat, xycoords='data',
                        xytext=xy_txt_offset, textcoords="offset pixels",
                        color=color, **kwargs)
        annotations.append(a)

    if rand_state is not None:
        np.random.set_state(rand_state)

    return annotations

def text_size_points(size=None, fontproperties=None):
    if not size:
        if fontproperties is None:
            fontproperties = FontProperties()
        elif isinstance(fontproperties, dict):
            fontproperties = FontProperties(**fontproperties)
        size = fontproperties.get_size_in_points()
    return size

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

    size = text_size_points(size=size, fontproperties=fontproperties)

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

def xy_as_pixels(xy, coords, ax=None):
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
    elif coords == "offset points":
        if not ax:
            ax = plt.gca()
        scale = ax.figure.dpi / 72
        xy_px = [scale * xy[0], scale * xy[1]]
    elif coords == "axes fraction":
        if not ax:
            ax = plt.gca()
        xl, yl = ax.bbox.corners()[0]
        xh, yh = ax.bbox.corners()[-1]
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
            # turn rgb -> rgba
            if colors.shape[1] == 3:
                colors = np.append(colors, np.ones_like(colors[:, :1]), axis=1)

        colors = [c for i, c in zip(range(n_hands), cycle(colors))]
    else:
        colors = [None] * n_hands
    return colors


def _test0():
    _x = np.linspace(0, 5, 128)
    plt.plot(_x, np.sin(_x), label="sin wave")

    _txt = "some text"

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

    _, ax0 = plt.subplots(1, 1, figsize=(10, 6))
    ax0.fill_between(x, series1, 0.0, color='#028482', alpha=0.7, lw=3,
                     label="Series 1")
    ax0.fill_between(x, series2, 0.0, color='#7ABA7A', alpha=0.7, lw=3,
                     label="Series 2")
    # plt.plot(x, series1, color='#028482', alpha=0.7, lw=3, label="Series 1")
    # plt.plot(x, series2, color='#7ABA7A', alpha=0.7, lw=3, label="Series 2")
    apply_labels(choices="02", magnet=(0.5, 0.0), alpha=None, _debug=True)
    plt.show()
    return 0

def _test2():
    t = np.linspace(20, 40, 128)
    y = np.cos(2 * np.pi * t / 3)
    plt.plot(t, y, '.-', label='wave')
    plt.gca().set_xlim(27, 33)
    plt.gca().set_ylim(-0.8, 0.8)
    apply_labels(n_candidates=40, choices="02", _debug=True)
    plt.show()
    return 0

def _test3():
    t = np.logspace(np.log10(1e0), np.log10(1e4), 128)
    y = np.cos(2 * np.pi * np.sqrt(t) / 3)
    plt.plot(t, y, '.-', label='wave')
    # plt.gca().set_ylim(-0.8, 0.8)
    plt.gca().set_xscale('log')
    apply_labels(n_candidates=40, choices="02", _debug=True)
    plt.show()
    return 0

def _test4():
    t = np.logspace(np.log10(1e0), np.log10(1e4), 32)
    y = np.cos(2 * np.pi * np.sqrt(t) / 3)
    plt.plot(t, y, '.-', label='wave')
    # plt.gca().set_ylim(-0.8, 0.8)
    plt.gca().set_xscale('log')
    apply_labels(n_candidates=40, choices="02", spacing='random', _debug=True)
    plt.show()
    return 0

def _main():
    errcode0 = _test0()
    errcode1 = _test1()
    errcode2 = _test2()
    errcode3 = _test3()
    errcode4 = _test4()
    return errcode0 + errcode1 + errcode2 + errcode3 + errcode4

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
