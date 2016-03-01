# -*- coding: utf-8 -*-
"""Cubehelix color map implementation

The cubehelix colormaps are good for the colorblind because lightness
monotonically increases throughout the scale. These maps also guard
against levels appearing in the data due to a local maxima in the
lightness of that color in the color map.

Note:
    The cubehelix maps should be deprecated since they are defined in
    CIE-LAB space, which has undesired perceptual properties. Instead,
    one should use one of the ['magma', 'inferno', 'plasma', 'viridis']
    cmaps. These colormaps are backported into Viscid for those still
    using matplotlib < 1.5.0.

Attributes:
    redhelix_rgba: From red to cyan, counter-clockwise in hue
    bloodhelix_rgba: Same as redhelix, but with gamma == 1, so the
        intensity ramp is more linear instead of power law
    coolhelix_rgba: From bordeax to cool white, counter-clockwise in
        hue
    cubeYF_rgba: My attempt to replicate the cubeYF map described at
        `mycarta <https://mycarta.wordpress.com/2013/02/21/perceptual-
        rainbow-palette-the-method/>`_.
"""

from __future__ import print_function

import numpy as np


def clac_helix_rgba(hue0=420.0, hue1=150.0, sat=0.8, intensity0=0.15,
                    intensity1=0.85, gamma=1.0, sat0=None, sat1=None,
                    start=None, rot=None, nlevels=256):
    """Make a cubehelix color map

    This function make color maps with monotonicly increasing
    perceptual intensity. The coefficients for the basis of constant
    intensity come from the paper [Green2011]_:

    Note:
        If start OR rot are given, then the color hue is calculated
        using the method from the fortran code in the paper. Otherwise,
        the color angles are found such that the color bar roughly ends
        on the colors given for hue0 and hue1.

    Args:
        hue0 (None, float): hue of the lower end 0..360
        hue1 (None, float): hue of the upper end 0..360
        sat (float, optional): color saturation for both endpoints
        intensity0 (float): intensity at the lower end of the mapping
            (same as L^*), 0..1
        intensity1 (float): intensity at the upper end of the mapping
            (same as L^*), 0..1
        gamma (float): gamma correction; gamma < 1 accents darker
            values, while gamma > 1 accents lighter values
        sat0 (None, float): if not None, set the saturation for the
            lower end of the color map (0..1)
        sat1 (None, float): if not None, set the saturation for the
            upper end of the color map (0..1)
        start (float): if given, set hue0 where 0.0, 1.0, 2.0, 3.0 mean
            blue, red, greed, blue (values are 1.0 + hue / 120.0).
        rot (float): if given, set hue1 such that over the course of
            the color map, the helix goes over rot hues. Values use the
            same convention as start.
        nlevels (int): number of points on which to calculate the
            helix. Matplotlib will linearly interpolate intermediate
            values

    Returns:
        Nx4 ndarray of rgba data

    References:
        .. [Green2011] Green, D. A., 2011, "A colour scheme for the display
           of astronomical intensity images", Bulletin of the Astronomical
           Society of India, 39, 289. <http://arxiv.org/abs/1108.5083>

    TODO:
        An option for the ramp of intensity that doesn't rely on gamma.
        I'm thinking or something more like the Weber-Fechner law.
    """
    # set saturation of extrema using sat if necessary
    if sat0 is None:
        sat0 = sat
    if sat1 is None:
        sat1 = sat

    # set start and rot hues from degrees if requested
    original_recipe = start is not None or rot is not None
    if start is None:
        start = 1.0 + hue0 / 120.0
    if rot is None:
        rot = (1.0 + hue1 / 120.0) - start

    # lgam is is lightness**gamma and ssat is the color saturation
    lam = np.linspace(intensity0, intensity1, nlevels)
    lgam = lam**gamma
    ssat = np.linspace(sat0, sat1, nlevels)
    # phi is related to hue along the colormap, but r=1, g=2, b=3
    # a is the amplitude of the helix away from the grayscale diagonal
    if original_recipe:
        phi = (2.0 * np.pi) * (start / 3.0 + 1.0 + rot * lam)
    else:
        phi = (2.0 * np.pi / 3.0) * np.linspace(start, start + rot, nlevels)
    a = 0.5 * ssat * lgam * (1.0 - lgam)

    coefs = np.array([[-0.14861, +1.78277],
                      [-0.29227, -0.90649],
                      [+1.97294, +0.00000]])
    helix_vec = np.array([np.cos(phi), np.sin(phi)])
    # rgb will be a 3xN ndarray
    rgb = lgam + a * np.dot(coefs, helix_vec)

    # limit rgb values to the range [0.0, 1.0]
    rgb[np.where((rgb < 0.0))] = 0.0
    rgb[np.where((rgb > 1.0))] = 1.0
    # rgb was calculated with shape 3xN, but we want to return shape Nx4
    rgba = np.hstack([rgb.T, np.ones_like(rgb[:1]).T])
    return rgba


redhelix_rgba = clac_helix_rgba(hue0=60, hue1=-150, sat0=1.8, sat1=1.0,
                                gamma=0.8, intensity0=0.0, intensity1=1.0)
bloodhelix_rgba = clac_helix_rgba(hue0=60, hue1=-150, sat0=1.8, sat1=1.0,
                                  gamma=1.0, intensity0=0.0, intensity1=1.0)
coolhelix_rgba = clac_helix_rgba(hue0=350, hue1=150, sat0=2.0, sat1=1.6,
                                 gamma=0.8, intensity0=0.0, intensity1=1.0)
cubeYF_rgba = clac_helix_rgba(hue0=340, hue1=60, sat0=1.0, sat1=2.0,
                              intensity0=0.3, intensity1=0.9, gamma=0.9)

try:
    from matplotlib.colors import ListedColormap
    from matplotlib.cm import get_cmap, register_cmap

    for (name, data) in ((u'redhelix', redhelix_rgba),
                         (u'bloodhelix', bloodhelix_rgba),
                         (u'coolhelix', coolhelix_rgba),
                         (u'cubeYF', cubeYF_rgba)):

        try:
            get_cmap(name)
        except ValueError:
            register_cmap(cmap=ListedColormap(data, name=name),
                          name=name)
            register_cmap(cmap=ListedColormap(list(reversed(data)), name=name),
                          name=name + '_r')
except ImportError:
    pass

##
## EOF
##
