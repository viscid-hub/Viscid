"""Register some perceptual colormaps with matplotlib

The cubehelix colormaps are good for the colorblind because lightness
monotonically increases throughout the scale. These maps also guard
against levels appearing in the data due to a local maxima in the
lightness of that color in the color map.

The additional color maps are:

* **cubeYF**: My attempt to replicate the cubeYF map described at
  `mycarta <https://mycarta.wordpress.com/2013/02/21/perceptual-
  rainbow-palette-the-method/>`_.
* **coolhelix**: From bordeax to cool white, counter-clockwise in hue
* **redhelix**: From red to cyan, counter-clockwise in hue
* **bloodhelix**: Same as redhelix, but with gamma == 1, so the
  intensity ramp is more linear instead of power law

All of the above maps should have '_r' variants as well. In addition,
this module has an attribute cmapname_rgb with the rgba ndarray data.

Note:
    The **default_cmap** attribute is intended to be set via the rc file.
"""

from __future__ import print_function

from viscid.plot.cubehelix import clac_helix_rgba


# NOTE: this changes the default colormap as soon as viscid.plot.mpl is
#       imported... Use your rc file to set this to a specific map or None
#       if you want to use matplotlib's default
default_cmap = "redhelix"

cubeYF_rgba = clac_helix_rgba(hue0=340, hue1=60, sat0=1.0, sat1=2.0,
                              intensity0=0.3, intensity1=0.9, gamma=0.9)
coolhelix_rgba = clac_helix_rgba(hue0=350, hue1=150, sat0=2.0, sat1=1.6,
                                 gamma=0.8, intensity0=0.0, intensity1=1.0)
redhelix_rgba = clac_helix_rgba(hue0=60, hue1=-150, sat0=1.8, sat1=1.0,
                                gamma=0.8, intensity0=0.0, intensity1=1.0)
bloodhelix_rgba = clac_helix_rgba(hue0=60, hue1=-150, sat0=1.8, sat1=1.0,
                                  gamma=1.0, intensity0=0.0, intensity1=1.0)

try:
    from viscid.plot.cmap_tools import register_cmap

    register_cmap('cubeYF', cubeYF_rgba, reverse=False)
    register_cmap('cubeYF_r', cubeYF_rgba, reverse=True)

    register_cmap('coolhelix', coolhelix_rgba, reverse=False)
    register_cmap('coolhelix_r', coolhelix_rgba, reverse=True)

    register_cmap('redhelix', redhelix_rgba, reverse=False)
    register_cmap('redhelix_r', redhelix_rgba, reverse=True)

    register_cmap('bloodhelix', bloodhelix_rgba, reverse=False)
    register_cmap('bloodhelix_r', bloodhelix_rgba, reverse=True)

except ImportError:
    # cmap_tools will fail to import if matplotlib is not installed, but
    # if default_cmap is in the rc file, then this module will be loaded.
    # In that case, quietly pretend everything is ok... the user will get
    # an ImportError if they try to use mpl without matplotlib anyway
    pass


def _main():
    from matplotlib import pyplot as plt
    from matplotlib import gridspec
    try:
        import seaborn  # pylint: disable=unused-variable
    except ImportError:
        pass
    from viscid.plot.cmap_tools import plot_ab, plot_lstar, plot_rgb
    # from viscid.plot.cmap_tools import to_linear_cmap, to_rgba

    names = ['cubehelix', 'cubeYF', 'coolhelix', 'redhelix', 'bloodhelix']
    gs = gridspec.GridSpec(4, len(names), height_ratios=[1, 3, 3, 3])
    for i, name in enumerate(names):
        cmap = plt.get_cmap(name)
        ax_cmap = plt.subplot(gs[0, i])
        ax_rgb = plt.subplot(gs[1, i])
        ax_ab = plt.subplot(gs[2, i])
        ax_lstar = plt.subplot(gs[3, i])

        plot_rgb(cmap, ax_cmap, ax_rgb)
        plot_ab(cmap, ax_ab)
        plot_lstar(cmap, ax_lstar)
        ax_cmap.set_title(name)
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95,
                        hspace=0.3, wspace=0.3)
    plt.show()
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(_main())

##
## EOF
##
