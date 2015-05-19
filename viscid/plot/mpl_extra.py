"""Some common customizations for matplotlib

Include Steve's tick formatting. This maybe should go somewhere else.

Register some perceptual cubehelix colormaps with matplotlib. These
are good for the colorblind ;)

Attributes:
    default_cmap (str): set from viscidrc to set default cmap by name
    symmetric_cmap (str): if given, use this colormap for plots
        symmetric about 0
    default_cbarfmt (str, ticker.Formatter): 'steve' or something else
    default_majorfmt (str, ticker.Formatter): 'steve' or something else
    default_minorfmt (str, ticker.Formatter): 'steve' or something else
    default_majorloc (ticker.Locator): default locator if given
    default_minorloc (ticker.Locator): default locator if given
    use_steve_cbarfmt (bool): set from viscidrc
    use_steve_axfmt (bool): set from viscidrc
    steve_cbarfmt: Steve's colorbar tick formatter
    steve_axfmt: Steve's axis tick formatter

Note:
    All of the above maps should have '_r' variants as well. In addition,
    this module has an attribute cmapname_rgb with the rgba ndarray data.

Note:
    The **default_cmap**, **use_steve_cbarfmt** and **use_steve_axfmt**
    attributes are intended to be set via the rc file.
"""

from __future__ import print_function
import locale
import math

import numpy as np
import matplotlib.ticker as ticker

from viscid.plot.cmap_tools import register_cmap
from viscid.plot import cubehelix # import clac_helix_rgba


# NOTE: this changes the default colormap as soon as viscid.plot.mpl is
#       imported... Use your rc file to set this to a specific map or None
#       if you want to use matplotlib's default
default_cmap = "afmhot"
symmetric_cmap = "seismic"

default_cbarfmt = None
default_majorfmt = None
default_minorfmt = None
default_majorloc = None
default_minorloc = None

# register the cubehelix color maps
register_cmap('cubeYF', cubehelix.cubeYF_rgba, reverse=False)
register_cmap('cubeYF_r', cubehelix.cubeYF_rgba, reverse=True)

register_cmap('coolhelix', cubehelix.coolhelix_rgba, reverse=False)
register_cmap('coolhelix_r', cubehelix.coolhelix_rgba, reverse=True)

register_cmap('redhelix', cubehelix.redhelix_rgba, reverse=False)
register_cmap('redhelix_r', cubehelix.redhelix_rgba, reverse=True)

register_cmap('bloodhelix', cubehelix.bloodhelix_rgba, reverse=False)
register_cmap('bloodhelix_r', cubehelix.bloodhelix_rgba, reverse=True)

class OrigSteveScalarFormatter(ticker.ScalarFormatter):
    def pprint_val(self, x):
        a, b = '{0:.2e}'.format(x-self.offset).split('e')
        b = int(b)
        if b > 2 or b < -2:
            return r'${0} \times 10^{{{1}}}$'.format(a, b)
        else:
            return r'${0:0.2f}$'.format(x-self.offset)

    def _formatSciNotation(self, s):
        # transform 1e+004 into 1e4, for example
        if self._useLocale:
            decimal_point = locale.localeconv()['decimal_point']
            positive_sign = locale.localeconv()['positive_sign']
        else:
            decimal_point = '.'
            positive_sign = '+'
        tup = '{0:.2e}'.format(float(s)).split('e')
        try:
            significand = tup[0].rstrip('0').rstrip(decimal_point)
            significand = '{0:.2f}'.format(float(significand))
            sign = tup[1][0].replace(positive_sign, '')
            exponent = tup[1][1:].lstrip('0')
            #if significand == '1':
               # reformat 1x10^y as 10^y
                #significand = ''
            if exponent:
                exponent = r'10^{%s%s}' % (sign, exponent)
            if significand and exponent:
                return r'$%s{\times}%s$' % (significand, exponent)
            else:
                return r'$%s%s$' % (significand, exponent)

        except IndexError:
            return s

# # fancy tick formatter
class SteveScalarFormatter(ticker.ScalarFormatter):
    def __init__(self, max_sigfigs=2, useMathText=True, **kwargs):
        super(SteveScalarFormatter, self).__init__(useMathText=useMathText,
                                                   **kwargs)
        self.max_sigfigs = max_sigfigs
        self.sigfigs = max_sigfigs

    def _set_format(self, vmin, vmax):
        super(SteveScalarFormatter, self)._set_format(vmin, vmax)
        dot_loc = self.format.rfind('.')
        f_loc = self.format.rfind('f', dot_loc)
        sigfigs = int(self.format[dot_loc + 1:f_loc])
        self.sigfigs = min(sigfigs, self.max_sigfigs)

    def pprint_val(self, x):
        a, b = '{0:0.{1}e}'.format(x - self.offset, self.sigfigs).split('e')
        b = int(b)
        if b < -self.max_sigfigs + 1 or b > self.max_sigfigs:
            s = (r'{0}$\mathdefault{{\times}}$10'
                 r'$^\mathdefault{{{{{1}}}}}$'.format(a, b))
        else:
            s = "{0:0.{1}f}".format(x - self.offset, self.sigfigs)
        return s

    def _formatSciNotation(self, s):
        # transform 1e+004 into 1e4, for example
        if self._useLocale:
            decimal_point = locale.localeconv()['decimal_point']
            positive_sign = locale.localeconv()['positive_sign']
        else:
            decimal_point = '.'
            positive_sign = '+'
        tup = '{0:.2e}'.format(float(s)).split('e')
        try:
            significand = tup[0].rstrip('0').rstrip(decimal_point)
            sign = tup[1][0].replace(positive_sign, '')
            exponent = tup[1][1:].lstrip('0')
            if self._useMathText or self._usetex:
                # if significand == '1' and exponent != '':
                #     # reformat 1x10^y as 10^y
                #     significand = ''
                if exponent:
                    exponent = '10^{%s%s}' % (sign, exponent)
                if significand and exponent:
                    return r'%s{\times}%s' % (significand, exponent)
                else:
                    return r'%s%s' % (significand, exponent)
            else:
                s = ('%se%s%s' % (significand, sign, exponent)).rstrip('e')
                return s
        except IndexError:
            return s

steve_cbarfmt = SteveScalarFormatter(max_sigfigs=2, useOffset=True)
steve_axfmt = SteveScalarFormatter(max_sigfigs=2, useOffset=False)


def _main():
    """Kick the tires on our perceptual colormaps"""
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
