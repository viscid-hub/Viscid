#!/usr/bin/env python
"""Apply some viscid-specific matplotlib styling

This module provides some extra style sheets as well as a layer to
set viscid-specific rc parameters and activate specific style sheets
from your `viscidrc` file.

Attributes:
    use_styles (sequence): a list of style sheet names to activate
    rc_params (dict): dictionary of parameters that get directly
        injected into matplotlib.rcParams
    rc (dict): specify rc parameters through matplotlib.rc(...). Keys
        are groups and the values should be dictionaries that will be
        unpacked into the rc function call.

Style Sheets:
    The following style sheets are available to matplotlib. Style
    sheets require matplotlib version >= 1.5.0

    - **viscid-default**: use afmhot
    - **viscid-colorblind**: use colorsequences that are colorblind
      friendly
    - **viscid-steve**: use Steve's label formatter for colorbars

Rc Parameters:
    Here are some viscid-specific options that can be put into the
    matplotlib rcParams

    - **viscid.cbarfmt**: Formatter for colorbars
    - **viscid.majorfmt**: Major tick formatter
    - **viscid.minorfmt**: Minor tick formatter
    - **viscid.majorloc**: Major tick locator
    - **viscid.minorloc**: Minor tick locator
    - **image.cmap**: Default colormap
    - **viscid.symmetric_cmap**: Default colormap for plots with
       symmetric limits
"""
# matplotlib's style sheets can be browsed at:
# https://github.com/matplotlib/matplotlib/tree/HEAD/lib/matplotlib/mpl-data/stylelib

from __future__ import division, print_function

import matplotlib
from viscid import logger
from viscid.plot.cmap_tools import register_cmap
from viscid.plot import cubehelix  # import clac_helix_rgba


# Set these in your RC file to
use_styles = ["viscid-default"]
rc_params = {}
rc = {}

# setup viscid-specific matplotlib rc entries
viscid_mpl_rc_params = {
    u"viscid.cbarfmt": [u"", unicode],
    u"viscid.majorfmt": [u"", unicode],
    u"viscid.minorfmt": [u"", unicode],
    u"viscid.majorloc": [u"", unicode],
    u"viscid.minorloc": [u"", unicode],
    u"viscid.symmetric_cmap": [u"", unicode]
}
for key, default_converter in viscid_mpl_rc_params.items():
    matplotlib.defaultParams[key] = default_converter
    matplotlib.rcParams.validate[key] = default_converter[1]
    matplotlib.rcParamsDefault.validate[key] = default_converter[1]
    matplotlib.rcParams[key] = default_converter[0]
    matplotlib.rcParamsOrig[key] = default_converter[0]
    matplotlib.rcParamsDefault[key] = default_converter[0]

VISCID_STYLES = {
    u"viscid-default": u"""
viscid.symmetric_cmap: RdBu_r
""",
    u"viscid-colorblind": u"""
axes.prop_cycle: cycler('color', ['004358', 'FD7400', '3DA88E', \
                                  '83522B', '00D4FD', 'E2D893'])
image.cmap: redhelix
viscid.symmetric_cmap: RdBu_r
""",
    u"viscid-steve": u"""
viscid.cbarfmt: steve
"""
}

# register Viscid's special colormaps
# register the cubehelix color maps
register_cmap('cubeYF', cubehelix.cubeYF_rgba, reverse=False)
register_cmap('cubeYF_r', cubehelix.cubeYF_rgba, reverse=True)

register_cmap('coolhelix', cubehelix.coolhelix_rgba, reverse=False)
register_cmap('coolhelix_r', cubehelix.coolhelix_rgba, reverse=True)

register_cmap('redhelix', cubehelix.redhelix_rgba, reverse=False)
register_cmap('redhelix_r', cubehelix.redhelix_rgba, reverse=True)

register_cmap('bloodhelix', cubehelix.bloodhelix_rgba, reverse=False)
register_cmap('bloodhelix_r', cubehelix.bloodhelix_rgba, reverse=True)


def inject_viscid_styles(show_warning=True):
    try:
        import tempfile
        import os

        from matplotlib import rc_params_from_file, style

        tmpfname = tempfile.mkstemp()[1]
        for style_name, text in VISCID_STYLES.items():
            with open(tmpfname, 'w') as f:
                f.write(text)
            params = rc_params_from_file(tmpfname, use_default_template=False)
            style.library[style_name] = params
        os.unlink(tmpfname)
        style.reload_library()
    except ImportError:
        if show_warning:
            logger.debug("Upgrade to matplotlib >= 1.5.0 to use style sheets")

def post_rc_actions(show_warning=True):
    try:
        from matplotlib import style
        for s in use_styles:
            style.use(s)
    except ImportError:
        if show_warning and use_styles:
            logger.warn("Upgrade to matplotlib >= 1.5.0 to use style sheets")

    matplotlib.rcParams.update(rc_params)
    for group, params in rc.items():
        matplotlib.rc(group, **params)

inject_viscid_styles()
post_rc_actions(show_warning=False)

##
## EOF
##
