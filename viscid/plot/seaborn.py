"""Graceful way to apply seaborn to mpl if you have it installed

This is mostly controlled with parameters in your rc script

After activate_from_viscid is called, if the seaborn import is
successful, then the entire namespace is imported here and
seaborn can be used from this module.
"""

import imp

from viscid import logger

SEABORN_ACTIVATED = False
try:
    imp.find_module('seaborn')
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

##############################################
# Global config; override before importing
# mpl or callign activate_from_visicd either
#  in script, or in rc file
##############################################
enabled = False
context = "notebook"
style = "darkgrid"
palette = "deep"
font = "sans-serif"
font_scale = 1
rc = {}  # this one can't be set from rc file
##############################################

def activate_from_viscid():
    """You should not need to call this

    This function is called by viscid.plot.mpl at time of import. All
    you need to do is set the global config options above before
    importing viscid.plot.mpl. This can be done with the rc file.
    """
    if enabled:
        try:
            import seaborn as real_seaborn
            real_seaborn.set(context=context, style=style,
                             palette=palette, font=font,
                             font_scale=font_scale, rc=rc)
            # now pull in the public namespace
            g = globals()
            for key, value in real_seaborn.__dict__.items():
                if not key.startswith("_"):
                    g[key] = value
            g['SEABORN_ACTIVATED'] = True
        except ImportError:
            logger.warn("seaborn package not installed")

##
## EOF
##
