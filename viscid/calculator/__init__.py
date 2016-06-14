"""Calculate using fields

This package has some general purpose calculator stuff, like
interpolation, streamline tracing, Div, Curl, calculate flux function,
etc.

"""

from viscid import import_injector

# seed import are for legacy code
from viscid import seed
from viscid.seed import *  # pylint: disable=wildcard-import

# import the cython code for legacy
from viscid.cython import interp_nearest
from viscid.cython import interp_trilin
from viscid.cython import calc_streamlines
from viscid.cython import cycalc
from viscid.cython import streamline

# Note: necalc is left out of __all__ on purpose

__all__ = ["calc",
           "cluster",
           "ecfc",
           "evaluator",
           "minvar_tools",
           "mpause",
           "plasma",
           "separator",
           "topology"]

import_injector(__all__, globals(), package="viscid.calculator")

##
## EOF
##
