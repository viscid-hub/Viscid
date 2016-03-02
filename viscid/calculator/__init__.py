"""Calculate using fields

This package has some general purpose calculator stuff, like
interpolation, streamline tracing, Div, Curl, calculate flux function,
etc.

"""

__all__ = ["calc", "cluster", "evaluator", "plasma", "topology"]

from viscid.calculator import evaluator
from viscid.calculator import calc
from viscid.calculator import cluster
from viscid.calculator import plasma
from viscid.calculator import minvar_tools
from viscid.calculator import mpause
from viscid.calculator import separator
from viscid.calculator import topology
from viscid import seed

from viscid.calculator.evaluator import evaluate
from viscid.calculator.calc import *  # pylint: disable=wildcard-import
from viscid.calculator.plasma import *  # pylint: disable=wildcard-import
from viscid.calculator.minvar_tools import *  # pylint: disable=wildcard-import
from viscid.calculator.mpause import *  # pylint: disable=wildcard-import
from viscid.calculator.topology import *  # pylint: disable=wildcard-import
from viscid.seed import *  # pylint: disable=wildcard-import

# # I'm not sure this should be public since it should be accessed through
# # viscid.calculator.calc
# try:
#     from viscid.calculator import necalc
# except ImportError:
#     necalc = _dummy("numexpr not installed")

# import the cython code for legacy
from viscid.cython import interp_nearest
from viscid.cython import interp_trilin
from viscid.cython import calc_streamlines

from viscid.cython import cycalc
from viscid.cython import streamline

##
## EOF
##
