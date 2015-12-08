"""A set of python modules that aid in plotting scientific data

Plotting depends on matplotlib and/or mayavi and file reading uses h5py
and to read hdf5 / xdmf files.

Note:
    Modules in calculator and plot must be imported explicitly since
    they have side effects on import.

Attributes:
    logger (logging.Logger): a logging object whose verbosity can be
        set from the command line using
        :py:func`viscid.vutil.common_argparse`.
"""

__version__ = """0.95.2 dev"""

__all__ = ['amr_field',  # Modules
           'amr_grid',
           'calculator',
           'readers',
           'bucket',
           'coordinate',
           'dataset',
           'field',
           'grid',
           'parallel',
           'pyeval',
           'seed',
           'verror',
           'vjson',
           'vlab',
           'vutil',
           'logger',  # logger
           'load_file',  # reader helpers
           'load_files',
           'unload_file',
           'reload_file',
           'get_file',
           'save_grid',
           'save_field',
           'save_fields',
           'arrays2field',  # Field helpers
           'dat2field',
           'empty',
           'zeros',
           'ones',
           'empty_like',
           'zeros_like',
           'ones_like',
           'scalar_fields_to_vector',
           'wrap_field',
           'arrays2crds',  # Crd helpers
           'interp_nearest',  # cython helpers
           'interp_trilin',
           'calc_streamlines',
           # viscid.calculator.calc.* is added below
           # viscid.seed.* is added below
          ]

# setup logger for use throughout viscid
import logging
logger = logging.getLogger("viscid")
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(fmt="%(levelname)s: %(message)s"))
logger.addHandler(_handler)


class CustomFilter(logging.Filter, object):
    def filter(self, record):
        spaces = ' ' * (len(record.levelname) + 2)
        record.msg = record.msg.replace('\n', '\n' + spaces)
        return super(CustomFilter, self).filter(record)


logger.addFilter(CustomFilter())
logger.propagate = False
del _handler

# pull file reading helpers into namespace
from viscid import readers
load_file = readers.load_file
load_files = readers.load_files
unload_file = readers.unload_file
reload_file = readers.reload_file
get_file = readers.get_file
save_grid = readers.save_grid
save_field = readers.save_field
save_fields = readers.save_fields

# pull field and coordnate helpers into the namespace
from viscid import field
arrays2field = field.arrays2field
dat2field = field.dat2field
empty = field.empty
zeros = field.zeros
ones = field.ones
empty_like = field.empty_like
zeros_like = field.zeros_like
ones_like = field.ones_like
scalar_fields_to_vector = field.scalar_fields_to_vector
wrap_field = field.wrap_field

from viscid import coordinate
arrays2crds = coordinate.arrays2crds
wrap_crds = coordinate.wrap_crds

from viscid.seed import *  # pylint: disable=wildcard-import
from viscid import seed
__all__ += seed.__all__

from viscid.calculator.cluster import cluster
__all__ += ["cluster"]

from viscid.calculator.topology import topology2color
__all__ += ["topology2color"]

from viscid.calculator.separator import topology_bitor_clusters
from viscid.calculator.separator import get_sep_pts_bitor
from viscid.calculator.separator import get_sep_pts_bisect
__all__ += ["topology_bitor_clusters", "get_sep_pts_bitor"]
__all__ += ["get_sep_pts_bisect"]

from viscid.calculator.plasma import *
from viscid.calculator import plasma
__all__ += plasma.__all__

from viscid.calculator.calc import *  # pylint: disable=wildcard-import
from viscid.calculator import calc
__all__ += calc.__all__

from viscid.cython import interp_nearest
from viscid.cython import interp_trilin
from viscid.cython import calc_streamlines

from viscid.cython import streamline
for attr in dir(streamline):
    if attr[0] != '_' and attr.isupper():
        vars()[attr] = getattr(streamline, attr)
        __all__.append(attr)
del streamline

# always bring in custom matplotlib stuff (colormaps & rc params)
from viscid.plot import mpl_style

# pull other useful modules into the namespace
# Note: plot and calculator are intentionally left
#       out of the viscid namespace since importing
#       some of these modules (like mpl and mvi) have
#       side effects
from viscid import amr_field
from viscid import amr_grid
from viscid import bucket
from viscid import dataset
from viscid import grid
from viscid import parallel
from viscid import pyeval
from viscid import tree
from viscid import verror
from viscid import vjson
from viscid import vlab
from viscid import vutil

# apply settings in the rc file
from viscid import _rc
_rc.load_rc_file("~/.viscidrc")
