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
    load_file (function): convience reference for
        :py:func:`viscid.readers.load_file`
    load_files (function): convience reference for
        :py:func:`viscid.readers.load_files`
    get_file (function): convience reference for
        :py:func:`viscid.readers.get_file`
    save_grid (function): convience reference for
        :py:func:`viscid.readers.save_grid`
    save_field (function): convience reference for
        :py:func:`viscid.readers.save_field`
    save_fields (function): convience reference for
        :py:func:`viscid.readers.save_fields`
    arrays2field (function): convience reference for
        :py:func:`viscid.field.arrays2field`
    dat2field (function): convience reference for
        :py:func:`viscid.field.dat2field`
    empty (function): convience reference for
        :py:func:`viscid.field.empty`
    zeros (function): convience reference for
        :py:func:`viscid.field.zeros`
    ones (function): convience reference for
        :py:func:`viscid.field.ones`
    empty_like (function): convience reference for
        :py:func:`viscid.field.empty_like`
    zeros_like (function): convience reference for
        :py:func:`viscid.field.zeros_like`
    ones_like (function): convience reference for
        :py:func:`viscid.field.ones_like`
    scalar_fields_to_vector (function): convience reference for
        :py:func:`viscid.field.scalar_fields_to_vector`
    wrap_field (function): convience reference for
        :py:func:`viscid.field.wrap_field`
    arrays2crds (function): convience reference for
        :py:func:`viscid.coordinate.arrays2crds`
"""

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
          ]

# setup logger for use throughout viscid
import logging
logger = logging.getLogger("viscid")
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter())
logger.addHandler(_handler)
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
from viscid import tree
from viscid import verror
from viscid import vjson
from viscid import vlab
from viscid import vutil

# apply settings in the rc file
from viscid import _rc
_rc.load_rc_file("~/.viscidrc")
