""" A set of pure python modules that aid in plotting gridded scientific data.
Plotting depends on matplotlib and/or mayavi and file reading uses h5py and
to read hdf5 / xdmf files.
"""

__all__ = ['amr_field',
           'amr_grid',
           'calculator',
           'plot',
           'readers',
           'bucket',
           'coordinate',
           'dataset',
           'field',
           'grid',
           'parallel',
           'parsers',
           'verror',
           'vlab',
           'vutil'
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

# apply settings in the rc file
from viscid import rc
rc.load_rc_file("~/.viscidrc")
