""" A set of pure python modules that aid in plotting gridded scientific data.
Plotting depends on matplotlib and/or mayavi and file reading uses h5py and
to read hdf5 / xdmf files.
"""

__all__ = ['calculator',
           'plot',
           'readers',
           'bucket',
           'coordinate',
           'dataset',
           'field',
           'grid',
           'parallel',
           'verror',
           'vlab',
           'vutil'
          ]

from viscid import readers
load_file = readers.load_file
load_files = readers.load_files
get_file = readers.get_file

from viscid import rc
rc.load_rc_file("~/.viscidrc")
