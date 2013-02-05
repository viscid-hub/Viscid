from __future__ import print_function
from warnings import warn

from . import vfile

HAS_H5PY = False
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    warn("h5py not in use, no hdf5 support.", ImportWarning)


class FileHDF5(vfile.VFile):
    """  """
    _detector = '.*\.h5\s*$'

    def __init__(self, fname, **kwargs):
        assert(HAS_H5PY)
        super(FileHDF5, self).__init__(fname, **kwargs)

    def _parse(self):
        # any subclass of this can do anything it wants, BUT
        # xdmf requires that the FileHDF5.file is set to the
        # correct instance of h5py.File
        self.file = h5py.File(self.fname, 'r')

    def get_data(self, handle):
        return self.file[handle]

##
## EOF
##
