from __future__ import print_function
import logging

from . import vfile

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    logging.warn("h5py library not found, no hdf5 support.")


class H5pyDataWrapper(vfile.DataWrapper): #pylint: disable=R0924
    """  """
    fname = None
    loc = None

    def __init__(self, fname, loc):
        super(H5pyDataWrapper, self).__init__()
        self.fname = fname
        self.loc = loc

        with h5py.File(self.fname, 'r') as f:
            dset = f[loc]
            self._shape = dset.shape
            self._dtype = dset.dtype

    def wrap_func(self, func_name, *args, **kwargs):
        with h5py.File(self.fname, 'r') as f:
            return getattr(f[self.loc], func_name)(*args, **kwargs)

    def __array__(self, *args, **kwargs):
        return self.wrap_func("__array__", *args, **kwargs)

    def read_direct(self, *args, **kwargs):
        return self.wrap_func("read_direct", *args, **kwargs)

    def len(self):
        return self.wrap_func("len")

    def __getitem__(self, item):
        return self.wrap_func("__getitem__", item)


class FileHDF5(vfile.VFile):
    """  """
    _detector = r".*\.h5\s*$"

    def __init__(self, fname, **kwargs):
        assert(HAS_H5PY)
        super(FileHDF5, self).__init__(fname, **kwargs)

    def _parse(self):
        # any subclass of this can do anything it wants, BUT
        # xdmf requires that the FileHDF5.file is set to the
        # correct instance of h5py.File
        # self.file = h5py.File(self.fname, 'r')
        pass

    def get_data(self, handle):
        return H5pyDataWrapper(self.fname, handle)
        # return h5py.File(self.fname, 'r')[handle]

    # def __getitem__(self, handle):
    #     return self.get_data(handle)

##
## EOF
##
