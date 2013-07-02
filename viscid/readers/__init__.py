""" docstring for readers """
# To add a new file type, subclass VFile and import the module here.
# This registers the file type class as a subclass of VFile, and
# it automatically becomes part of the file detection cascade...
# Note that subclasses are given precedence in type detection, so
# care must be taken when crafting the detector regex.
# Also, look at csv for an example of overriding detect_type(...)

# import vfile
from . import vfile_bucket

# these imports are necessary to register file types
from . import csv
from . import hdf5
from . import xdmf


__filebucket__ = vfile_bucket.VFileBucket()


def load(fname):
    """ This function is the primary means of loading a file """
    fle = __filebucket__.load(fname)
    return fle

# load = load_vfile
