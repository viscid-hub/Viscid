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

def load(fnames):
    from warnings import warn
    # this is not a deprecated warning since by default those aren't shown
    # and i want this to be a loud and clear do not use :)
    warn("readers.load deprecated in favor of load_file or load_files",
         stacklevel=2)
    files = load_files(fnames)
    if isinstance(fnames, (list, tuple)):
        return files
    else:
        return files[0]

def load_file(fname):
    """ Load a single file and return a vFille.
    This function is the primary means of loading a file. """
    return __filebucket__.load_file(fname)

def load_files(fnames):
    """ Load a single file and return a vFille.
    This function is the primary means of loading a file. """
    return __filebucket__.load_files(fnames)
