""" docstring for readers """
# To add a new file type, subclass VFile and import the module here.
# This registers the file type class as a subclass of VFile, and
# it automatically becomes part of the file detection cascade...
# Note that subclasses are given precedence in type detection, so
# care must be taken when crafting the detector regex.
# Also, look at csv for an example of overriding detect_type(...)

# import vfile
from .vfile import VFile
from . import vfile_bucket

# these imports are necessary to register file types
from . import xdmf
from . import hdf5
from . import numpy_binary
from . import ascii

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

def load_file(fname, **kwargs):
    """ Load a single file and return a vFille.
    This function is the primary means of loading a file.
    kwargs is passed to File constructor for changing the
    grid_type and stuff.
    """
    return __filebucket__.load_file(fname, **kwargs)

def load_files(fnames, **kwargs):
    """ Load a single file and return a vFille.
    This function is the primary means of loading a file.
    kwargs is passed to File constructor for changing the
    grid_type and stuff.
    """
    return __filebucket__.load_files(fnames, **kwargs)

def save_grid(fname, grd, **kwargs):
    """ save a grid, filetype is inferred from fname """
    ftype = VFile.detect_type(fname)
    ftype.save_grid(fname, grd, **kwargs)

def save_field(fname, fld, **kwargs):
    """ save a field, filetype is inferred from fname """
    ftype = VFile.detect_type(fname)
    ftype.save_field(fname, fld, **kwargs)

def save_fields(fname, flds, **kwargs):
    """ save a list of fields, filetype is inferred from fname """
    ftype = VFile.detect_type(fname)
    ftype.save_fields(fname, flds, **kwargs)
