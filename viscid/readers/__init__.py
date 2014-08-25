""" docstring for readers """
# To add a new file type, subclass VFile and import the module here.
# This registers the file type class as a subclass of VFile, and
# it automatically becomes part of the file detection cascade...
# Note that subclasses are given precedence in type detection, so
# care must be taken when crafting the detector regex.
# Also, look at csv for an example of overriding detect_type(...)

__DEBUG_IMPORT_ERRORS = False

# import vfile
from viscid.readers.vfile import VFile
from viscid.readers import vfile_bucket

# these imports are necessary to register file types
from viscid.readers import xdmf
from viscid.readers import hdf5
from viscid.readers import numpy_binary
from viscid.readers import ascii

# these imports register convenience readers for data from
# specific sim packages
from viscid.readers import ggcm_xdmf
from viscid.readers import ggcm_fortbin
from viscid.readers import psc
try:
    from viscid.readers import ggcm_jrrle
except ImportError:
    if __DEBUG_IMPORT_ERRORS:
        raise

__filebucket__ = vfile_bucket.VFileBucket()

def load(fnames):
    """Generic load

    Dispatches to :meth:`load_file` or :meth:`load_files`. This
    function is deprecated.

    Parameters:
        fnames: One or many file names

    Returns:
        one or many VFiles
    """
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
    """Load a file

    Parameters:
        fname (str): a file name, relative to CWD
        kwargs: passed to the VFile constructor

    Returns:
        A VFile instance
    """
    return __filebucket__.load_file(fname, **kwargs)

def load_files(fnames, **kwargs):
    """Load a list of files

    Parameters:
        fnames (list): list of file names, glob patterns accepted
        kwargs: passed to the VFile constructor

    Returns:
        A list of VFile instances. The length may not be the same
        as the length of fnames, and the order may not be the same
        in order to accomidate globs and file grouping.
    """
    return __filebucket__.load_files(fnames, **kwargs)

def get_file(handle):
    """ return a file that's already been loaded by either
    number (as in nth file loaded), of file name
    """
    return __filebucket__[handle]

def save_grid(fname, grd, **kwargs):
    """ save a grid, filetype is inferred from fname
    """
    ftype = VFile.detect_type(fname)
    ftype.save_grid(fname, grd, **kwargs)

def save_field(fname, fld, **kwargs):
    """ save a field, filetype is inferred from fname"""
    ftype = VFile.detect_type(fname)
    ftype.save_field(fname, fld, **kwargs)

def save_fields(fname, flds, **kwargs):
    """ save a list of fields, filetype is inferred from fname
    """
    ftype = VFile.detect_type(fname)
    ftype.save_fields(fname, flds, **kwargs)

def unload_all_files():
    __filebucket__.remove_all_items()
