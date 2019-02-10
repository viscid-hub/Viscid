"""Package for the various data readers"""

# To add a new file type, subclass VFile and import the module here.
# This registers the file type class as a subclass of VFile, and
# it automatically becomes part of the file detection cascade...
# Note that subclasses are given precedence in type detection, so
# care must be taken when crafting the detector regex.
# Also, look at csv for an example of overriding detect_type(...)

import viscid
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
from viscid.readers import gkeyll
from viscid.readers import athena_bin
from viscid.readers import athena_tab
from viscid.readers import athena_hst
from viscid.readers import athena_xdmf
from viscid.readers import ggcm_jrrle
from viscid.readers import vpic


__all__ = ['load_file', 'load_files', 'unload_file', 'unload_all_files',
           'reload_file', 'get_file', 'save_grid', 'save_field', 'save_fields']


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
    # this is not a deprecated warning since by default those aren't shown
    # and i want this to be a loud and clear do not use :)
    viscid.logger.warning("readers.load is deprecated in favor of load_file or "
                          "load_files")
    files = load_files(fnames)
    if isinstance(fnames, (list, tuple)):
        return files
    else:
        return files[0]

def load_file(fname, force_reload=False, **kwargs):
    """Load a file

    Parameters:
        fnames (list): single file name, or list of files that are part
            of the same time series. Glob patterns and slices are
            accepted, see :doc:`/tips_and_tricks` for more info.
        fname (str): a file name, relative to CWD
        force_reload (bool): Force reload if file is already in memory
        **kwargs: passed to the VFile constructor

    See Also:
        * :doc:`/tips_and_tricks`

    Returns:
        A VFile instance
    """
    return __filebucket__.load_file(fname, force_reload=force_reload, **kwargs)

def load_files(fnames, force_reload=False, **kwargs):
    """Load a list of files

    Parameters:
        fnames (list): list of file names. Glob patterns and slices are
            accepted, see :doc:`/tips_and_tricks` for more info.
        force_reload (bool): Force reload if file is already in memory
        **kwargs: passed to the VFile constructor

    See Also:
        * :doc:`/tips_and_tricks`

    Returns:
        A list of VFile instances. The length may not be the same
        as the length of fnames, and the order may not be the same
        in order to accommodate globs and file grouping.
    """
    return __filebucket__.load_files(fnames, force_reload=force_reload,
                                     **kwargs)

def unload_file(handle):
    """call unload on the handle in the bucket"""
    __filebucket__[handle].unload()

def reload_file(handle):
    """call reload on the handle in the bucket"""
    __filebucket__[handle].reload()

def get_file(handle):
    """ return a file that's already been loaded by either
    number (as in nth file loaded), of file name
    """
    return __filebucket__[handle]

def save_grid(fname, grd, **kwargs):
    """ save a grid, filetype is inferred from fname
    """
    ftype = VFile.detect_type(fname, mode='w', prefer=kwargs.pop('prefer', None))
    ftype.save_grid(fname, grd, **kwargs)

def save_field(fname, fld, **kwargs):
    """ save a field, filetype is inferred from fname"""
    ftype = VFile.detect_type(fname, mode='w', prefer=kwargs.pop('prefer', None))
    ftype.save_field(fname, fld, **kwargs)

def save_fields(fname, flds, **kwargs):
    """ save a list of fields, filetype is inferred from fname
    """
    ftype = VFile.detect_type(fname, mode='w', prefer=kwargs.pop('prefer', None))
    ftype.save_fields(fname, flds, **kwargs)

def unload_all_files():
    """Hammer-of-Thor the cache"""
    __filebucket__.remove_all_items()
