#!/usr/bin/env python

from __future__ import print_function
import os
from glob import glob

from viscid import logger
from viscid.compat import string_types
from viscid.bucket import Bucket
from viscid.readers.vfile import VFile
from viscid.compat import OrderedDict


class VFileBucket(Bucket):
    """ manages open files, create / get with get_file_bucket() as you
    generally only need one instance, but you can construct directly
    if you need more than one manager
    """

    def __init__(self, **kwargs):
        super(VFileBucket, self).__init__(ordered=True, **kwargs)

    # This routine is just sort of confusing
    # def add(self, fname, file):
    #     absfname = os.path.abspath(fname)
    #     self[(absfname, fname)] = f

    def load_file(self, fname, index_handle=True, **kwargs):
        """ load a single file and return a vFile instance, not a list
        of vFiles like load does
        """
        fls = self.load_files(fname, index_handle=index_handle, **kwargs)
        if len(fls) == 0:
            return None
        else:
            if len(fls) > 1:
                logger.warn("Loaded > 1 file for '{0}', did you mean to call "
                            "load_files()?".format(fname))
            return fls[0]

    def load_files(self, fnames, index_handle=True, file_type=None, **kwargs):
        """Load files, and add them to the bucket

        Initialize obj before it's put into the list, whatever is returned
        is what gets stored, returning None means object init failed, do
        not add to the _objs list

        Parameters:
            fnames: a list of file names (can cantain glob patterns)
            index_handle: ??
            file_type: a class that is a subclass of VFile, if given,
                use this file type, don't use the autodetect mechanism
            kwargs: passed to file constructor

        Returns:
            A list of VFile instances. The length may not be the same
            as the length of fnames, and the order may not be the same
            in order to accomidate globs and file grouping.
        """
        orig_fnames = fnames

        if not isinstance(fnames, (list, tuple)):
            fnames = [fnames]
        file_lst = []

        # glob and convert to absolute paths
        globbed_fnames = []
        for fname in fnames:
            expanded_fname = os.path.expanduser(os.path.expandvars(fname))
            absfname = os.path.abspath(expanded_fname)
            if '*' in absfname or '?' in absfname:
                globbed_fnames += glob(absfname)
            else:
                globbed_fnames += [absfname]
            # Is it necessary to recall abspath here? We did it before
            # the glob to make sure it didn't start with a '.' since that
            # tells glob not to fill wildcards
        fnames = globbed_fnames

        # detect file types
        types_detected = OrderedDict()
        for i, fname in enumerate(fnames):
            _ftype = None
            if file_type is None:
                _ftype = VFile.detect_type(fname)
            else:
                _ftype = file_type
            if not _ftype:
                raise RuntimeError("Can't determine type "
                                   "for {0}".format(fname))
            value = (fname, i)
            try:
                types_detected[_ftype].append(value)
            except KeyError:
                types_detected[_ftype] = [value]

        # see if the file's already been loaded, or load it, and add it
        # to the bucket and all that good stuff
        file_lst = []
        for ftype, vals in types_detected.items():
            names = [v[0] for v in vals]
            # group all file names of a given type
            groups = ftype.group_fnames(names)

            # iterate all the groups and add them
            for group in groups:
                f = None

                handle_name = ftype.collective_name(group)

                try:
                    f = self[handle_name]
                except KeyError:
                    try:
                        f = ftype(group, parent_bucket=self, **kwargs)
                    except IOError as e:
                        s = " IOError on file: {0}\n".format(handle_name)
                        s += "              File Type: {0}\n".format(handle_name)
                        s += "              {0}".format(str(e))
                        logger.warn(s)
                    except ValueError as e:
                        # ... why am i explicitly catching ValueErrors?
                        # i'm probably breaking something by re-raising
                        # this exception, but i didn't document what :(
                        s = " ValueError on file load: {0}\n".format(handle_name)
                        s += "              File Type: {0}\n".format(handle_name)
                        s += "              {0}".format(str(e))
                        logger.warn(s)
                        # re-raise the last expection
                        raise

                self.set_item([handle_name], f, index_handle=index_handle)
                file_lst.append(f)

        if len(file_lst) == 0:
            logger.warn("No files loaded for '{0}', is the path "
                        "correct?".format(orig_fnames))
        return file_lst

    def remove_item(self, item):
        item.unload()
        super(VFileBucket, self).remove_item(item)

    def remove_all_items(self):
        for val in self.values():
            val.unload()
        super(VFileBucket, self).remove_all_items()

    def __getitem__(self, handle):
        if isinstance(handle, string_types):
            handle = os.path.expanduser(os.path.expandvars(handle))
        return super(VFileBucket, self).__getitem__(handle)

    def __contains__(self, handle):
        if isinstance(handle, string_types):
            handle = os.path.expanduser(os.path.expandvars(handle))
        return super(VFileBucket, self).__contains__(handle)


class ContainerFile(VFile):  # pylint: disable=abstract-method
    """A container file is a VFile that can load other files

    The use case is always something like the relationship between XDMF
    files and HDF5 files. It's nice for an XDMF file to keep track of
    all the HDF5 Files that it refers to.
    """
    child_bucket = None
    _child_files = None
    _child_file_handles = None

    def __init__(self, fname, parent_bucket=None, **kwargs):
        if parent_bucket is None:
            self.child_bucket = VFileBucket()
        else:
            self.child_bucket = parent_bucket
        self._child_files = []
        self._child_file_handles = []
        super(ContainerFile, self).__init__(fname, **kwargs)

    def _load_child_file(self, fname, **kwargs):
        """Add file to self.child_bucket and remember it for when I unload"""
        f = self.child_bucket.load_file(fname, **kwargs)
        self._child_files.append(f)
        # self._child_file_handles.append(???)
        return f

    def unload(self):
        # for child in self._child_file_handles:
        #     pass
        super(ContainerFile, self).unload()

    def clear_cache(self):
        # for child in self._child_file_handles:
        #     pass
        super(ContainerFile, self).clear_cache()

##
## EOF
##
