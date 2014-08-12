#!/usr/bin/env python

from __future__ import print_function
import os
from glob import glob
import logging
import six
# import sys
try:
    from collections import OrderedDict
except ImportError:
    from viscid.compat import OrderedDict

from viscid.bucket import Bucket
from viscid.readers.vfile import VFile


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
        fls = self.load_files([fname], index_handle=index_handle, **kwargs)
        if len(fls) == 0:
            logging.warn("No files loaded for '{0}', is the path "
                         "correct?".format(fname))
            return None
        else:
            if len(fls) > 1:
                logging.warn("Loaded > 1 file for '{0}', did you mean to call "
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
        if not isinstance(fnames, (list, tuple)):
            fnames = [fnames]
        file_lst = []

        # glob and convert to absolute paths
        globbed_fnames = []
        for fname in fnames:
            better_fname = os.path.expanduser(os.path.expandvars(fname))
            absfnames = [os.path.abspath(fn) for fn in glob(better_fname)]
            globbed_fnames += absfnames
        fnames = globbed_fnames

        # detect file types
        types_detected = OrderedDict()
        for i, fname in enumerate(fnames):
            if file_type is None:
                file_type = VFile.detect_type(fname)
            if not file_type:
                raise RuntimeError("Can't determine type "
                                   "for {0}".format(fname))
            value = (fname, i)
            try:
                types_detected[file_type].append(value)
            except KeyError:
                types_detected[file_type] = [value]

        # see if the file's already been loaded, or load it, and add it
        # to the bucket and all that good stuff
        file_lst = []
        for ftype, vals in types_detected.items():
            names = [v[0] for v in vals]
            # group all file names of a given type
            try:
                groups = ftype.group_fnames(names)
            except AttributeError:
                # can't group fnames for this type, that's ok
                groups = names

            # iterate all the groups and add them
            for group in groups:
                f = None
                if isinstance(group, list):
                    # FIXME: if the glob has changed since it was loaded,
                    # this won't know that it has changed... I'm not sure
                    # if there's a good way to figure this out... the user
                    # can always just unload / reload
                    g0 = group[0]
                else:
                    g0 = group

                if g0 in self:
                    f = self[g0]
                else:
                    try:
                        f = ftype(group, vfilebucket=self, **kwargs)
                    except IOError as e:
                        cname = ftype.collective_name(group)
                        s = " IOError on file: {0}\n".format(cname)
                        s += "              File Type: {0}\n".format(file_type)
                        s += "              {0}".format(e.message)
                        logging.warn(s)
                    except ValueError:
                        # what's this about?
                        pass

                try:
                    handle_name = f.collective_name(group)
                except AttributeError:
                    handle_name = VFile.collective_name(group)
                self.set_item([handle_name], f, index_handle=index_handle)
                file_lst.append(f)

        return file_lst

    def remove_item(self, item):
        item.unload()
        super(VFileBucket, self).remove_item(item)

    def remove_all_items(self):
        for item in self._items.keys():
            item.unload()
        super(VFileBucket, self).remove_all_items()

    def __getitem__(self, handle):
        if isinstance(handle, six.string_types):
            handle = os.path.expanduser(os.path.expandvars(handle))
        return super(VFileBucket, self).__getitem__(handle)

    def __contains__(self, handle):
        if isinstance(handle, six.string_types):
            handle = os.path.expanduser(os.path.expandvars(handle))
        return super(VFileBucket, self).__contains__(handle)
