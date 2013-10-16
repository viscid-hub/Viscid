#!/usr/bin/env python

from __future__ import print_function
import os
import logging
# import sys

from ..bucket import Bucket
from .vfile import VFile


class VFileBucket(Bucket):
    """ manages open files, create / get with get_file_bucket() as you
        generally only need one instance, but you can construct directly
        if you need more than one manager """

    def __init__(self, **kwargs):
        super(VFileBucket, self).__init__(**kwargs)

    # This routine is just sort of confusing
    # def add(self, fname, file):
    #     absfname = os.path.abspath(fname)
    #     self[(absfname, fname)] = f

    def load_file(self, fname, index_handle=True, **kwargs):
        """ load a single file and return a vFile instance, not a list
        of vFiles like load does """
        return self.load_files([fname], index_handle=index_handle, **kwargs)[0]

    def load_files(self, fnames, index_handle=True, **kwargs):
        """ initialize obj before it's put into the list, whatever is returned
            is what gets stored, returning None means object init failed, do
            not add to the _objs list

            kwargs is passed to file constructor """
        if not isinstance(fnames, (list, tuple)):
            fnames = [fnames]
        file_lst = []

        for fname in fnames:
            f = None
            absfname = os.path.abspath(fname)

            # if the file was already loaded, return it
            if absfname in self:
                # self._handles[absfname]  # "this statement has no effect"
                f = self[absfname]
            else:
                # load a new file
                ftype = VFile.detect_type(absfname)
                if not ftype:
                    raise RuntimeError("Can't determine type "
                                       "for {0}".format(absfname))
                try:
                    # vfilebucket kwarg is ignored by file types that don't care
                    f = ftype(absfname, vfilebucket=self, **kwargs)
                except IOError as e:
                    s = " IOError on file: {0}\n".format(absfname)
                    s += "              File Type: {0}\n".format(ftype)
                    s += "              {0}".format(e.message)
                    logging.warn(s)
                except ValueError:
                    pass
                #     raise IOError(s)

            if f is not None:
                self.set_item([absfname, fname], f, index_handle=index_handle)
            file_lst.append(f)

        return file_lst
