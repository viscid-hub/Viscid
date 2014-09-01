# This module should only be imported by modules in 'readers/'. If you just
# want to load files, use vfile_factory. This prevents circular imports.
#
# reader_base: provides file readers. This will eventually house the
# backend for data input.

from __future__ import print_function
# import sys
import os
import re
from time import time

from viscid.dataset import Dataset
from viscid import grid

class DataWrapper(object):
    _shape = None
    _dtype = None

    def __init__(self):
        self._shape = None
        self._dtype = None

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    def __array__(self, *args, **kwargs):
        raise NotImplementedError()

    def read_direct(self, *args, **kwargs):
        raise NotImplementedError()

    def len(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        raise NotImplementedError()


class VFile(Dataset):
    # _detector is a regex string used for file type detection
    _detector = None
    # _gc_warn = True  # i dont think this is used... it should go away?
    _grid_type = grid.Grid
    _grid_opts = {}

    vfilebucket = None
    load_time = None

    fname = None
    dirname = None

    file = None  # this is for files that stay open after being parsed,
                 # for instance hdf5 File object
    # grids = None  # already part of Dataset

    def __init__(self, fname, vfilebucket=None, grid_type=None, grid_opts=None,
                 **kwargs):
        """  """
        super(VFile, self).__init__(fname, **kwargs)

        if grid_type is not None:
            self._grid_type = grid_type
        if grid_opts is not None:
            self.grid_opts = grid_opts
        assert self._grid_type is not None
        assert isinstance(self._grid_opts, dict)

        self.vfilebucket = vfilebucket
        self.load(fname)

    def load(self, fname):
        #self.unload()
        fname = os.path.expanduser(os.path.expandvars(fname))
        self.fname = os.path.abspath(fname)
        self.dirname = os.path.dirname(self.fname)
        self.load_time = time()
        self._parse()

    def refresh(self):
        #self.unload()
        self.load(self.fname)

    def unload(self):
        """ unload is meant to give children a chance to free caches, the idea
        being that an unload will free memory, but all the functionality is
        preserved, so data is accessable without an explicit reload
        """
        super(VFile, self).unload()

    # some classy saving utility methods, should be sufficient to override
    # save and save_fields
    def save(self, fname=None, **kwargs):
        """ save an instance of VFile, fname defaults to the name
        of the file object as read """
        raise NotImplementedError()

    @classmethod
    def save_grid(cls, fname, grd, **kwargs):
        flds = list(grd.iter_fields())
        cls.save_fields(fname, flds, **kwargs)

    @classmethod
    def save_field(cls, fname, fld, **kwargs):
        cls.save_fields(fname, [fld], **kwargs)

    @classmethod
    def save_fields(cls, fname, flds, **kwargs):
        """ save some fields using the format given by the class """
        raise NotImplementedError()

    # already implemented in dataset
    #def add_grid(self, grid):
    #    self.grids[grid.name] = grid

    # if i need this, it should be implemented at the Dataset level
    #def get_grid(self, grid_handle):
    #    return self.grids[grid_handle]

    # this is get_field implemented in Dataset
    # def get_data(self, item):
    #     if self.active_grid and item in self.activate_grid:
    #         # ask the active grid for the item
    #         return self.active_grid[item]
    #     else:
    #         return self.grids[item]

    def _parse(self):
        # make _parse 'abstract'
        raise NotImplementedError("override _parse to read a file")

    @classmethod
    def detect_type(cls, fname):
        """ recursively detect a filetype using _detector regex string.
        this is called recursively for all subclasses and results
        further down the tree are given precedence.
        NOTE: THIS WILL ONLY WORK FOR CLASSES THAT HAVE ALREADY BEEN IMPORTED.
        THIS IS A FRAGILE MECHANISM IN THAT SENSE.
        TODO: move this functionality into a more robust/extendable factory
        class... that can also take care of the bucket / circular reference
        problem maybe
        """
        # reversed gives precedence to the more recently declared classes
        for filetype in reversed(cls.__subclasses__()): #pylint: disable=E1101
            td = filetype.detect_type(fname)
            if td:
                return td
        if cls._detector and re.match(cls._detector, fname):
            return cls
        return None

    @classmethod
    def group_fnames(cls, fnames):
        """Group File names

        The default implementation just returns fnames, but some file
        types might do something fancy here

        Parameters:
            fnames (list): names that can be logically grouped, as in
                a bunch of file names that are different time steps
                of a given run

        Returns:
            A list of things that can be given to the constructor of
            this class
        """
        return fnames

    @classmethod
    def collective_name_from_group(cls, group):
        raise NotImplementedError()

    @classmethod
    def collective_name(cls, group):
        """
        Parameters:
            group: single file name or list of file names that would
                be grouped by group_fnames

        Returns:
            str: a single name
        """
        if not isinstance(group, (list, tuple)):
            group = [group]

        if len(group) > 1:
            return cls.collective_name_from_group(group)
        else:
            return group[0]

##
## EOF
##
