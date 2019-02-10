# This module should only be imported by modules in 'readers/'. If you just
# want to load files, use vfile_factory. This prevents circular imports.
#
# reader_base: provides file readers. This will eventually house the
# backend for data input.

from __future__ import print_function
# import sys
from operator import attrgetter
import os
import re
from time import time

from viscid import logger
from viscid.dataset import Dataset, DatasetTemporal
from viscid import grid
from viscid import field
from viscid.compat import string_types


def serialize_subclasses(root, _lst=None):
    if _lst is None:
        _lst = list()

    for kls in reversed(root.__subclasses__()):
        serialize_subclasses(kls, _lst=_lst)
    _lst += [root]

    return _lst


class DataWrapper(object):
    _hypersliceable = False  # can read slices from disk

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
    """Generic File

    Note:
        If you want a file that can load other files (like how XDMF
        files need to be able to load HDF5 files) then subclass off of
        :py:class:`viscid.readers.vfile_bucket.ContainerFile` instead.

    Note:
        Important when subclassing: Do not call the constructors for a
        dataset / grid yourself, dispatch through _make_dataset and
        _make_grid.
    """
    # _detector is a regex string used for file type detection
    _detector = None
    _priority = 0
    # _gc_warn = True  # i dont think this is used... it should go away?
    _grid_type = grid.Grid
    _dataset_type = Dataset
    _temporal_dataset_type = DatasetTemporal
    _grid_opts = {}

    SAVE_ONLY = False

    parent_bucket = None
    load_time = None
    handle_name = None  # set in VFileBucket.load_files

    fname = None
    dirname = None

    # this is for files that stay open after being parsed,
    # for instance hdf5 File object
    file = None

    # grids = None  # already part of Dataset

    def __init__(self, fname, parent_bucket=None, grid_type=None, grid_opts=None,
                 **kwargs):
        """  """
        super(VFile, self).__init__(name=fname, **kwargs)

        if grid_type is not None:
            self._grid_type = grid_type
        if grid_opts is not None:
            self.grid_opts = grid_opts
        assert isinstance(self._grid_opts, dict)

        self.parent_bucket = parent_bucket

        self.load(fname)

    def load(self, fname):
        # self.unload()
        fname = os.path.expanduser(os.path.expandvars(fname))
        self.fname = os.path.abspath(fname)
        self.dirname = os.path.dirname(self.fname)
        self.set_info("_viscid_dirname", self.dirname)
        self.load_time = time()
        self._parse()

    def reload(self):
        self._clear_cache()
        self.remove_all_items()
        self.load(self.fname)

    def unload(self, **kwargs):
        """Really unload a file, don't just clear the cache"""
        self._clear_cache()
        self.remove_all_items()
        if self.parent_bucket:
            self.parent_bucket.remove_reference(self, **kwargs)

    def __exit__(self, exc_type, value, traceback):
        self.unload()
        return None

    # some classy saving utility methods, should be sufficient to override
    # save and save_fields
    def save(self, fname=None, **kwargs):
        """ save an instance of VFile, fname defaults to the name
        of the file object as read """
        raise NotImplementedError()

    @classmethod
    def save_grid(cls, fname, grd, **kwargs):
        cls.save_fields(fname, grd.field_dict(), **kwargs)

    @classmethod
    def save_field(cls, fname, fld, **kwargs):
        cls.save_fields(fname, {kwargs.pop('name', fld.name): fld}, **kwargs)

    @classmethod
    def save_fields(cls, fname, flds, **kwargs):
        """ save some fields using the format given by the class """
        raise NotImplementedError()

    def _make_dataset(self, parent_node, dset_type="dataset", name=None,
                      **kwargs):
        """Use this instead of calling Dataset(...) yourself

        Args:
            parent_node (Dataset, Grid, or None): Hint at parent in
                the tree, needed if info is used before this object
                is added to its parent
            grid_type (str, subclass of Dataset, optional): type of
                dataset to create
        """
        dset_type = dset_type.lower()
        if isinstance(dset_type, string_types):
            if dset_type == "dataset":
                dset_type = self._dataset_type
            elif dset_type == "temporal":
                dset_type = self._temporal_dataset_type
            else:
                raise ValueError("unknown dataset type: {0}".format(dset_type))
        dset = dset_type(name=name, **kwargs)
        if parent_node is not None:
            parent_node.prepare_child(dset)
        return dset

    def _make_grid(self, parent_node, grid_type=None, name=None, **kwargs):
        """Use this instead of calling Grid(...) yourself

        Args:
            parent_node (Dataset, Grid, or None): Hint at parent in
                the tree, needed if info is used before this object
                is added to its parent
            grid_type (subclass of Grid, optional): if not given, use
                self._grid_type
            name (str, optional): self explanatory
        """
        other = dict(self._grid_opts)
        other.update(kwargs)

        if grid_type is None:
            grid_type = self._grid_type
        if grid_type is None:
            raise TypeError("{0} can't create grids".format(type(self)))

        g = grid_type(name=name, **other)
        if parent_node is not None:
            parent_node.prepare_child(g)
        return g

    def _make_field(self, parent_node, fldtype, name, crds, data, **kwargs):
        """Use this instead of calling Grid(...) yourself

        Args:
            parent_node (Dataset, Grid, or None): Hint at parent in
                the tree, needed if info is used before this object
                is added to its parent
        """
        fld = field.wrap_field(data, crds, name=name, fldtype=fldtype, **kwargs)
        if parent_node is not None:
            parent_node.prepare_child(fld)
        return fld

    def _parse(self):
        # make _parse 'abstract'
        raise NotImplementedError("override _parse to read a file")

    @classmethod
    def _detector_func(cls, fname):
        return True

    @classmethod
    def detect_type(cls, fname, mode='r', prefer=None):
        """recursively detect a filetype using _detector regex string.

        This is called recursively for all subclasses and results
        further down the tree are given precedence.

        TODO: move this functionality into a more robust/extendable factory
            class... that can also take care of the bucket / circular
            reference problem maybe

        Args:
            fname (str): Filename
            mode (str): 'r' or 'w'
            prefer (str): If multiple file types match, give some
                part of the class name for the reader that you prefer

        Note: THIS WILL ONLY WORK FOR CLASSES THAT HAVE ALREADY BEEN
            IMPORTED. THIS IS A FRAGILE MECHANISM IN THAT SENSE.

        Returns:
            VFile subclass: Some reader that matches fname
        """
        matched_classes = []
        for kls in serialize_subclasses(cls):
            if (kls._detector
                and re.match(kls._detector, fname)
                and kls._detector_func(fname)
                ):
                matched_classes.append(kls)

        # sort by reader priority
        matched_classes.sort(key=attrgetter('_priority'), reverse=True)

        ret = None
        if matched_classes:
            ret = matched_classes[0]
            if prefer:
                for kls in reversed(matched_classes):
                    if prefer.lower() in kls.__name__.lower():
                        ret = kls

        return ret

    @classmethod
    def resolve_type(cls, ftype):
        ftype = ftype.replace(' ', '').replace('_', '').replace('-', '').lower()
        _idx = ftype.find('file')
        if _idx >= 0:
            ftype = ftype[:_idx] + ftype[_idx + len('file'):]

        for filetype in reversed(cls.__subclasses__()):  # pylint: disable=E1101
            td = filetype.resolve_type(ftype)
            if td:
                return td

        cls_name = cls.__name__.lower()
        _idx = cls_name.find('file')
        if _idx >= 0:
            cls_name = cls_name[:_idx] + cls_name[_idx + len('file'):]

        if ftype in cls_name:
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
