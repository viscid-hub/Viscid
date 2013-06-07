#!/usr/bin/env python
# i guess this is the 'new' one???

from __future__ import print_function

from .bucket import Bucket
from . import verror
from .vutil import spill_prefix

class Grid(object):
    """ Grids contain fields... Datasets recurse to grids using __getitem__
    and get_field in order to find fields """
    topology_info = None
    geometry_info = None
    crds = None
    fields = None
    # so... not all grids have times? if we try to access time on a grid
    # that doesnt have one, there is probably a bug somewhere
    # time = None

    name = None

    def __init__(self, name=None):
        self.name = name
        self.fields = Bucket()

        self.coords = None  # Coordinates()

    def add_field(self, field):
        if isinstance(field, (list, tuple)):
            for f in field:
                self.fields[f.name] = f
        else:
            self.fields[field.name] = field

    def unload(self):
        """ unload is meant to give children a chance to free caches, the idea
        being that an unload will free memory, but all the functionality is
        preserved, so data is accessable without an explicit reload """
        for fld in self.fields:
            fld.unload()
        # TODO: does anything else need to be unloaded in here?

    def n_times(self, *args, **kwargs): #pylint: disable=R0201
        return 1

    def iter_times(self, *args, **kwargs):
        # FIXME: it is unclear to me what to do here, since a dataset
        # may have > 1 grid... and if so only the first will be returned...
        # i guess there is some ambiguity if there is no temporal dataset...
        return [self]

    def spill(self, recursive=False, prefix=""):
        self.fields.spill(prefix=prefix + spill_prefix)

    # def add_fields(self, name_field_list):
    #     """ input: [(name, data), (name1, name2, data), ...] """
    #     for nd in name_field_list:
    #         self.add_field(nd[:-1], nd[-1])

    # def set_time(self, time):
    #     self.time = time

    def set_crds(self, crds_object):
        self.crds = crds_object

    def get_crd(self, handle):
        return self.crds[handle]

    def get_field(self, fldname, time=None): #pylint: disable=W0613
        try:
            return self.fields[fldname]
        except KeyError:
            raise verror.FieldNotFound("{0} not found".format(fldname))

    def __contains__(self, item):
        return item in self.fields

    def __getitem__(self, item):
        if item in self.fields:
            return self.get_field(item)
        elif self.crds is not None and item in self.crds:
            return self.get_crd(item)
        else:
            raise verror.FieldNotFound("{0} not found".format(item))

    def __str__(self):
        return "<Grid name={0}>".format(self.name)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.unload()
        return None

##
## EOF
##
