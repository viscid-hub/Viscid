#!/usr/bin/env python
# i guess this is the 'new' one???

from __future__ import print_function

from .bucket import Bucket
from .vutil import tree_prefix

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

        self.crds = None  # Coordinates()

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

    def nr_times(self, *args, **kwargs): #pylint: disable=W0613,R0201
        return 1

    def iter_times(self, *args, **kwargs): #pylint: disable=W0613
        # FIXME: it is unclear to me what to do here, since a dataset
        # may have > 1 grid... and if so only the first will be returned...
        # i guess there is some ambiguity if there is no temporal dataset...
        with self as me:
            yield me

    def iter_fields(self, named=None, **kwargs): #pylint: disable=W0613
        """ iterate over fields in a grid, if named is given, it should be a
        list of field names to iterate over """
        if named is not None:
            for name in named:
                with self.fields[name] as f:
                    yield f
        else:
            for fld in self.fields:
                with fld as f:
                    yield f

    def print_tree(self, recursive=False, prefix=""): #pylint: disable=W0613
        self.fields.print_tree(prefix=prefix + tree_prefix)

    # def set_time(self, time):
    #     self.time = time

    def set_crds(self, crds_object):
        self.crds = crds_object

    ##################################
    ## Utility methods to get at crds
    # these are the same as something like self.crds['xnc']
    # or self.crds.get_crd()
    def get_crd_nc(self, axis, shaped=False):
        """ returns a flat ndarray of coordinates along a given axis
        axis can be crd name as string, or index, as in x==2, y==1, z==2 """
        return self.crds.get_nc(axis, shaped=shaped)

    def get_crd_cc(self, axis, shaped=False):
        """ returns a flat ndarray of coordinates along a given axis
        axis can be crd name as string, or index, as in x==2, y==1, z==2 """
        return self.crds.get_cc(axis, shaped=shaped)

    def get_crd_ec(self, axis, shaped=False):
        """ returns a flat ndarray of coordinates along a given axis
        axis can be crd name as string, or index, as in x==2, y==1, z==2 """
        return self.crds.get_ec(axis, shaped=shaped)

    def get_crd_fc(self, axis, shaped=False):
        """ returns a flat ndarray of coordinates along a given axis
        axis can be crd name as string, or index, as in x==2, y==1, z==2 """
        return self.crds.get_fc(axis, shaped=shaped)

    ## these return all crd dimensions
    # these are the same as something like self.crds.get_crds()
    def get_crds_nc(self, axes=None, shaped=False):
        """ returns all node centered coords as a list of ndarrays, flat if
        shaped==False, or shaped if shaped==True """
        return self.crds.get_crds_nc(axes=axes, shaped=shaped)

    def get_crds_cc(self, axes=None, shaped=False):
        """ returns all cell centered coords as a list of ndarrays, flat if
        shaped==False, or shaped if shaped==True """
        return self.crds.get_crds_cc(axes=axes, shaped=shaped)

    def get_crds_fc(self, axes=None, shaped=False):
        """ returns all face centered coords as a list of ndarrays, flat if
        shaped==False, or shaped if shaped==True """
        return self.crds.get_crds_fc(axes=axes, shaped=shaped)

    def get_crds_ec(self, axes=None, shaped=False):
        """ returns all edge centered coords as a list of ndarrays, flat if
        shaped==False, or shaped if shaped==True """
        return self.crds.get_crds_ec(axes=axes, shaped=shaped)

    ##
    def get_field(self, fldname, time=None): #pylint: disable=W0613
        return self.fields[fldname]

    def __contains__(self, item):
        return item in self.fields

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, item):
        """ returns a field by name, or if no field is found, a coordinate by
        name some crd identifier, see Coordinate.get_item for details """
        if item in self.fields:
            return self.get_field(item)
        elif self.crds is not None and item in self.crds:
            return self.crds[item]
        else:
            raise KeyError(item)

    def __setitem__(self, fldname, fld):
        self.fields[fldname] = fld

    def __delitem__(self, fldname):
        self.fields.__delitem__(fldname)

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
