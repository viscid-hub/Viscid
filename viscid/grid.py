#!/usr/bin/env python
# i guess this is the 'new' one???

from __future__ import print_function

from viscid import field
from viscid.bucket import Bucket
from viscid.vutil import tree_prefix

class Grid(object):
    """ Grids contain fields... Datasets recurse to grids using __getitem__
    and get_field in order to find fields

    The following attributes are for customizing data reads, intended to
    be changed by `grid.Grid.flag = value`:
    force_vector_layout = field.LAYOUT_*, force all vectors to be of a certain
                          layout when they're created (default: LAYOUT_DEFAULT)
    """
    topology_info = None
    geometry_info = None
    crds = None
    fields = None
    # so... not all grids have times? if we try to access time on a grid
    # that doesnt have one, there is probably a bug somewhere
    # time = None

    name = None

    # these attributes are intended to control how fields are read
    # they can be customized by setting the class value, and inherited
    # by grids that get created in the future
    force_vector_layout = field.LAYOUT_DEFAULT

    def __init__(self, name=None, **kwargs):
        """ all kwargs are added to the grid as attributes """
        self.name = name
        self.fields = Bucket()

        self.crds = None  # Coordinates()
        for opt, value in kwargs.items():
            setattr(self, opt, value)

    @staticmethod
    def null_transform(something):
        return something

    # def set_time(self, time):
    #     self.time = time

    def set_crds(self, crds_object):
        self.crds = crds_object

    def add_field(self, *fields):
        """ Note: in XDMF reader, the grid will NOT have crds when adding
        fields, so any grid crd transforms won't be set """
        for f in fields:
            if isinstance(f, field.VectorField):
                f.layout = self.force_vector_layout
            self.fields[f.name] = f

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
        try:
            return self.fields[fldname]
        except KeyError:
            func = "_get_" + fldname
            if hasattr(self, func):
                return getattr(self, func)()
            else:
                raise KeyError("field not found")

    def get_grid(self, time=None): #pylint: disable=W0613
        return self

    def __contains__(self, item):
        return item in self.fields

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, item):
        """ returns a field by name, or if no field is found, a coordinate by
        name some crd identifier, see Coordinate.get_item for details """
        try:
            return self.get_field(item)
        except KeyError:
            if self.crds is not None and item in self.crds:
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

    def __exit__(self, typ, value, traceback):
        self.unload()
        return None

##
## EOF
##
