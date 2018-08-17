#!/usr/bin/env python
""" simple reader that tries to understand a numpy binary npz file
WARNING: not lazy
Current working assumption is that all fields in an npz file share the same
grid """

# import string
from __future__ import print_function
import os

import numpy as np

from viscid.compat import OrderedDict
from viscid import logger
from viscid.readers import vfile
from viscid import coordinate

class NPZDataWrapper(vfile.DataWrapper):
    """  """
    fname = None
    loc = None

    _shape = None
    _dtype = None

    def __init__(self, fname, loc):
        super(NPZDataWrapper, self).__init__()
        self.fname = fname
        self.loc = loc

    def _read_info(self):
        # this takes super long when reading 3 hrs worth of ggcm data
        # over sshfs
        # import pdb; pdb.set_trace()
        try:
            with np.load(self.fname) as f:
                dset = f[self.loc]
                self._shape = dset.shape
                self._dtype = dset.dtype
        except IOError:
            logger.error("Problem opening npz file, '%s'", self.fname)
            raise

    @property
    def shape(self):
        """ only ask for this if you really need it; can be a speed problem
        for large temporal datasets over sshfs """
        if self._shape is None:
            self._read_info()
        return self._shape

    @property
    def dtype(self):
        """ only ask for this if you really need it; can be a speed problem
        for large temporal datasets over sshfs """
        if self._dtype is None:
            self._read_info()
        return self._dtype

    def wrap_func(self, func_name, *args, **kwargs):
        with np.load(self.fname) as f:
            return getattr(f[self.loc], func_name)(*args, **kwargs)

    def __array__(self, *args, **kwargs):
        return self.wrap_func("__array__", *args, **kwargs)

    def read_direct(self, *args, **kwargs):
        raise NotImplementedError()

    def len(self):
        return self.wrap_func("len")

    def __getitem__(self, item):
        return self.wrap_func("__getitem__", item)


class FileNumpyNPZ(vfile.VFile):
    """ open an ascii file with viscid format, or if not specified, assume
    it's in gnuplot format, gnuplot format not yet specified """
    _detector = r".*\.(npz)\s*$"

    _KEY_CRDS = "crd_names"
    _KEY_FLDS = {"node": "field_names_nc",
                 "cell": "field_names_cc",
                 "face": "field_names_fc",
                 "edge": "field_names_ec"}

    def __init__(self, fname, **kwargs):
        super(FileNumpyNPZ, self).__init__(fname, **kwargs)

    def _wrap_lazy_field(self, parent_node, file_name, fld_name, crds, center):
        lazy_arr = NPZDataWrapper(file_name, fld_name)
        if len(lazy_arr.shape) == crds.nr_dims:
            fldtype = "Scalar"
        elif len(lazy_arr.shape) == crds.nr_dims + 1:
            fldtype = "Vector"
        else:
            raise IOError("can't infer field type")
        return self._make_field(parent_node, fldtype, fld_name, crds, lazy_arr,
                                center=center)

    def _parse(self):
        g = self._make_grid(self, **self._grid_opts)

        with np.load(self.fname) as f:
            fld_names = list(f.keys())

            crd_names = []
            # try to get crds names from an array of strings called _KEY_CRDS
            # else, assume it's x, y, z and see if that works
            try:
                clist = [(ax, f[ax]) for ax in f[self._KEY_CRDS]]
                crd_names = f[self._KEY_CRDS]
                fld_names.remove(self._KEY_CRDS)
            except KeyError:
                for axisname in "xyz":
                    if axisname in f:
                        crd_names.append(axisname)
            clist = [(cn, NPZDataWrapper(self.fname, cn)) for cn in crd_names]
            crds = coordinate.wrap_crds("nonuniform_cartesian", clist)
            g.set_crds(crds)
            for c in clist:
                # we should be sure by now that the keys exist
                fld_names.remove(c[0])

            # try to get field names from arrays of nc, cc, ec, fc
            # fields
            for fld_center, names_key in self._KEY_FLDS.items():
                try:
                    names = f[names_key]
                    fld_names.remove(names_key)
                except KeyError:
                    names = []

                for name in names:
                    fld = self._wrap_lazy_field(g, self.fname, name, crds,
                                                fld_center)
                    g.add_field(fld)
                    fld_names.remove(name)

            # load any remaining fields as though they were node centered
            for name in fld_names:
                fld = self._wrap_lazy_field(g, self.fname, name, crds, "Node")
                g.add_field(fld)

        self.add(g)
        self.activate(0)

    def save(self, fname=None, **kwargs):
        if fname is None:
            fname = self.fname
        self.save_fields(fname, self.field_dict())

    @classmethod
    def save_fields(cls, fname, flds, **kwargs):
        assert len(flds) > 0
        fname = os.path.expanduser(os.path.expandvars(fname))

        if isinstance(flds, list):
            if isinstance(flds[0], (list, tuple)):
                flds = OrderedDict(flds)
            else:
                flds = OrderedDict([(fld.name, fld) for fld in flds])

        fld_dict = {}

        # setup crds
        # FIXME: all coordinates are saved as non-uniform, the proper
        #        way to do this is to have let coordinate format its own
        #        hdf5 / xdmf / numpy binary output
        fld0 = next(iter(flds.values()))
        clist = fld0.crds.get_clist(full_arrays=True)
        axis_names = []
        for axis_name, crdarr in clist:
            fld_dict[axis_name] = crdarr
            axis_names.append(axis_name)
        fld_dict[cls._KEY_CRDS] = np.array(axis_names)

        # setup fields
        # dict comprehension invalid in Python 2.6
        # fld_names = {key.lower(): [] for key in cls._KEY_FLDS.keys()}
        fld_names = {}
        for key in cls._KEY_FLDS.keys():
            fld_names[key.lower()] = []

        for name, fld in flds.items():
            fld_names[fld.center.lower()].append(name)
            fld_dict[name] = fld.data

        for center, names_lst in fld_names.items():
            fld_dict[cls._KEY_FLDS[center.lower()]] = np.array(names_lst)

        if fname.endswith(".npz"):
            fname = fname[:-4]
        np.savez(fname, **fld_dict)

##
## EOF
##
