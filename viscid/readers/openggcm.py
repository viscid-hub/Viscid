#!/usr/bin/env python
""" Wrapper grid for some OpenGGCM convenience """

from __future__ import print_function
import numpy as np

from .. import grid
from . import xdmf


class GGCMGrid(grid.Grid):
    # this puts a 60% performance hit when loading data from h5 file
    # to an ndarray in memory. Most of that seems to be creating a temp
    # ndarray for the translation since h5py won't read the array
    # backward, which is kind of a good thing cause it wouldn't be straight
    # forward making a streamlined translation interface if it could
    # (since fld._dat_to_ndarray handles lists too)
    mhd_to_gse_on_read = True

    @staticmethod
    def transform_mhd_to_gse_field(arr):
        # Note: field._data will be set to whatever is returned (after
        # being reshaped to the crd shape), so if you return a view,
        # field._data will be a view
        return np.array(arr[:, ::-1, ::-1])

    @staticmethod
    def transform_mhd_to_gse_crds(arr):
        return np.array(-1.0 * arr[::-1])

    def set_crds(self, crds_object):
        super(GGCMGrid, self).set_crds(crds_object)
        if self.mhd_to_gse_on_read:
            transform_dict = {}
            transform_dict['y'] = self.transform_mhd_to_gse_crds
            transform_dict['x'] = self.transform_mhd_to_gse_crds
            self.crds.transform_funcs = transform_dict

    def add_field(self, fields):
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for f in fields:
            if self.mhd_to_gse_on_read:
                f.transform_func = self.transform_mhd_to_gse_field
                f.info["crd_system"] = "gse"
            else:
                f.info["crd_system"] = "mhd"
            self.fields[f.name] = f

    def _get_T(self):
        pp = self["pp"]
        rr = self["rr"]
        T = pp / rr
        T.name = "T"
        return T

    def _get_bx(self):
        return self['b'].component_fields()[0]

    def _get_by(self):
        return self['b'].component_fields()[1]

    def _get_bz(self):
        return self['b'].component_fields()[2]


class GGCMFile(xdmf.FileXDMF):  # pylint: disable=W0223
    _detector = r"^\s*.*\.(p[xyz]_[0-9]+|3d|3df)" \
                r"(\.[0-9]{6})?\.(xmf|xdmf)\s*$"
    _grid_type = GGCMGrid

##
## EOF
##
