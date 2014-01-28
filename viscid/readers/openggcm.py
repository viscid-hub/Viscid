#!/usr/bin/env python
""" Wrapper grid for some OpenGGCM convenience """

from __future__ import print_function

import numpy as np
try:
    import numexpr
    _has_numexpr = True
except ImportError:
    _has_numexpr = False

from . import xdmf
from .. import grid
from .. import field


class GGCMGrid(grid.Grid):
    """ This defines some cool openggcm convinience stuff...
    The following attributes can be set by saying,
        `viscid.grid.readers.openggcm.GGCMGrid.flag = value`.
    This should be done before a call to readers.load_file so that all grids
    that are instantiated have the flags you want.

    mhd_to_gse_on_read = True|False, flips arrays on load to be in GSE crds
                         (default=True)
    copy_on_transform = True|False, True means array will be contiguous
                        after transform (if one is done), but makes data
                        load 50%-60% slower (default=True)
    derived_vector_layout = viscid.field.LAYOUT_*, force layout when
                            preparing a derived vector, like B from a file with
                            bx, by, bz scalar arrays
                            (default=LAYOUT_DEFAULT)

    derived quantities accessable by dictionary lookup:
      - T (Temperature, for now, just pressure / density)
      - bx, by, bz (CC mag field components, if not already stored by
                    component)
      - b (mag field as vector, layout affected by
           GGCMGrid.derived_vector_layout)
      - v (velocity as vector, same idea as b)
    """
    mhd_to_gse_on_read = True
    copy_on_transform = True
    derived_vector_layout = field.LAYOUT_DEFAULT

    def transform_mhd_to_gse_field(self, arr):
        # Note: field._data will be set to whatever is returned (after
        # being reshaped to the crd shape), so if you return a view,
        # field._data will be a view
        return np.array(arr[:, ::-1, ::-1], copy=self.copy_on_transform)

    def transform_mhd_to_gse_crds(self, arr):
        return np.array(-1.0 * arr[::-1], copy=self.copy_on_transform)

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
                f.post_reshape_transform_func = self.transform_mhd_to_gse_field
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

    def _get_b(self):
        bx, by, bz = self['bx'], self['by'], self['bz']
        b = field.scalar_fields_to_vector("B", [bx, by, bz],
            deep_meta={"force_layout": self.derived_vector_layout})
        return b

    def _get_v(self):
        vx, vy, vz = self['vx'], self['vy'], self['vz']
        v = field.scalar_fields_to_vector("V", [vx, vy, vz],
            deep_meta={"force_layout": self.derived_vector_layout})
        return v

    def _calc_mag(self, vx, vy, vz):
        if _has_numexpr:
            vmag = numexpr.evaluate("sqrt(vx**2 + vy**2 + vz**2)")
            return vx.wrap(vmag, typ="Scalar")
        else:
            vmag = np.sqrt(vx**2 + vy**2 + vz**2)
            return vmag

    def _get_bmag(self):
        bx, by, bz = self['bx'], self['by'], self['bz']
        bmag = self._calc_mag(bx, by, bz)
        bmag.name = "|B|"
        return bmag

    def _get_jmag(self):
        jx, jy, jz = self['jx'], self['jy'], self['jz']
        jmag = self._calc_mag(jx, jy, jz)
        jmag.name = "|J|"
        return jmag

    def _get_speed(self):
        vx, vy, vz = self['vx'], self['vy'], self['vz']
        speed = self._calc_mag(vx, vy, vz)
        speed.name = "Speed"
        return speed


class GGCMFile(xdmf.FileXDMF):  # pylint: disable=W0223
    _detector = r"^\s*.*\.(p[xyz]_[0-9]+|3d|3df)" \
                r"(\.[0-9]{6})?\.(xmf|xdmf)\s*$"
    _grid_type = GGCMGrid

##
## EOF
##
