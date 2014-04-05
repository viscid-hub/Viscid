#!/usr/bin/env python
""" Wrapper grid for some OpenGGCM convenience """

from __future__ import print_function

import numpy as np
try:
    import numexpr
    _has_numexpr = True
except ImportError:
    _has_numexpr = False

from viscid.readers import xdmf
from viscid import grid
from viscid import field


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
    force_vector_layout = inherited from grid.Grid, enforces layout for
                          vector fields on load (default=LAYOUT_DEFAULT)

    derived quantities accessable by dictionary lookup:
      - T (Temperature, for now, just pressure / density)
      - bx, by, bz (CC mag field components, if not already stored by
                    component)
      - b (mag field as vector, layout affected by
           GGCMGrid.derived_vector_layout)
      - v (velocity as vector, same idea as b)
    """
    _flip_vect_comp_names = "bx, by, b1x, b1y, " \
                            "vx, vy, rv1x, rv1y, " \
                            "jx, jy, ex, ey, ex_cc, ey_cc".split(', ')
    _flip_vect_names = "v, b, j, xj".split(', ')
    # _flip_vect_comp_names = []
    # _flip_vect_names = []

    mhd_to_gse_on_read = False
    copy_on_transform = False

    def mhd2gse_field_scalar(self, fld, arr):  # pylint: disable=W0613
        # Note: field._data will be set to whatever is returned (after
        # being reshaped to the crd shape), so if you return a view,
        # field._data will be a view
        return np.array(arr[:, ::-1, ::-1], copy=self.copy_on_transform)

    def mhd2gse_field_scalar_m1(self, fld, arr):  # pylint: disable=W0613
        # This is always copied since the -1.0 * arr will need new
        # memory anyway
        a = np.array(arr[:, ::-1, ::-1], copy=False)

        if self.copy_on_transform:
            if _has_numexpr:
                m1 = np.array([-1.0], dtype=arr.dtype)
                a = numexpr.evaluate("a * m1")
            else:
                a = a * -1
        else:
            a *= -1.0
        return a

    def mhd2gse_field_vector(self, fld, arr):
        layout = fld.layout
        if layout == field.LAYOUT_INTERLACED:
            a = np.array(arr[:, ::-1, ::-1, :], copy=False)
            factor = np.array([-1.0, -1.0, 1.0],
                              dtype=arr.dtype).reshape(1, 1, 1, -1)
        elif layout == field.LAYOUT_FLAT:
            a = np.array(arr[:, :, ::-1, ::-1], copy=False)
            factor = np.array([-1.0, -1.0, 1.0],
                              dtype=arr.dtype).reshape(-1, 1, 1, 1)
        else:
            raise RuntimeError("well what am i looking at then...")

        if self.copy_on_transform:
            if _has_numexpr:
                a = numexpr.evaluate("arr * factor")
            else:
                a = a * factor
        else:
            a *= factor
        return a

    def mhd2gse_crds(self, crds, arr):  # pylint: disable=W0613
        return np.array(-1.0 * arr[::-1], copy=self.copy_on_transform)

    def set_crds(self, crds_object):
        if self.mhd_to_gse_on_read:
            transform_dict = {}
            transform_dict['y'] = self.mhd2gse_crds
            transform_dict['x'] = self.mhd2gse_crds
            crds_object.transform_funcs = transform_dict
        super(GGCMGrid, self).set_crds(crds_object)

    def add_field(self, *fields):
        for f in fields:
            if self.mhd_to_gse_on_read:
                # what a pain... vector components also need to be flipped
                if f.name in self._flip_vect_comp_names:
                    f.post_reshape_transform_func = self.mhd2gse_field_scalar_m1
                elif f.name in self._flip_vect_names:
                    f.post_reshape_transform_func = self.mhd2gse_field_vector
                else:
                    f.post_reshape_transform_func = self.mhd2gse_field_scalar
                f.info["crd_system"] = "gse"
            else:
                f.info["crd_system"] = "mhd"
        super(GGCMGrid, self).add_field(*fields)

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
        with self['bx'] as bx, self['by'] as by, self['bz'] as bz:
            b = field.scalar_fields_to_vector("B", [bx, by, bz],
                            _force_layout=self.force_vector_layout,
                            forget_source=True)
        return b

    def _get_v(self):
        with self['vx'] as vx, self['vy'] as vy, self['vz'] as vz:
            v = field.scalar_fields_to_vector("V", [vx, vy, vz],
                            _force_layout=self.force_vector_layout,
                            forget_source=True)
        return v


    @staticmethod
    def _calc_mag(vx, vy, vz):
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
