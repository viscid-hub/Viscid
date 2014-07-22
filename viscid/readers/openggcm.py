#!/usr/bin/env python
""" Wrapper grid for some OpenGGCM convenience """

from __future__ import print_function
import os
import re
import logging

import numpy as np
try:
    import numexpr
    _has_numexpr = True
except ImportError:
    _has_numexpr = False

from viscid.readers.vfile_bucket import VFileBucket
from viscid.readers.ggcm_logfile import GGCMLogFile
from viscid.readers import xdmf
from viscid import grid
from viscid import field
from viscid.calculator import plasma


def find_file_uptree(directory, basename, max_depth=8, _depth=0):
    """Find basename by going up the file tree

    Keep going up a directory until you find one that has the file
    "basename"

    Parameters:
        directory (str): directory to start the search
        basename (str): bare file name
        max_depth (int): max number of directories to seach

    Returns:
        Relative path to file, or None if not found
    """
    max_depth = 5

    if not os.path.isdir(directory):
        raise RuntimeError("this is non-sensicle")

    fname = os.path.join(directory, basename)
    # log_fname = "{0}/{1}.log".format(d, self.info["run"])
    if os.path.isfile(fname):
        return fname

    if _depth > max_depth or os.path.abspath(directory) == '/':
        return None

    return find_file_uptree(os.path.join(directory, ".."), basename,
                            _depth=(_depth + 1))


class GGCMGrid(grid.Grid):
    r""" This defines some cool openggcm convinience stuff...
    The following attributes can be set by saying

        ``viscid.grid.readers.openggcm.GGCMGrid.flag = value``.

    This should be done before a call to readers.load_file so that all
    grids that are instantiated have the flags you want.

    Derived quantities accessable by dictionary lookup
        - T: Temperature, for now, just pressure / density
        - bx, by, bz: CC mag field components, if not already stored by
          component)
        - b: mag field as vector, layout affected by
          GGCMGrid.derived_vector_layout
        - v: velocity as vector, same idea as b
        - beta: plasma beta, just pp/b^2
        - psi: flux function (only works for 2d files/grids)

    Attributes:
        mhd_to_gse_on_read (bool): flips arrays on load to be in
            GSE crds (default is False)
        copy_on_transform (bool): True means array will be contiguous
            after transform (if one is done), but makes data load
            50\%-60\% slower (default is True)
        force_vector_layout (str): inherited from grid.Grid, enforces
            layout for vector fields on load (default is
            :py:const:`viscid.field.LAYOUT_DEFAULT`)
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
                m1 = np.array([-1.0], dtype=arr.dtype)  # pylint: disable=unused-variable
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
        T.pretty_name = "T"
        return T

    def _get_bx(self):
        return self['b'].component_fields()[0]

    def _get_by(self):
        return self['b'].component_fields()[1]

    def _get_bz(self):
        return self['b'].component_fields()[2]

    def _get_vx(self):
        return self['v'].component_fields()[0]

    def _get_vy(self):
        return self['v'].component_fields()[1]

    def _get_vz(self):
        return self['v'].component_fields()[2]

    def _assemble_vector(self, base_name, comp_names="xyz", forget_source=True,
                         **kwargs):

        opts = dict(forget_source=forget_source, **kwargs)

        if len(comp_names) == 3:
            with self[base_name + comp_names[0]] as vx, \
                 self[base_name + comp_names[1]] as vy, \
                 self[base_name + comp_names[2]] as vz:
                v = field.scalar_fields_to_vector(base_name, [vx, vy, vz],
                                                  **opts)
        else:
            comps = [self[base_name + c] for c in comp_names]
            v = field.scalar_fields_to_vector(base_name, comps, **opts)
            for comp in comps:
                comp.unload()
        return v

    def _get_b(self):
        return self._assemble_vector("b", _force_layout=self.force_vector_layout,
                                     pretty_name="B")

    def _get_v(self):
        return self._assemble_vector("v", _force_layout=self.force_vector_layout,
                                     pretty_name="V")

    def _get_e(self):
        return self._assemble_vector("e", _force_layout=self.force_vector_layout,
                                     pretty_name="E")


    def _get_e_cc(self):
        with self["ex_cc"] as ex, \
             self["ey_cc"] as ey, \
             self["ez_cc"] as ez:
            v = field.scalar_fields_to_vector("e", [ex, ey, ez],
                                              _force_layout=self.force_vector_layout,
                                              pretty_name="E")
        return v

    def _get_j(self):
        return self._assemble_vector("j", _force_layout=self.force_vector_layout,
                                     pretty_name="J")

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

    def _get_beta(self):
        return plasma.calc_beta(self['pp'], self['b'])

    def _get_psi(self):
        B = self['b']
        # try to guess if a dim of a 3D field is invariant
        if B.nr_sdims > 2:
            slcs = [slice(None)] * B.nr_sdims
            for i, nxi in enumerate(B.sshape):
                if nxi <= 2:
                    slcs[i] = 0
            B = B[slcs]
        return plasma.calc_psi(B)


class GGCMFile(xdmf.FileXDMF):  # pylint: disable=W0223
    """File type for GGCM style convenience stuff

    Attributes:
        read_log_file (bool): search for a log file to load some of the
            libmrc runtime parameters. This does not read parameters
            from all libmrc classes, but can be customized with
            :py:const`viscid.readers.ggcm_logfile.GGCMLogFile.
            watched_classes`. Defaults to False for performance.
    """
    _detector = r"^\s*(.*)\.(p[xyz]_[0-9]+|3d|3df)" \
                r"(\.[0-9]{6})?\.(xmf|xdmf)\s*$"
    _grid_type = GGCMGrid

    # this can be set to true if these parameters are needed
    read_log_file = False

    def load(self, fname):
        super(GGCMFile, self).load(fname)
        basename = os.path.basename(self.fname)
        self.info['run'] = re.match(self._detector, basename).group(1)

        # look for a log file to auto-load some parameters about the run
        if self.read_log_file:
            log_basename = "{0}.log".format(self.info['run'])
            # FYI, default max_depth should be 8
            log_fname = find_file_uptree(self.dirname, log_basename)
            if log_fname is None:
                log_fname = find_file_uptree(".", log_basename)
            if log_fname is None:
                log_fname = find_file_uptree(self.dirname, "log.txt")
            if log_fname is None:
                log_fname = find_file_uptree(".", "log.txt")

            if log_fname is not None:
                self.info["log_fname"] = log_fname
                if self.vfilebucket is None:
                    self.vfilebucket = VFileBucket()
                log_f = self.vfilebucket.load_file(log_fname,
                                                   file_type=GGCMLogFile,
                                                   index_handle=False)
                self.info.update(log_f.info)
            else:
                logging.warn("You wanted to read parameters from the logfile, but "
                             "I couldn't find one. Maybe you need to copy it from "
                             "somewhere?")

##
## EOF
##
