""" athena reader common stuff """

from __future__ import print_function
import re
import os
from operator import itemgetter

import numpy as np
try:
    import numexpr
    _has_numexpr = True
except ImportError:
    _has_numexpr = False

from viscid import field
from viscid import grid
from viscid.calculator import plasma

def group_athena_files_common(detector, fnames):
    infolst = []
    for name in fnames:
        m = re.match(detector, name)
        grps = m.groups()
        d = dict(runname=grps[0], time=int(grps[1]), fname=m.string)
        infolst.append(d)

    # careful with sorting, only consecutive files will be grouped
    infolst.sort(key=itemgetter("time"))
    infolst.sort(key=itemgetter("runname"))

    info_groups = []
    info_group = [infolst[0]]
    for info in infolst[1:]:
        last = info_group[-1]
        if info['runname'] == last['runname']:
            info_group.append(info)
        else:
            info_groups.append(info_group)
            info_group = [info]
    info_groups.append(info_group)

    # turn info_groups into groups of just file names
    groups = []
    for info_group in info_groups:
        groups.append([info['fname'] for info in info_group])

    return groups

def athena_collective_name_from_group(detector, fnames):
    fname0 = fnames[0]
    basename = os.path.basename(fname0)
    m = re.match(detector, basename)
    run = m.group(1)
    ext = m.group(3)
    new_basename = "{0}.STAR.{1}".format(run, ext)
    return os.path.join(os.path.dirname(fname0), new_basename)


class AthenaGrid(grid.Grid):
    def _get_rr(self):
        return self['d']

    def _get_p(self):
        # TODO: calc pressure from conserved fields
        return self['P']

    def _get_pp(self):
        return self['p']

    def _get_bx(self):
        try:
            f = self['B1c']
        except KeyError:
            f = self['B1']
        f.name = "bx"
        f.pretty_name = "$B_x$"
        return f

    def _get_by(self):
        try:
            f = self['B2c']
        except KeyError:
            f = self['B2']
        f.name = "by"
        f.pretty_name = "$B_y$"
        return f

    def _get_bz(self):
        try:
            f = self['B3c']
        except KeyError:
            f = self['B3']
        f.name = "bz"
        f.pretty_name = "$B_z$"
        return f

    def _get_vx(self):
        try:
            f = self['V1']
        except KeyError:
            f = self['M1'] / self['d']
        f.name = "vx"
        f.pretty_name = "$V_x$"
        return f

    def _get_vy(self):
        try:
            f = self['V2']
        except KeyError:
            f = self['M2'] / self['d']
        f.name = "vy"
        f.pretty_name = "$V_y$"
        return f

    def _get_vz(self):
        try:
            f = self['V3']
        except KeyError:
            f = self['M3'] / self['d']
        f.name = "bz"
        f.pretty_name = "$V_z$"
        return f

    def _get_mx(self):
        try:
            f = self['M1']
        except KeyError:
            f = self['V1'] * self['d']
        f.name = "mx"
        f.pretty_name = "$M_x$"
        return f

    def _get_my(self):
        try:
            f = self['M2']
        except KeyError:
            f = self['V2'] * self['d']
        f.name = "my"
        f.pretty_name = "$M_y$"
        return f

    def _get_mz(self):
        try:
            f = self['M3']
        except KeyError:
            f = self['V3'] * self['d']
        f.name = "mz"
        f.pretty_name = "$M_z$"
        return f

    def _get_jx(self):
        f = self['j'].component_fields()[0]
        f.pretty_name = "$J_x$"
        return f

    def _get_jy(self):
        f = self['j'].component_fields()[1]
        f.pretty_name = "$J_y$"
        return f

    def _get_jz(self):
        f = self['j'].component_fields()[2]
        f.pretty_name = "$J_z$"
        return f

    def _get_b(self):
        bx, by, bz = self['bx'], self['by'], self['bz']
        opts = dict(_force_layout=self.force_vector_layout)
        return field.scalar_fields_to_vector([bx, by, bz], name="B", **opts)

    def _get_v(self):
        vx, vy, vz = self['vx'], self['vy'], self['vz']
        opts = dict(_force_layout=self.force_vector_layout)
        return field.scalar_fields_to_vector([vx, vy, vz], name="V", **opts)

    def _get_j(self):
        # b = self['b']
        # j = calc.curl(b)
        # j.name = 'j'
        # j.pretty_name = "J"
        # return j
        # TODO: make sure curl works with 2.5-D fields and 1-D fields
        raise NotImplementedError("Haven't hooked up the curl yet to get J")

    @staticmethod
    def _calc_mag(vx, vy, vz):
        if _has_numexpr:
            vmag = numexpr.evaluate("sqrt(vx**2 + vy**2 + vz**2)")
            return vx.wrap(vmag, fldtype="Scalar")
        else:
            vmag = np.sqrt(vx**2 + vy**2 + vz**2)
            return vmag

    def _get_bmag(self):
        bx, by, bz = self['bx'], self['by'], self['bz']
        bmag = self._calc_mag(bx, by, bz)
        bmag.name = "|B|"
        return bmag

    def _get_speed(self):
        vx, vy, vz = self['vx'], self['vy'], self['vz']
        vmag = self._calc_mag(vx, vy, vz)
        vmag.name = "|V|"
        return vmag

    def _get_jmag(self):
        jx, jy, jz = self['j'].component_fields()
        jmag = self._calc_mag(jx, jy, jz)
        jmag.name = "|J|"
        return jmag

    def _get_beta(self):
        return plasma.calc_beta(self['pp'], self['b'])

    def _get_psi(self):
        return plasma.calc_psi(self['b'])


class AthenaFile(object):
    """Mixin some Athena convenience stuff"""
    _grid_type = AthenaGrid


##
## EOF
##
