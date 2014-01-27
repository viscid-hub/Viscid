#! /usr/bin/env python

from __future__ import print_function

import numpy as np
import h5py

from .. import grid
from .. import field
from . import xdmf


class PscGrid(grid.Grid):
    downscale = 0
    d_i = 1.0 # used to rescale from d_e (natural normalization) to d_i

    def _get_bx(self):
        return self["hx"]

    def _get_by(self):
        return self["hy"]

    def _get_bz(self):
        return self["hz"]

    def _get_epar(self):
        hx, hy, hz = self["hx"], self["hy"], self["hz"]
        ex, ey, ez = self["ex"], self["ey"], self["ez"]
        return (ex * hx + ey * hy + ez * hz) * (hx**2 + hy**2 + hz**2)**-.5

    def _get_psi(self):
        hz, hy = self["hz"], self["hy"]
        crd_z, crd_y = hz.get_crds_nc(['z', 'y'])
        dz = crd_z[1] - crd_z[0]
        dy = crd_y[1] - crd_y[0]
        nz, ny, _ = hy.shape

        A = np.empty((nz, ny))
        hz = hz.data.reshape(nz, ny)
        hy = hy.data.reshape(nz, ny)
        A[0, 0] = 0.0
        for i in range(1, nz):
            A[i, 0] = A[i - 1, 0] + dz * (hy[i, 0] + hy[i - 1, 0]) / 2.0

        for j in range(1, ny):
            A[:, j] = A[:, j - 1] - dy * (hz[:, j - 1] + hz[:, j]) / 2.0

        return field.wrap_field("Scalar", "psi", self["hz"].crds, A,
                                center="Cell")


class PscFieldFile(xdmf.FileXDMF):  # pylint: disable=W0223
    _detector = r"^\s*(.*/)?(.*)fd\.([0-9]{6}).xdmf"
    _grid_type = PscGrid

    def __init__(self, fname, *args, **kwargs):
        print("Opening '%s'" % (fname))
        super(PscFieldFile, self).__init__(fname, *args, **kwargs)


class PscParticles(object):
    def __init__(self, path, step):
        filename = "%s/prt.%06d_p%06d.h5" % (path, step, 0)
        print("Opening '%s'" % (filename))
        self._h5file = h5py.File(filename, 'r')

        # path = _find_path(self._h5file, "psc")
        # self.time = self._h5file[path].attrs["time"][0]
        # self.timestep = self._h5file[path].attrs["timestep"][0]

        self.data = self._h5file["particles/p0/1d"]


def open_psc_file(path, step, pfx="p"):
    """ WARNING: using this goes around the file bucket, be careful
    not to call this twice on the same file. Nothing bad will happen,
    you'll just have 2 versions in memory.
    """
    return PscFieldFile(make_fname(path, step, pfx))

def make_fname(path, step, pfx="p"):
    return "{0}/{1}fd.{2:.06d}.xdmf".format(path, pfx, step)

##
## EOF
##
