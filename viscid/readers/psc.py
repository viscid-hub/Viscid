#! /usr/bin/env python

from __future__ import print_function

try:
    import h5py
    _HAVE_H5PY = True
except ImportError:
    _HAVE_H5PY = False

from viscid import grid
from viscid import field
from viscid.readers import xdmf
from viscid.calculator import plasma


class PscGrid(grid.Grid):
    downscale = 0
    d_i = 1.0  # used to rescale from d_e (natural normalization) to d_i

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

    def _assemble_vector(self, base_name, comp_names="xyz", forget_source=True,
                         **kwargs):

        opts = dict(forget_source=forget_source, **kwargs)

        if len(comp_names) == 3:
            vx = self[base_name + comp_names[0]]
            vy = self[base_name + comp_names[1]]
            vz = self[base_name + comp_names[2]]
            v = field.scalar_fields_to_vector([vx, vy, vz], name=base_name,
                                              **opts)
        else:
            comps = [self[base_name + c] for c in comp_names]
            v = field.scalar_fields_to_vector(comps, name=base_name, **opts)
            for comp in comps:
                comp.unload()
        return v

    def _get_b(self):
        return self._assemble_vector("b", _force_layout=self.force_vector_layout,
                                     pretty_name="B")

    def _get_h(self):
        return self._assemble_vector("h", _force_layout=self.force_vector_layout,
                                     pretty_name="H")

    def _get_e(self):
        return self._assemble_vector("e", _force_layout=self.force_vector_layout,
                                     pretty_name="E")

    def _get_j(self):
        return self._assemble_vector("j", _force_layout=self.force_vector_layout,
                                     pretty_name="J")

    def _get_psi(self):
        return plasma.calc_psi(self['b'])


class PscFieldFile(xdmf.FileXDMF):  # pylint: disable=W0223
    _detector = r"^\s*(.*/)?(.*)fd(?:\.([0-9]{6}))?\.xdmf"
    _grid_type = PscGrid

    def __init__(self, fname, *args, **kwargs):
        # print("Opening '%s'" % (fname))
        super(PscFieldFile, self).__init__(fname, *args, **kwargs)


class PscParticles(object):
    def __init__(self, path, step):
        filename = "%s/prt.%06d_p%06d.h5" % (path, step, 0)
        # print("Opening '%s'" % (filename))
        if not _HAVE_H5PY:
            raise RuntimeError("Can't load psc particles w/o h5py")
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
