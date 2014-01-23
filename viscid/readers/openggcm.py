#!/usr/bin/env python
""" Wrapper grid for some OpenGGCM convenience """

from .. import grid
from . import xdmf


class GGCMGrid(grid.Grid):
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
    _detector = r"^\s*.*\.(p[xyz]_[0-9]+|3d|3df|iof)" \
                r"(\.[0-9]{6})?\.(xmf|xdmf)\s*$"
    _grid_type = GGCMGrid

##
## EOF
##
