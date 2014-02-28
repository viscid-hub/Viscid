#!/usr/bin/env python
""" super simple reader for gnuplot-like 1d ascii data
WARNING: not lazy """

# import string
from __future__ import print_function
# import logging

import numpy as np

from . import vfile
from .. import coordinate
from .. import field

class FileASCII(vfile.VFile):  # pylint: disable=W0223
    """ open an ascii file with viscid format, or if not specified, assume
    it's in gnuplot format, gnuplot format not yet specified """
    _detector = r".*\.(txt|asc)\s*$"  # NOTE: detect type is overridden

    def __init__(self, fname, **kwargs):
        super(FileASCII, self).__init__(fname, **kwargs)

    def _parse(self):
        g = self._grid_type(**self._grid_opts)

        arr = np.loadtxt(self.fname)
        crds = coordinate.wrap_crds("nonuniform_cartesian", [['x', arr[:, 0]]])
        g.set_crds(crds)

        if len(arr.shape) > 1:
            for i in range(arr.shape[1] - 1):
                fld = field.wrap_field("Scalar", str(i), crds, arr[:, i + 1])
                g.add_field(fld)

        self.add(g)
        self.activate(0)

##
## EOF
##
