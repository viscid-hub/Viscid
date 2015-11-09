#!/usr/bin/env python
""" super simple reader for gnuplot-like 1d ascii data
WARNING: not lazy

Fields are named "c%d" where %d is the column number. Numbers are 0
based, but the first column is interpreted as coordinates, so the
first field is column 1 in the file and the grid.
"""

# import string
from __future__ import print_function

import numpy as np

# from viscid import logger
from viscid.readers import vfile
from viscid import coordinate
from viscid import field

class FileASCII(vfile.VFile):  # pylint: disable=W0223
    """ open an ascii file with viscid format, or if not specified, assume
    it's in gnuplot format, gnuplot format not yet specified """
    _detector = r".*\.(txt|asc)\s*$"  # NOTE: detect type is overridden

    def __init__(self, fname, **kwargs):
        super(FileASCII, self).__init__(fname, **kwargs)

    def _parse(self):
        g = self._make_grid(self)

        arr = np.loadtxt(self.fname)
        crds = coordinate.wrap_crds("nonuniform_cartesian", [['x', arr[:, 0]]])
        g.set_crds(crds)

        if len(arr.shape) > 1:
            for i in range(1, arr.shape[1]):
                fld = self._make_field(g, "Scalar", 'c' + str(i), crds,
                                       arr[:, i])
                g.add_field(fld)

        self.add(g)
        self.activate(0)

##
## EOF
##
