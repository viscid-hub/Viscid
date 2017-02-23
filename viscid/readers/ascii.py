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


class FileNDASCII(vfile.VFile):  # pylint: disable=W0223
    """Read ascii files with the following format:

    NDim: <NDimensions>
    Nfields: <NFields>
    <Axis Name> <Axis Dtype>
    <Crd data as space separated values>
    # ... repeat for the remaining axes
    <Field Name> <Field Dtype>
    <Field as space separated values with last axis as fastest index>
    # ... repeat for the remaining fields
    """
    _detector = r".*\.(nda)\s*$"  # NOTE: detect type is overridden

    def __init__(self, fname, **kwargs):
        super(FileNDASCII, self).__init__(fname, **kwargs)

    def _parse(self):
        g = self._make_grid(self)

        with open(self.fname, 'r') as fin:
            ndim = int(fin.readline().strip().split()[1])
            nfields = int(fin.readline().strip().split()[1])

            clist = []

            for _ in range(ndim):
                ax_name, ax_dtype = fin.readline().strip().split()
                crd_dat = np.fromstring(fin.readline().strip(), sep=' ',
                                        dtype=ax_dtype)
                clist.append([ax_name, crd_dat])

            crds = coordinate.wrap_crds("nonuniform_cartesian", clist)
            g.set_crds(crds)

            for _ in range(nfields):
                fld_name, fld_dtype = fin.readline().strip().split()
                fld_dat = np.fromstring(fin.readline().strip(), sep=' ',
                                        dtype=fld_dtype)
                fld = self._make_field(g, "Scalar", fld_name, crds,
                                       fld_dat)
                g.add_field(fld)

        self.add(g)
        self.activate(0)


##
## EOF
##
