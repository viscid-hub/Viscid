""" Wrapper grid for some OpenGGCM convenience """

from __future__ import print_function
import re

import numpy as np

from viscid.readers import vfile
from viscid import field
from viscid import coordinate


class AthenaHstFile(vfile.VFile):  # pylint: disable=abstract-method
    """An Athena time history file"""
    _detector = r"^\s*(.*)\.(hst)\s*$"

    def __init__(self, fname, **kwargs):
        """
        Keyword Arguments:
            float_type_name (str): should be 'f4' or 'f8' if you know
                the data type of the file's data.
        """
        super(AthenaHstFile, self).__init__(fname, **kwargs)

    def _parse(self):
        # get field names from header
        with open(self.fname, 'r') as f:
            line = f.readline()
            line = f.readline().lstrip("#").strip()
            fld_names = re.split(r"\[[0-9]+\]=", line)[1:]
            fld_names = [fn.strip() for fn in fld_names]

        # crds here are really times
        dat = np.loadtxt(self.fname, unpack=True)
        t = dat[0]
        crds = coordinate.wrap_crds("nonuniform_cartesian", [('t', t)])

        g = self._make_grid(self, name="AthenaHstGrid")
        g.set_crds(crds)
        g.time = 0
        for i in range(1, len(fld_names)):
            fld = self._make_field(g, "Scalar", fld_names[i], crds, dat[i],
                                   center="Node")
            g.add_field(fld)
        self.add(g)
        self.activate(0)

##
## EOF
##
