"""Loads a ggcm log file

Parses the view info """

from __future__ import print_function
import itertools
import re

import numpy as np

from viscid.readers import vfile


# FIXME: This lookup table is based on an enum in ggcm_mhd.h, ie, the numerical
#        values (index in this list) could change in the future - but libmrc
#        only writes out the integer value for mhd->ggcm_mhd_fld->mhd_type
#        into the log file, so a guess is better than nothing...
MHD_TYPES = ["MT_PRIMITIVE",
             # the following have B staggered the openggcm way: [-1..mx[
             "MT_SEMI_CONSERVATIVE_GGCM",
             # the following have B staggered the "normal" way: [0..mx]
             "MT_SEMI_CONSERVATIVE",
             "MT_FULLY_CONSERVATIVE",
             # cell-centered fully conservative MHD
             "MT_FULLY_CONSERVATIVE_CC",
             # the multi-moment schemes are cell-centered for all quantities
             "MT_GKEYLL"]


class GGCMLogFile(vfile.VFile):  # pylint: disable=W0223
    """Libmrc log file reader

    This class looks at the mrc_view info before the run starts
    to gather info about libmrc runtime parameters.

    Attributes:
        watched_classes (list): list of libmrc classes whose parameters
            will be loaded
    """
    _detector = None
    _grid_type = None

    watched_classes = ["ggcm_mhd",
                       "mrc_domain",
                       "mrc_crds",
                       "ggcm_mhd_ic",
                       "ggcm_dipole",
                       "ggcm_mhd_step",
                       "ggcm_mhd_fld"]

    info = None

    def _parse(self):
        _info = {}

        armed = False
        with open(self.fname, 'r') as f:
            # find end of view
            def is_timestep(s):
                return not s.strip().startswith(('cp=', 'step='))
            lines_iter = itertools.takewhile(is_timestep, f)
            for line in lines_iter:
                line = line.strip()
                if armed:
                    try:
                        key, val = self._parse_param(line)
                        _info["{0}_{1}".format(armed, key)] = val
                    except ValueError:
                        # this is expected for lines that look like
                        # "-------+------ type -- ???"
                        # as well as blank lines that mark the end of a
                        # section
                        clstype = re.match(r"-+\+-+ type -- (\w+)", line)
                        if clstype:
                            _info["{0}_type".format(armed)] = clstype.group(1)
                            # yes, keep armed, for super's parameters
                        elif line == "":
                            armed = False
                else:
                    try:
                        c = re.match(r"=+ class == (.+)", line).group(1)
                        c = c.strip()
                        if c in self.watched_classes:
                            armed = c
                            # the next lines just say
                            # "parameter  | value"
                            # "-----------|------"
                            # ignore them
                            next(lines_iter)
                            next(lines_iter)
                    except AttributeError:
                        # not the start of a new class view, that's ok
                        pass

        try:
            mhd_type = _info["ggcm_mhd_fld_mhd_type"]
            _info["ggcm_mhd_fld_mhd_type_str"] = MHD_TYPES[mhd_type]
        except (IndexError, KeyError):
            _info["ggcm_mhd_fld_mhd_type_str"] = "UNKNOWN"

        self.info = _info

    @staticmethod
    def _parse_value(s):
        """Parse a parameter and infer its type

        Parameters:
            s (str): the value of a libmrc parameter

        Returns:
            Either an int, float, string, or list of mixed types as
            inferred by the data
        """
        try:
            return int(s)
        except ValueError:
            pass

        try:
            return float(s)
        except ValueError:
            pass

        s = s.strip()
        if re.match(r"\-?[\d\.]+(\s*,\s*\-?[\d\.]+)+", s):
            l = s.split(",")
            for i, s in enumerate(l):
                try:
                    l[i] = int(s)
                except ValueError:
                    try:
                        l[i] = float(s)
                    except ValueError:
                        l[i] = s.strip()
            return np.array(l)
        elif re.match(r"[A-Za-z\d\.\-]+(\s*:\s*[A-Za-z\d\.\-]+)+", s):
            return s.split(":")
        else:
            return s

    @classmethod
    def _parse_param(cls, s):
        """Parse a libmrc view output line

        Parameters:
            s (str): full line of a parameter in the view

        Raises
            ValueError: If there is > 1 '|' character in s
        """
        key, val = [part.strip() for part in s.split("|")]
        val = cls._parse_value(val)
        return key, val

    def unload(self, **kwargs):
        self.info = {}
        super(GGCMLogFile, self).unload(**kwargs)

##
## EOF
##
