"""Loads a ggcm log file

Parses the view info """

from __future__ import print_function

import itertools
import re


class GGCMLogFile(object):  # pylint: disable=W0223
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
                       "ggcm_mhd_ic",
                       "ggcm_dipole"]

    fname = None
    info = None

    def __init__(self, fname, *args, **kwargs):  # pylint: disable=unused-argument
        self.fname = fname
        self._parse()

    def _parse(self):
        _info = {}

        armed = False
        with open(self.fname, 'r') as f:
            # find end of view
            is_timestep = lambda s: not s.strip().startswith(('cp=', 'step='))
            lines_iter = itertools.takewhile(is_timestep, f)
            for line in lines_iter:
                line = line.strip()
                if armed:
                    try:
                        key, val = self._parse_param(line)
                        _info[key] = val
                    except ValueError:
                        # this is expected for lines that look like
                        # "-------+------ type -- ???"
                        # as well as blank lines that mark the end of a
                        # section
                        if line == "":
                            armed = False
                else:
                    try:
                        c = re.match(r"=+ class == (.+)", line).group(1)
                        c = c.strip()
                        if c in self.watched_classes:
                            armed = True
                            # the next lines just say
                            # "parameter  | value"
                            # "-----------|------"
                            # ignore them
                            lines_iter.next()
                            lines_iter.next()
                    except AttributeError:
                        # not the start of a new class view, that's ok
                        pass

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
        if re.match(r"[A-Za-z\d\.]+(\s*:\s*[A-Za-z\d\.]+)+", s):
            l = s.split(":")
            for i, s in enumerate(l):
                try:
                    l[i] = int(s)
                except ValueError:
                    try:
                        l[i] = float(s)
                    except ValueError:
                        l[i] = s.strip()
            return l
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

    def unload(self):
        self.info = {}

##
## EOF
##
