#!/usr/bin/env python

# import string
from __future__ import print_function
import re
from warnings import warn

import numpy as np

from . import vfile
# from .. import grid


class FileASCII(vfile.VFile):
    _detector = r".*\.(txt|asc)\s*$"  # NOTE: detect type is overridden

    meta = {'format': 'none',  # or ascii1d, ascii2d, etc.
            'type': 'normal',  # or time-series
            'delimiter': ',',
            'comments': '#'}

    def __init__(self, fname, **kwargs):
        super(FileASCII, self).__init__(fname, **kwargs)

    def _parse(self):
        self._parse_meta()

    def _parse_meta(self):
        with open(self.fname, 'r') as f:
            line = f.readline().strip().lower()
            if line.startswith('#(meta)'):
                mdat = re.findall(
                        r"(\S+)\s*=\s*(?:['\"](.*?)['\"]|(.+?))(?:\s|$)",
                        line,
                        )
                for m in mdat:
                    key = m[0]
                    val = m[1] if m[1] is not '' else m[2]
                    try:
                        self.meta[key] = val
                    except KeyError:
                        warn("Invalid option in {0}: {1} = {2}".format(
                            self.fname, key, val))

    @classmethod
    def detect_type(cls, fname):
        for filetype in cls.__subclasses__():
            td = filetype.detect_type(fname)
            if td:
                return td
        if cls._detector and re.match(cls._detector, fname):
            #  this is the overloaded bit... decide from first
            # line of file how many dimensions it is, else default
            # to 1D
            with open(fname, 'r') as f:
                line = f.readline().lower()
                if line.find("ascii1d") >= 0:
                    return FileASCII1D
                elif line.find("ascii2d") >= 0:
                    return FileASCII2D
                elif line.find("ascii3d") >= 0:
                    return FileASCII3D
                else:
                    return FileASCII1D
        return None


class FileASCII1D(FileASCII):
    """  """
    _detector = r".*\.(asc1)\s*$"

    time_series = False

    def __init__(self, fname, **kwargs):
        super(FileASCII1D, self).__init__(fname=fname, **kwargs)

    def _parse(self):
        # get any possible meta data from file header
        super(FileASCII1D, self)._parse()

        try:
            grid_name = self.meta['name']
        except KeyError:
            grid_name = self.fname

        grd = data_item.RectilinearGrid(grid_name)

        if self.time_series:
            mode = 0
            with open(self.fname, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line is '' or line.startswith('#'):
                        continue

                    if mode == 0:
                        x = np.fromstring(f.readline(), sep=' ', dtype='f')
                        mode += 1
                    elif mode == 1:
                        for i, line in enumerate(f):
                            first_space = line.find(self.meta['delimiter'])
                            t = float(line[:first_space])
                            y = np.fromstring(line[first_space:],
                                    sep=self.meta['delimiter'],
                                    dtype='f')
        else:
            dat = np.loadtxt(self.fname, unpack=True,
                             comments=self.meta['comments'])
            grd.set_coords([dat[0]])
            for i, d in enumerate(dat[1:]):
                grd.add_field("y{0}".format(i), d)

        self._add_grid(grd)


class FileASCII2D(FileASCII):
    """  """
    _detector = r".*\.(asc2)\s*$"

    def __init__(self, fname, **kwargs):
        super(FileASCII2D, self).__init__(fname, **kwargs)
        self.parse(fname)

    def _parse(self, fname):
        raise NotImplemented("ascii 2d not implemented")


class FileASCII3D(FileASCII):
    """  """
    _detector = r".*\.(asc3)\s*$"

    def __init__(self, fname, **kwargs):
        super(FileASCII3D, self).__init__(fname, **kwargs)
        self.parse(fname)

    def _parse(self, fname):
        raise NotImplemented("ascii 3d not implemented")

if __name__ == '__main__':
    import sys
    import os
    import viscid

    _viscid_root = os.path.dirname(viscid.__file__)
    f1 = FileASCII(_viscid_root + '/../../sample/test.asc')
    f2 = FileASCII(_viscid_root + '/../../sample/test_time.asc')
    sys.stderr.write("{0}\n".format(f1))
    sys.stderr.write("{0}\n".format(f2))

##
## EOF
##
