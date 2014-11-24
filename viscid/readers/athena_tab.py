"""Athena ascii file reader"""

from __future__ import print_function
import os
import re
from itertools import takewhile, count

import numpy as np

from viscid.readers.vfile_bucket import VFileBucket
from viscid.readers import athena
from viscid.readers import vfile
from viscid import dataset
from viscid import field
from viscid import coordinate


class AthenaTabFile(athena.AthenaFile, vfile.VFile):  # pylint: disable=abstract-method
    """An Athena ascii file reader"""
    _detector = r"^\s*(.*)\.([0-9]{4})\.(tab)\s*$"

    _def_fld_center = "Cell"

    _collection = None

    _fld_list = None
    _crds = None

    def __init__(self, fname, vfilebucket=None, crds=None, fld_list=None,
                 **kwargs):
        if vfilebucket is None:
            vfilebucket = VFileBucket()
        self._fld_list = fld_list
        self._crds = crds

        super(AthenaTabFile, self).__init__(fname, vfilebucket, **kwargs)

    @classmethod
    def group_fnames(cls, fnames):
        return athena.group_athena_files_common(cls._detector, fnames)

    @classmethod
    def collective_name_from_group(cls, fnames):
        return athena.athena_collective_name_from_group(cls._detector,
                                                        fnames)

    def load(self, fname):
        if isinstance(fname, list):
            self._collection = fname
        else:
            self._collection = [fname]

        fname0 = self._collection[0]
        fname1 = self.collective_name(fname)

        basename = os.path.basename(fname0)
        self.info['run'] = re.match(self._detector, basename).group(1)

        super(AthenaTabFile, self).load(fname1)

    def _parse(self):
        if self._crds is None:
            meta = self.parse_header(self._collection[0])
            self._crds = self.read_crds(self._collection[0], dims=meta['dims'])
            self._fld_list = meta['fld_names']

        if len(self._collection) == 1:
            # load a single file
            _grid = self._parse_file(self.fname)
            self.add(_grid)
            self.activate(0)
        else:
            # load each file, and add it to teh bucket
            data_temporal = dataset.DatasetTemporal("AthenaTemporalCollection")

            for fname in self._collection:
                f = self.vfilebucket.load_file(fname, index_handle=False,
                                               file_type=type(self),
                                               crds=self._crds,
                                               fld_list=self._fld_list)
                data_temporal.add(f)
            data_temporal.activate(0)
            self.add(data_temporal)
            self.activate(0)

    def _parse_file(self, filename):
        # we do minimal file parsing here for performance. we just
        # make data wrappers from the templates we got from the first
        # file in the group, and package them up into grids

        # find the time from the first field's meta data
        meta = self.parse_header(filename)
        time = meta['time']

        _grid = self._make_grid("<AthenaGrid>")
        self.time = time
        _grid.time = time
        _grid.set_crds(self._crds)

        # make a DataWrapper and a Field for each template that we
        # have from the first file that we parsed, then add it to
        # the _grid
        data_wrapper = AthenaTabDataWrapper

        for i, fld_name in enumerate(self._fld_list):
            if self._def_fld_center.lower() == "cell":
                shape = self._crds.shape_cc
            else:
                shape = self._crds.shape_nc

            data = data_wrapper(filename, fld_name, shape, i)
            fld = field.wrap_field("Scalar", fld_name, self._crds,
                                   data, center=self._def_fld_center,
                                   time=time)
            _grid.add_field(fld)
        return _grid

    @staticmethod
    def parse_header(fname):
        meta = {}
        t = 0.0
        dims = []
        fld_names = []

        with open(fname, 'r') as f:
            for line in takewhile(lambda l: l.strip().startswith('#'), f):
                line = line.lstrip("#").strip()

                if line.startswith('Nx'):
                    dims.append(int(line.split('=')[1]))
                if "Time" in line:
                    t = re.search(r"Time\s*=\s*([0-9\.]+)", line).groups()[0]
                # il y a level & domain for SMR outptut, but i'm ignoring them

                fns = re.split(r"\[[0-9]+\]=", line)
                # print("line:", line, 'fns', fns)
                if len(fns) > 1:
                    # take off empty first element and strip the rest
                    fld_names = [fn.strip() for fn in fns[1:]]

            meta['time'] = float(t)
            # these dims are xyz... this is not the standard 'fortran ordering'
            # zyx of the data that all the other readers use
            meta['dims'] = dims
            # the first 2 * len(dims) are coordinates
            meta['fld_names'] = fld_names[2 * len(dims):]

        return meta

    @classmethod
    def read_crds(cls, fname, dims=None):
        # dims are xyz order unlike all other interfaces
        if dims is None:
            dims = cls.parse_header(fname)['dims']
        dat = np.loadtxt(fname, usecols=list(range(len(dims), 2 * len(dims))),
                         unpack=True, ndmin=2)

        dxmin = np.inf
        cclist = []
        nclist = []

        for i, dim, axis in zip(count(), dims, "xyz"):
            stop = int(np.prod(dims[:i + 1]))
            step = int(np.prod(dims[:i]))
            cc = dat[i][:stop:step]
            dxmin = np.min(dxmin, np.min(cc[1:] - cc[:-1]))
            assert len(cc) == dim
            cclist.append((axis, cc))

        for axis, cc in cclist:
            if len(cc) > 1:
                hd = 0.5 * (cc[1:] - cc[:-1])
                nc = np.hstack([cc[0] - hd[0],
                                cc[:-1] + hd,
                                cc[-1] + hd[-1]])
            else:
                hd = 0.5 * dxmin
                nc = np.array([cc[0] - hd, cc[0] + hd])
            nclist.append((axis, nc))

        # flip crds to zyx before wrapping
        return coordinate.wrap_crds("nonuniform_cartesian", nclist[::-1])

class AthenaTabDataWrapper(vfile.DataWrapper):
    filename = None
    fld_name = None
    expected_shape = None
    fld_number = None

    def __init__(self, filename, fld_name, expected_shape, fld_number):
        """Lazy wrapper for a field in a Tab (athena ascii) file

        Parameters:
            expected_shape (tuple): shape of data in the file (zyx)
        """
        super(AthenaTabDataWrapper, self).__init__()
        self.filename = filename
        self.fld_name = fld_name
        self.expected_shape = expected_shape
        self.fld_number = fld_number

    @property
    def shape(self):
        """
        Returns:
            zyx shape since that's the shape __array__ returns
        """
        return self.expected_shape

    @property
    def dtype(self):
        return np.dtype("float32")

    def __array__(self, *args, **kwargs):
        # the first 2 * nr_dims columns are for coordinates
        col = 2 * len(self.shape) + self.fld_number
        arr = np.loadtxt(self.filename, usecols=(col,), unpack=True)
        arr = arr.reshape(self.shape).astype(self.dtype)
        return arr

    def read_direct(self, *args, **kwargs):
        return self.__array__()

    def len(self):
        return self.shape[0]

    def __getitem__(self, item):
        return self.__array__().__getitem__(item)

##
## EOF
##
