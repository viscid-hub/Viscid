""" Athena binary file reader

At the moment, this reader assumes the conserved fields are in the
files. The Athena custom Grid has methods to do vx = M1 / d, etc.
"""

from __future__ import print_function
import os
import re

import numpy as np

from viscid import glob2
from viscid.readers import vfile
from viscid.readers.vfile_bucket import ContainerFile
from viscid.readers import athena
from viscid import coordinate


class AthenaBinFile(athena.AthenaFile, ContainerFile):  # pylint: disable=abstract-method
    """An Athena binary file reader"""
    _detector = r"^\s*(.*)\.([0-9]+)\.(bin)\s*$"

    _def_fld_center = "Cell"

    _collection = None

    _fwrapper = None
    float_type_name = None
    var_type = None
    _crds = None

    def __init__(self, fname, crds=None, float_type_name=None, var_type=None,
                 **kwargs):
        """
        Keyword Arguments:
            float_type_name (str): should be 'f4' or 'f8' if you know
                the data type of the file's data.
            var_type (str): either 'cons' or 'prim'
        """
        # there is no parent bucket, so we need to new one up for children
        self.float_type_name = float_type_name
        self.var_type = var_type
        self._crds = crds
        super(AthenaBinFile, self).__init__(fname, **kwargs)

    @classmethod
    def group_fnames(cls, fnames):
        return athena.group_athena_files_common(cls._detector, fnames)

    @classmethod
    def collective_name_from_group(cls, fnames):
        return athena.athena_collective_name_from_group(cls._detector,
                                                        fnames)

    def get_file_wrapper(self, filename):
        if self._fwrapper is None:
            self._fwrapper = AthenaBinFileWrapper(filename,
                                                  float_type_name=self.float_type_name,
                                                  var_type=self.var_type)
        else:
            assert (self._fwrapper.filename == filename or
                    glob2(self._fwrapper.filename) == glob2(filename))
        return self._fwrapper

    def set_file_wrapper(self, wrapper):
        raise NotImplementedError("This must be done at file init")

    def load(self, fname):
        if isinstance(fname, list):
            self._collection = fname
        else:
            self._collection = [fname]

        fname0 = self._collection[0]
        fname1 = self.collective_name(fname)

        basename = os.path.basename(fname0)
        self.set_info('run', re.match(self._detector, basename).group(1))

        super(AthenaBinFile, self).load(fname1)

    def _parse(self):
        if self._crds is None:
            self._crds = self._make_crds(self._collection[0])

        if len(self._collection) == 1:
            # load a single file
            _grid = self._parse_file(self.fname, self)
            self.add(_grid)
            self.activate(0)
        else:
            # load each file, and add it to teh bucket
            data_temporal = self._make_dataset(self, dset_type="temporal",
                                               name="AthenaTemporalCollection")

            for fname in self._collection:
                f = self._load_child_file(fname, index_handle=False,
                                          file_type=type(self),
                                          crds=self._crds,
                                          float_type_name=self.float_type_name,
                                          var_type=self.var_type)
                data_temporal.add(f)
            data_temporal.activate(0)
            self.add(data_temporal)
            self.activate(0)

    def _parse_file(self, filename, parent_node):
        # we do minimal file parsing here for performance. we just
        # make data wrappers from the templates we got from the first
        # file in the group, and package them up into grids

        # find the time from the first field's meta data
        _file_wrapper = self.get_file_wrapper(filename)
        _file_wrapper.read_header()
        time = _file_wrapper.time

        _grid = self._make_grid(parent_node, name="<AthenaGrid>")
        self.time = time
        _grid.time = time
        _grid.set_crds(self._crds)

        # make a DataWrapper and a Field for each template that we
        # have from the first file that we parsed, then add it to
        # the _grid
        data_wrapper = AthenaBinDataWrapper

        for i, fld_name in enumerate(_file_wrapper.fld_names):
            if self._def_fld_center.lower() == "cell":
                shape = self._crds.shape_cc
            else:
                shape = self._crds.shape_nc

            data = data_wrapper(_file_wrapper, fld_name,
                                shape[::-1], i)
            fld = self._make_field(_grid, "Scalar", fld_name,
                                   self._crds, data, time=time,
                                   center=self._def_fld_center,
                                   zyx_native=True)
            _grid.add_field(fld)
        return _grid

    def _make_crds(self, filename):
        fw = AthenaBinFileWrapper(filename, keep_crd_clist=True,
                                  float_type_name=self.float_type_name,
                                  var_type=self.var_type)
        with fw as f:
            crd_clist = f.crd_clist
            new_clist = []
            dxmin = np.inf
            for c in crd_clist:
                if len(c[1]) > 1:
                    dxmin = np.min([dxmin, np.min(c[1][1:] - c[1][:-1])])
            for i, cli in enumerate(crd_clist):
                cc = cli[1]
                try:
                    hd = 0.5 * (cc[1:] - cc[:-1])
                    nc = np.hstack([cc[0] - hd[0],
                                    cc[:-1] + hd,
                                    cc[-1] + hd[-1]])
                except IndexError:
                    dxminh = 0.5 * dxmin
                    nc = np.array([cc[0] - dxminh, cc[0] + dxminh])
                new_clist.append([crd_clist[i][0], nc])
            crds = coordinate.wrap_crds("nonuniform_cartesian", new_clist[::-1])
        return crds


class AthenaBinFileWrapper(object):
    """A File-like object for interfacing with Athena binary files

    Attributes:
        float_type_name (str): default float data type, should be
            'f4' or 'f8'; deufaults to double ('f8')
    """
    # translation is is purely for convenience
    float_type_name = "f8"
    var_type = "cons"

    _file = None
    _loc_after_header = None
    _endian = None
    _float_dtype = None  # = np.dtype(_endian + float_type_name)

    filename = None
    keep_crd_clist = None

    fld_names = None
    nvars = None
    nscalars = None
    shape = None
    count = None
    _file_meta = None
    time = None
    dt = None
    crd_clist = None

    def __init__(self, filename, keep_crd_clist=False, float_type_name=None,
                 var_type=None):
        self.filename = filename
        self.keep_crd_clist = keep_crd_clist
        if float_type_name is not None:
            self.float_type_name = float_type_name
        if var_type is not None:
            self.var_type = var_type

    def __del__(self):
        self.close()

    @property
    def float_dtype(self):
        if self._float_dtype is None:
            with self as _:
                # just opening the file makes it read the meta data
                pass
        return self._float_dtype

    @property
    def field_names(self):
        if self._fld_names_lookup is None:
            with self as _:
                # just opening the file makes it read the meta data
                pass

    def read_field(self, fld_id):
        """Read a field given a seekable location

        Parameters:
            fld_id(int): number of field in file

        Returns:
            tuple array
        """
        if fld_id >= self.nvars:
            raise IndexError("File {0} only has {1} fields, you asked for "
                             "fld number {2}".format(self.filename,
                                                     self.nvars, fld_id))

        fld_size_bytes = self.count * self._float_dtype.itemsize
        self._file.seek(self._loc_after_header + fld_id * fld_size_bytes)
        data = np.fromfile(self._file, dtype=self._float_dtype,
                           count=self.count)
        # return ndarray as native endian
        return data.astype(self._float_dtype.name)

    def read_header(self):
        if self._endian is None:
            with self as _:
                # just opening the file makes it read the header
                pass

    def open(self):
        if self._file is None:
            self._file = open(self.filename, 'rb')
            try:
                if self._endian is None:
                    self._read_file_header()
            except IOError as e:
                self.close()
                raise e

    @property
    def isopen(self):
        return self._file is not None

    def close(self):
        if self._file is not None:
            f = self._file
            self._file = None
            f.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, value, traceback):
        self.close()

    def _read_file_header(self):
        """load up the file's meta data"""
        self._file.seek(0, 0)

        coordsys = np.fromfile(self._file, dtype="<i", count=1)[0]
        dims = np.fromfile(self._file, dtype="<i", count=5)

        # if nvar makes sense, we were right, use little endian
        if dims[3] < 1000:
            self._endian = "<"
        else:
            self._endian = ">"
            coordsys = coordsys.byteswap()
            dims = dims.byteswap()

        nx, ny, nz = dims[:3]
        nvars, nscalars = dims[3:5]  # pylint: disable=unused-variable

        dtyp_int = np.dtype(self._endian + "i4")  # 32bit int
        self._float_dtype = np.dtype(self._endian + self.float_type_name)

        # ignore self_gravity and particles flags for now
        _, _ = np.fromfile(self._file, dtype=dtyp_int, count=2)

        # ignore gamm1 and cs for now
        _, _ = np.fromfile(self._file, dtype=self._float_dtype, count=2)

        # determine field names from hints about how athena was run
        # for BAROTROPIC EOS HDYRO: NVAR = 4 + NSCALARS
        # for BAROTROPIC EOS MHD: NVAR = 7 + NSCALARS
        # for ADIABATIC HDYRO: NVAR = 5 + NSCALARS
        # for ADIABATIC MHD: NVAR = 8 + NSCALARS
        if self.var_type == "cons":
            _all_fld_names = ['d', 'M1', 'M2', 'M3', 'E', "B1", "B2", "B3"]
        elif self.var_type == "prim":
            _all_fld_names = ['d', 'V1', 'V2', 'V3', 'P', "B1c", "B2c", "B3c"]
        else:
            raise RuntimeError("var_type must be cons or prim")

        nprim_flds = nvars - nscalars
        fld_names = _all_fld_names[:4]
        # adiabatic runs have an 'E' (energy)
        if nprim_flds in (5, 8):
            fld_names.append(_all_fld_names[4])
        # MHD runs have magnetic fields
        if nprim_flds in (7, 8):
            fld_names.extend(_all_fld_names[5:8])
        fld_names.extend((str(i) for i in range(nprim_flds, nvars)))
        self.fld_names = fld_names

        self.time, self.dt = np.fromfile(self._file, dtype=self._float_dtype,
                                         count=2)
        x = np.fromfile(self._file, dtype=self._float_dtype, count=nx)
        y = np.fromfile(self._file, dtype=self._float_dtype, count=ny)
        z = np.fromfile(self._file, dtype=self._float_dtype, count=nz)

        if self.keep_crd_clist:
            self.crd_clist = [('z', z.astype(self._float_dtype.name)),
                              ('y', y.astype(self._float_dtype.name)),
                              ('x', x.astype(self._float_dtype.name))]
        self.nvars = nvars
        self.nscalars = nscalars
        self.shape = (nz, ny, nx)
        self.count = np.prod(self.shape)
        self._loc_after_header = self._file.tell()

        return None


class AthenaBinDataWrapper(vfile.DataWrapper):
    file_wrapper = None
    filename = None
    fld_name = None
    expected_shape = None
    fld_number = None

    def __init__(self, file_wrapper, fld_name, expected_shape, fld_number):
        """Lazy wrapper for a field in a Fortbin file

        Parameters:
            expected_shape (tuple): shape of data in the file (zyx)
        """
        super(AthenaBinDataWrapper, self).__init__()
        self.file_wrapper = file_wrapper
        self.filename = file_wrapper.filename
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
        return self.file_wrapper.float_dtype

    def __array__(self, *args, **kwargs):
        with self.file_wrapper as f:
            arr = f.read_field(self.fld_number).reshape(self.expected_shape)
        return arr.astype(self.dtype)

    def read_direct(self, *args, **kwargs):
        return self.__array__()

    def len(self):
        return self.shape[0]

    def __getitem__(self, item):
        return self.__array__().__getitem__(item)

##
## EOF
##
