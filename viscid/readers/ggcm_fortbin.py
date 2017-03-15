from __future__ import print_function
import struct
import os
import re
from datetime import datetime, timedelta

import numpy as np

from viscid import grid
from viscid.readers import vfile
from viscid.readers import openggcm
from viscid.compat import OrderedDict


# raise NotImplementedError("fortbin reader is not at all")


class GGCMFortbinFileWrapper(object):
    """A File-like object for interfacing with OpenGGCM binary files"""
    _file = None
    _endian = None

    filename = None
    _file_meta = None
    fields_seen = None
    seen_all_fields = None

    def __init__(self, filename):
        self.filename = filename
        self.fields_seen = OrderedDict()
        self.seen_all_fields = False

    def __del__(self):
        self.close()

    @property
    def file_meta(self):
        if self._file_meta is None:
            with self as _:
                # just opening the file makes it read the meta data
                pass
        return self._file_meta

    def read_field(self, fld_name, pos=None):
        """Read a field given a seekable location

        Parameters:
            fld_name(str): name of field we're expecting to read
            pos(int): position in file we can seek to

        Returns:
            tuple (field name, dict of meta data, array)
        """

        if pos is not None:
            self._file.seek(pos)
            found_fld, meta = self.inquire_next()
            if found_fld != fld_name:
                raise ValueError("The file {0} didn't contain field {1} at "
                                 "position {2}".format(self.filename,
                                                       fld_name, pos))
        else:
            meta = self.inquire(fld_name)

        self._file.seek(meta['file_position'] + meta['header_size'])
        data = np.fromfile(self._file, dtype=np.dtype(self._endian + 'f'),
                           count=meta['nelem'])
        return meta, data.reshape(meta['dims'], order='F')

    def inquire_all_fields(self, reinquire=False):
        if reinquire:
            self.seen_all_fields = False
            self.fields_seen = OrderedDict()

        if self.seen_all_fields:
            return

        self._file.seek(0)
        while not self.seen_all_fields:
            self.inquire_next()
            self._file.seek(self.file_meta['nbytes'], 1)

    def inquire(self, fld_name):
        try:
            meta = self.fields_seen[fld_name]
            self.seek(meta['file_position'])
            return meta
        except KeyError:
            try:
                last_added = next(reversed(self.fields_seen))
                # go to the last seen field and go one field past it
                self.seek(self.fields_seen[last_added]['file_position'] +
                          self.file_meta['nbytes'])
            except StopIteration:
                self._file.seek(0)

            while not self.seen_all_fields:
                found_fld_name, meta = self.inquire_next()
                if found_fld_name == fld_name:
                    return meta
                self._file.seek(self.file_meta['nbytes'], 1)

            raise KeyError("file '{0}' has no field '{1}'"
                           "".format(self.filename, fld_name))

    def inquire_next(self):
        """Collect the meta-data from the next field in the file

        Returns:
            tuple (field name, dict of meta data) both
            of which will be None if there are no more Fields

        Note:
            After this operation is done, the file-pointer will be reset
            to the position it was before the inquiry.
        """
        if not self.isopen:
            raise RuntimeError("file is not open")

        try:
            fld_name, meta = self._read_header()
        except IOError:
            fld_name, meta = None, None

        if not fld_name:
            self.seen_all_fields = True
            return None, None

        if fld_name not in self.fields_seen:
            self.fields_seen[fld_name] = meta

        return fld_name, meta

    def open(self):
        if self._file is None:
            self._file = open(self.filename, 'rb')
            try:
                if self._endian is None or self._file_meta is None:
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

    def _read_file_header(self, data_size=4):
        """load up the file's meta data"""
        assert self._file.tell() == 0
        _, meta = self._read_header(data_size=data_size)
        self._file_meta = meta

    def _read_header(self, data_size=4):
        """read a field's header; returns None, None if EOF
        Raises:
            Both of the following are raised if there are no more
            fields in a file (although it'll more likely be an IOError)

            IOError: couldn't detect endian
            struct.error: not enough lines in file to get a header
        """
        if not self.isopen:
            raise RuntimeError("Trying to read header, but file is closed.")

        try:
            pos = self._file.tell()
            endian_marker = self._file.read(4)
            if endian_marker == struct.pack('<i', 2):
                self._endian = '<'
            elif endian_marker == struct.pack('>i', 2):
                self._endian = '>'
            else:
                raise IOError("Can't detect endian, not a fortbin file: "
                              "{0}".format(self.filename))

            inttime = struct.unpack(self._endian + 'i', self._file.read(4))[0]
            ndim = struct.unpack(self._endian + 'i', self._file.read(4))[0]
            dims = struct.unpack(self._endian + '{0}i'.format(ndim),
                                 self._file.read(4 * ndim))

            fld_name = self._file.read(80).decode().strip()
            asciitime = self._file.read(80).decode()

            header_size = 4 + 4 + 4 + (4 * ndim) + (2 * 80)
            nelem = np.prod(dims)
            nbytes = header_size + data_size * nelem

            fld_meta = dict(header_size=header_size,
                            timestr=asciitime,
                            inttime=inttime,
                            ndim=ndim,
                            dims=dims,
                            nelem=nelem,
                            nbytes=nbytes,
                            file_position=pos)
            self._file.seek(pos)

            if fld_name == "":
                return None, None
            else:
                return fld_name, fld_meta
        except struct.error:
            return None, None


class FortbinDataWrapper(vfile.DataWrapper):
    """Interface for lazily pointing to a field in a binary file"""
    file_wrapper = None
    filename = None
    fld_name = None
    expected_shape = None
    file_position = None

    def __init__(self, file_wrapper, fld_name, expected_shape, file_position):
        """Lazy wrapper for a field in a Fortbin file

        Parameters:
            expected_shape (tuple): shape of data in the file (xyz)
        """
        super(FortbinDataWrapper, self).__init__()
        self.file_wrapper = file_wrapper
        self.filename = file_wrapper.filename
        self.fld_name = fld_name
        self.expected_shape = expected_shape
        self.file_position = file_position

    @property
    def shape(self):
        """
        Returns:
            zyx shape since that's the shape __array__ returns
        """
        return self.expected_shape[::-1]

    @property
    def dtype(self):
        return np.dtype("float32")

    def __array__(self, *args, **kwargs):
        with self.file_wrapper as f:
            # fld_name, meta, arr = f.read_field_at(self.loc, ndim)
            meta, arr = f.read_field(self.fld_name, pos=self.file_position)
            arr = np.array(arr.flatten(order='F').reshape(meta['dims'][::-1]),
                           order='C')

        # meta's dims are xyz (from file), but ex
        if meta['dims'] != self.expected_shape:
            raise RuntimeError("Field '{0}' from file '{1}' has shape {2} "
                               "instead of {3}".format(
                                   self.fld_name,
                                   self.filename, meta['dims'],
                                   self.expected_shape))
        return arr.astype(self.dtype)

    def read_direct(self, *args, **kwargs):
        return self.__array__()

    def len(self):
        return self.shape[0]

    def __getitem__(self, item):
        return self.__array__().__getitem__(item)


class GGCMFileFortbinMHD(openggcm.GGCMFileFortran):  # pylint: disable=abstract-method
    """Binary files"""
    _detector = r"^\s*(.*)\.(p[xyz]_[0-9]+|3df)" \
                r"\.([0-9]{6}).b\s*$"

    _fwrapper_type = GGCMFortbinFileWrapper
    _data_item_templates = None
    _def_fld_center = "Cell"

    def __init__(self, filename, **kwargs):
        super(GGCMFileFortbinMHD, self).__init__(filename, **kwargs)

    def _shape_discovery_hack(self, filename):
        with self.get_file_wrapper(filename) as f:
            _, meta = f.inquire_next()
        return meta['dims']

    def _parse_file(self, filename, parent_node):
        # we do minimal file parsing here for performance. we just
        # make data wrappers from the templates we got from the first
        # file in the group, and package them up into grids

        # find the time from the first field's meta data
        int_time = int(re.match(self._detector, filename).group(3))
        time = float(int_time)

        _grid = self._make_grid(parent_node, name="<FortbinGrid>",
                                **self._grid_opts)
        self.time = time
        _grid.time = time
        _grid.set_crds(self._crds)

        templates = self._fld_templates
        if templates is None:
            templates = self._make_template(filename)

        # make a DataWrapper and a Field for each template that we
        # have from the first file that we parsed, then add it to
        # the _grid
        if self._iono:
            data_wrapper = FortbinDataWrapper
        else:
            data_wrapper = FortbinDataWrapper

        for item in templates:
            data = data_wrapper(self.get_file_wrapper(filename),
                                item['fld_name'], item['shape'],
                                item['file_position'])
            fld = self._make_field(_grid, "Scalar", item['fld_name'],
                                   self._crds, data, center=self._def_fld_center,
                                   time=time, zyx_native=True)
            _grid.add_field(fld)
        return _grid

    def _make_template(self, filename):
        """read meta data for all fields in a file to get
        a list of field names and shapes, all the required info
        to make a FortbinDataWrapper
        """
        with self.get_file_wrapper(filename) as f:
            f.inquire_all_fields()
            template = []

            meta = None
            for fld_name, meta in f.fields_seen.items():
                d = dict(fld_name=fld_name,
                         shape=meta['dims'],
                         file_position=meta['file_position'])
                template.append(d)

            if meta is not None:
                if self.find_info('basetime', default=None) is None:
                    basetime, _ = self.parse_timestring(meta['timestr'])
                    if self.parents:
                        self.parents[0].set_info("basetime", basetime)
                    else:
                        self.set_info("basetime", basetime)

        return template

    @classmethod
    def collective_name_from_group(cls, fnames):
        fname0 = fnames[0]
        basename = os.path.basename(fname0)
        run = re.match(cls._detector, basename).group(1)
        fldtype = re.match(cls._detector, basename).group(2)
        new_basename = "{0}.{1}.b".format(run, fldtype)
        return os.path.join(os.path.dirname(fname0), new_basename)


class GGCMFileFortbinIono(GGCMFileFortbinMHD):  # pylint: disable=abstract-method
    """Binary files"""
    _detector = r"^\s*(.*)\.(iof)\.([0-9]{6}).b\s*$"
    _iono = True
    _grid_type = grid.Grid
    _def_fld_center = "Node"


# class FortbinIonoDataWrapper(FortbinDataWrapper):
#     @property
#     def shape(self):
#         ret = tuple([n - 1 for n in reversed(self.expected_shape)])
#         return ret

#     def __array__(self, *args, **kwargs):
#         arr = super(FortbinIonoDataWrapper, self).__array__(*args, **kwargs)
#         ndim = len(self.expected_shape)
#         return arr[[slice(None, -1)]*ndim]

##
## EOF
##
