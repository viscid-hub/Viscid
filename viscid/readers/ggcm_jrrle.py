from __future__ import print_function
import os
import re

import numpy as np

from viscid import field
from viscid import grid
from viscid.readers import vfile
from viscid.readers import openggcm
from viscid.readers._fortfile_wrapper import FortranFile
from viscid.readers import _jrrle
from viscid.compat import OrderedDict


class GGCMFileJrrleMHD(openggcm.GGCMFileFortran):  # pylint: disable=abstract-method
    """Jimmy's run length encoding files"""
    _detector = r"^\s*(.*)\.(p[xyz]_[0-9]+|3df)" \
                r"\.([0-9]{6})\s*$"

    _file_wrapper = None
    _data_item_templates = None
    _def_fld_center = "Cell"

    def __init__(self, filename, vfilebucket=None, **kwargs):
        super(GGCMFileJrrleMHD, self).__init__(filename, vfilebucket, **kwargs)

    def _parse_file(self, filename):
        # we do minimal file parsing here for performance. we just
        # make data wrappers from the templates we got from the first
        # file in the group, and package them up into grids

        # find the time from the first field's meta data
        self._file_wrapper = JrrleFileWrapper(filename)

        with self._file_wrapper as f:
            _, meta = f.inquire_next()
            time = float(str(meta['timestr']).split('=')[1].split()[0])

        _grid = self._make_grid("<JrrleGrid>", **self._grid_opts)
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
            data_wrapper = JrrleDataWrapper
        else:
            data_wrapper = JrrleDataWrapper

        for item in templates:
            data = data_wrapper(self._file_wrapper, item['fld_name'],
                                item['shape'])
            fld = field.wrap_field("Scalar", item['fld_name'], self._crds,
                                   data, center=self._def_fld_center,
                                   time=time)
            _grid.add_field(fld)
        return _grid

    @staticmethod
    def _make_template(filename):
        """read meta data for all fields in a file to get
        a list of field names and shapes, all the required info
        to make a JrrleDataWrapper
        """
        with JrrleFileWrapper(filename) as f:
            f.inquire_all_fields()
            template = []
            for fld_name, meta in f.fields_seen.items():
                d = dict(fld_name=fld_name,
                         shape=meta['dims'])
                template.append(d)
        return template

    @classmethod
    def collective_name_from_group(cls, fnames):
        fname0 = fnames[0]
        basename = os.path.basename(fname0)
        run = re.match(cls._detector, basename).group(1)
        fldtype = re.match(cls._detector, basename).group(2)
        new_basename = "{0}.{1}".format(run, fldtype)
        return os.path.join(os.path.dirname(fname0), new_basename)


class GGCMFileJrrleIono(GGCMFileJrrleMHD):  # pylint: disable=abstract-method
    """Jimmy's run length encoding files"""
    _detector = r"^\s*(.*)\.(iof)\.([0-9]{6})\s*$"
    _iono = True
    _grid_type = grid.Grid
    _def_fld_center = "Node"

class JrrleFileWrapper(FortranFile):
    """Interface for actually opening / reading a jrrle file"""
    _read_func = [_jrrle.read_jrrle1d, _jrrle.read_jrrle2d,
                  _jrrle.read_jrrle3d]

    fields_seen = None
    seen_all_fields = None

    def __init__(self, filename):
        self.fields_seen = OrderedDict()
        self.seen_all_fields = False
        super(JrrleFileWrapper, self).__init__(filename)

    def read_field(self, fld_name, ndim):
        """Read a field given a seekable location

        Parameters:
            loc(int): position in file we can seek to
            ndim(int): dimensionality of field

        Returns:
            tuple (field name, dict of meta data, array)
        """
        meta = self.inquire(fld_name)
        arr = np.empty(meta['dims'], dtype='float32', order='F')
        self._read_func[ndim - 1](self.unit, arr, fld_name)
        return meta, arr

    def inquire_all_fields(self, reinquire=False):
        if reinquire:
            self.seen_all_fields = False
            self.fields_seen = OrderedDict()

        if self.seen_all_fields:
            return

        self.rewind()
        while not self.seen_all_fields:
            self.inquire_next()
            # last_seen, meta = self.inquire_next()
            # if meta is not None:
            #     print(last_seen, "lives at", meta["file_position"])
            self.advance_one_line()

    def inquire(self, fld_name):
        try:
            meta = self.fields_seen[fld_name]
            self.seek(meta['file_position'])
            return meta
        except KeyError:
            last_added = next(reversed(self.fields_seen))
            self.seek(self.fields_seen[last_added]['file_position'])
            self.advance_one_line()

            while not self.seen_all_fields:
                found_fld_name, meta = self.inquire_next()
                if found_fld_name == fld_name:
                    return meta
                self.advance_one_line()

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

        varname = np.array(" "*80, dtype="S80")
        tstring = np.array(" "*80, dtype="S80")
        found_field, ndim, nx, ny, nz, it = _jrrle.inquire_next(self._unit,
                                                                varname,
                                                                tstring)
        if not found_field:
            self.seen_all_fields = True
            return None, None

        vname = (str(varname)).strip()

        if vname in self.fields_seen:
            meta = self.fields_seen[vname]
        else:
            dims = tuple(x for x in (nx, ny, nz) if x > 0)

            meta = dict(timestr=tstring,
                        inttime=it,
                        ndim=ndim,
                        dims=dims,
                        file_position=self.tell())
            self.fields_seen[vname] = meta

        return vname, meta


class JrrleDataWrapper(vfile.DataWrapper):
    """  """
    file_wrapper = None
    filename = None
    fld_name = None
    expected_shape = None

    def __init__(self, file_wrapper, fld_name, expected_shape):
        """Lazy wrapper for a field in a jrrle file

        Parameters:
            expected_shape (tuple): shape of data in the file (xyz)
        """
        super(JrrleDataWrapper, self).__init__()
        self.file_wrapper = file_wrapper
        self.filename = file_wrapper.filename
        self.fld_name = fld_name
        # translate to zyx
        self.expected_shape = expected_shape

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
            ndim = len(self.expected_shape)
            # fld_name, meta, arr = f.read_field_at(self.loc, ndim)
            meta, arr = f.read_field(self.fld_name, ndim)
            arr = np.array(arr.flatten(order='F').reshape(meta['dims'][::-1]),
                           order='C')

        # meta's dims are xyz (from file), but ex
        if meta['dims'] != self.expected_shape:
            raise RuntimeError("Field '{0}' from file '{1}' has shape {2} "
                               "instead of {3}".format(self.fld_name,
                               self.filename, meta['dims'],
                               self.expected_shape))
        return arr.astype(self.dtype)

    def read_direct(self, *args, **kwargs):
        return self.__array__()

    def len(self):
        return self.shape[0]

    def __getitem__(self, item):
        return self.__array__().__getitem__(item)


# class JrrleIonoDataWrapper(JrrleDataWrapper):
#     @property
#     def shape(self):
#         ret = tuple([n - 1 for n in reversed(self.expected_shape)])
#         return ret

#     def __array__(self, *args, **kwargs):
#         arr = super(JrrleIonoDataWrapper, self).__array__(*args, **kwargs)
#         ndim = len(self.expected_shape)
#         return arr[[slice(None, -1)]*ndim]

##
## EOF
##
