"""VPIC binary file reader
"""

from __future__ import print_function
from collections import namedtuple
import os
import re
import struct
import sys

import numpy as np

import viscid
from viscid import amr_grid, coordinate, field, grid
from viscid import glob2, logger
from viscid.readers import vfile


class VPIC_Grid(grid.Grid):
    # TODO: add any _get_* methods to make retrieving fields more natural

    def _get_ex(self):
        return self['Electric Field X']

    def _get_ey(self):
        return self['Electric Field Y']

    def _get_ez(self):
        return self['Electric Field Z']

    def _get_e(self):
        ex, ey, ez = self['ex'], self['ey'], self['ez']
        opts = dict(_force_layout=self.force_vector_layout)
        return field.scalar_fields_to_vector([ex, ey, ez], name="E", **opts)

    def _get_dive(self):
        return self['Electric Field Divergence Error']

    def _get_bx(self):
        return self['Magnetic Field X']

    def _get_by(self):
        return self['Magnetic Field Y']

    def _get_bz(self):
        return self['Magnetic Field Z']

    def _get_b(self):
        bx, by, bz = self['bx'], self['by'], self['bz']
        opts = dict(_force_layout=self.force_vector_layout)
        return field.scalar_fields_to_vector([bx, by, bz], name="B", **opts)

    def _get_divb(self):
        return self['Magnetic Field Divergence Error']

    def _get_n_q(self):
        return self['Charge Density']

    # electron
    def _get_jx_e(self):
        return self['Current Density (ehydro) X']

    def _get_jy_e(self):
        return self['Current Density (ehydro) Y']

    def _get_jz_e(self):
        return self['Current Density (ehydro) Z']

    def _get_j_e(self):
        jx_e, jy_e, jz_e = self['jx_e'], self['jy_e'], self['jz_e']
        opts = dict(_force_layout=self.force_vector_layout)
        return field.scalar_fields_to_vector([jx_e, jy_e, jz_e], name="J_e", **opts)

    def _get_rhovx_e(self):
        return self['Momentum Density (ehydro) X']

    def _get_rhovy_e(self):
        return self['Momentum Density (ehydro) Y']

    def _get_rhovz_e(self):
        return self['Momentum Density (ehydro) Z']

    def _get_rhov_e(self):
        rhox_e, rhoy_e, rhoz_e = self['rhovx_e'], self['rhovy_e'], self['rhovz_e']
        opts = dict(_force_layout=self.force_vector_layout)
        return field.scalar_fields_to_vector([rhox_e, rhoy_e, rhoz_e], name="rhov_e", **opts)

    def _get_n_e(self):
        return self['Charge Density (ehydro)']

    def _get_ek_e(self):
        return self['Kinetic Energy Density (ehydro)']

    def _get_Pxx_e(self):
        return self['Stress Tensor (ehydro) XX']

    def _get_Pyy_e(self):
        return self['Stress Tensor (ehydro) YY']

    def _get_Pzz_e(self):
        return self['Stress Tensor (ehydro) ZZ']

    def _get_Pzx_e(self):
        return self['Stress Tensor (ehydro) ZX']

    def _get_Pxy_e(self):
        return self['Stress Tensor (ehydro) XY']

    def _get_Pyz_e(self):
        return self['Stress Tensor (ehydro) YZ']

    # ion
    def _get_jx_i(self):
        return self['Current Density (Hhydro) X']

    def _get_jy_i(self):
        return self['Current Density (Hhydro) Y']

    def _get_jz_i(self):
        return self['Current Density (Hhydro) Z']

    def _get_j_i(self):
        jx_i, jy_i, jz_i = self['jx_i'], self['jy_i'], self['jz_i']
        opts = dict(_force_layout=self.force_vector_layout)
        return field.scalar_fields_to_vector([jx_i, jy_i, jz_i], name="J_i", **opts)

    def _get_rhovx_i(self):
        return self['Momentum Density (Hhydro) X']

    def _get_rhovy_i(self):
        return self['Momentum Density (Hhydro) Y']

    def _get_rhovz_i(self):
        return self['Momentum Density (Hhydro) Z']

    def _get_rhov_i(self):
        rhox_i, rhoy_i, rhoz_i = self['rhovx_i'], self['rhovy_i'], self['rhovz_i']
        opts = dict(_force_layout=self.force_vector_layout)
        return field.scalar_fields_to_vector([rhox_i, rhoy_i, rhoz_i], name="rhov_i", **opts)

    def _get_n_i(self):
        return self['Charge Density (Hhydro)']

    def _get_ik_i(self):
        return self['Kinetic Energy Density (Hhydro)']

    def _get_Pxx_i(self):
        return self['Stress Tensor (Hhydro) XX']

    def _get_Pyy_i(self):
        return self['Stress Tensor (Hhydro) YY']

    def _get_Pzz_i(self):
        return self['Stress Tensor (Hhydro) ZZ']

    def _get_Pzx_i(self):
        return self['Stress Tensor (Hhydro) ZX']

    def _get_Pxy_i(self):
        return self['Stress Tensor (Hhydro) XY']

    def _get_Pyz_i(self):
        return self['Stress Tensor (Hhydro) YZ']


class _VPICGlobalFile(object):
    VPICFieldDescr = namedtuple("VPICFieldDescr",
                                ["name", "kind", "n_comps", "dtype", "off"])
    VPICFieldSet = namedtuple("VPICFieldSet",
                              ["dir", "basename", "descr"])

    _keywords = {
        "DATA_HEADER_SIZE": "i",
        "GRID_DELTA_T": "f",
        "GRID_CVAC": "f",
        "GRID_EPS0": "f",
        "GRID_EXTENTS_X": "f2",
        "GRID_EXTENTS_Y": "f2",
        "GRID_EXTENTS_Z": "f2",
        "GRID_DELTA_X": "f",
        "GRID_DELTA_Y": "f",
        "GRID_DELTA_Z": "f",
        "GRID_TOPOLOGY_X": "i",
        "GRID_TOPOLOGY_Y": "i",
        "GRID_TOPOLOGY_Z": "i",
        "FIELD_DATA_DIRECTORY": "s",
        "FIELD_DATA_BASE_FILENAME": "s",
    }

    def __init__(self, filename):
        self.kw = dict()

        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                words = line.split()

                funcname = "parse_" + words[0]
                if hasattr(self, funcname):
                    parse = getattr(self, funcname)
                    parse(words[1:], f)
                elif words[0] in self._keywords:
                    descr = self._keywords[words[0]]
                    if descr[0] == 'i':
                        values = [int(w) for w in words[1:]]
                    elif descr[0] == 'f':
                        values = [float(w) for w in words[1:]]
                    elif descr[0] == 's':
                        values = words[1:]
                    else:
                        raise NotImplementedError("unhandled descr {0}"
                                                  "".format(descr))
                    if len(descr) > 1:
                        n_values = int(descr[1:])
                        assert len(words[1:]) == n_values
                    else:
                        values, = values
                    self.kw[words[0]] = values
                else:
                    logger.error("Parsing problem: {0}".format(words))

        self._setup()

    @classmethod
    def _parseFieldDescr(cls, line, off):
        name, rest = line.split('"')[1:]
        words = rest.split()
        assert len(words) == 4
        kind = words[0]
        n_comps = int(words[1])

        basic_type = (words[2], int(words[3]))
        if basic_type == ("FLOATING_POINT", 4):
            dtype = np.float32
        elif basic_type == ("INTEGER", 2):
            dtype = np.int16
        else:
            raise NotImplementedError("unhandled type {0}".format(basic_type))

        return cls.VPICFieldDescr(name, kind, n_comps, dtype, off)


    def parse_VPIC_HEADER_VERSION(self, args, f):
        if args != ["1.0.0"]:
            raise NotImplementedError("Only know how to read version 1.0.0 files!")

    def parse_FIELD_DATA_VARIABLES(self, args, f):
        assert len(args) == 1
        n_field_vars = int(args[0])
        self.fields = self.VPICFieldSet(dir=self.kw["FIELD_DATA_DIRECTORY"],
                                        basename=self.kw["FIELD_DATA_BASE_FILENAME"],
                                        descr=[])
        off = 0
        for n in range(n_field_vars):
            line = next(f).strip()
            self.fields.descr.append(self._parseFieldDescr(line, off=off))
            off += self.fields.descr[-1].n_comps

    def parse_NUM_OUTPUT_SPECIES(self, args, f):
        assert len(args) == 1
        n_species = int(args[0])
        self.species = []
        for s in range(n_species):
            kw = dict()
            while True:
                line = next(f).strip()
                if line.startswith('#') or not line:
                    continue

                words = line.split()
                if words[0] == "HYDRO_DATA_VARIABLES":
                    assert len(words[1:]) == 1
                    n_field_vars = int(words[1])
                    fs = self.VPICFieldSet(dir=kw["SPECIES_DATA_DIRECTORY"],
                                           basename=kw["SPECIES_DATA_BASE_FILENAME"],
                                           descr=[])
                    off = 0
                    for n in range(n_field_vars):
                        line = next(f).strip()
                        fs.descr.append(self._parseFieldDescr(line, off))
                        off += fs.descr[-1].n_comps

                    # We're done with this species
                    self.species.append(fs)
                    break
                else:
                    assert len(words[1:]) == 1
                    kw[words[0]] = words[1]

    def _makeFieldDict(self):
        self._field_dict = dict()
        for fs in [self.fields] + self.species:
            for d in fs.descr:
                name = d.name
                if fs.basename != "fields":
                    name += " ({0})".format(fs.basename)
                self._field_dict[name] = (fs, d)

    def _setup(self):
        self._makeFieldDict()
        self.topology = np.array([self.kw["GRID_TOPOLOGY_{0}".format(comp)]
                                  for comp in 'XYZ'])
        self.n_procs = np.prod(self.topology)

        self.dx = np.array([self.kw["GRID_DELTA_{0}".format(comp)]
                            for comp in 'XYZ'])
        # low
        self.xl = np.array([self.kw["GRID_EXTENTS_{0}".format(comp)][0]
                            for comp in 'XYZ'])
        # high
        xh_local = np.array([self.kw["GRID_EXTENTS_{0}".format(comp)][1]
                             for comp in 'XYZ'])
        # local domain size (extent)
        self.extent = xh_local - self.xl
        self.ldims = (self.extent / self.dx + 0.5).astype(np.int_)
        self.gdims = self.ldims * self.topology

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

    def __str__(self):
        s = "VPICGlobalFile:\n"
        s += "fields:\n {0}\n".format(self.fields)
        s += "species:\n"
        for sp in self.species:
            s += "  {0}\n".format(sp)
        s += "kw:\n {0}\n".format(self.kw)
        s += "topology: {0}\n".format(self.topology)
        s += "n_procs: {0}\n".format(self.n_procs)
        return s


class VPIC_File(vfile.VFile):  # pylint: disable=abstract-method
    """An VPIC binary file reader"""
    _detector = r"^\s*(.*)\.(vpc)\s*$"
    _grid_type = VPIC_Grid

    _def_fld_center = "Cell"

    _fwrapper = None


    def __init__(self, fname, **kwargs):
        """
        Keyword Arguments:
            float_type_name (str): should be 'f4' or 'f8' if you know
                the data type of the file's data.
            var_type (str): either 'cons' or 'prim'
        """
        super(VPIC_File, self).__init__(fname, **kwargs)

    def load(self, fname):
        self.set_info('run', '???')
        super(VPIC_File, self).load(fname)

    def _parse(self):
        _gfile = _VPICGlobalFile(self.fname)

        dir_to_list = os.path.join(self.dirname, _gfile.fields.dir)
        tframe_dirs = [d for d in sorted(os.listdir(dir_to_list))]
        time_list = [int(s.split('.')[1]) for s in tframe_dirs]

        data_temporal = self._make_dataset(self, dset_type="temporal",
                                           name="VPIC_TemporalCollection")

        _last_amr_skeleton = None

        block_crds = []
        for k in range(_gfile.topology[2]):
            for j in range(_gfile.topology[1]):
                for i in range(_gfile.topology[0]):
                    idx = [i, j, k]
                    xl = [_gfile.xl[d] + _gfile.ldims[d] * idx[d] * _gfile.dx[d]
                          for d in range(3)]
                    xh = [_gfile.xl[d] + _gfile.ldims[d] * (idx[d] + 1) * _gfile.dx[d]
                          for d in range(3)]
                    arrs = [np.linspace(_xl, _xh, _n)
                            for _xl, _xh, _n in zip(xl, xh, _gfile.ldims)]
                    block_crds += [coordinate.arrays2crds(arrs)]

        file_wrapper_cls = VPIC_BinFileWrapper
        data_wrapper_cls = VPIC_DataWrapper

        for time in time_list:
            data_spatial = self._make_dataset(self,
                                              name='VPIC_SpatialCollection')
            data_spatial.time = time

            parent_node = self # FIXME: is this correct?

            for i, crds in enumerate(block_crds):
                _grid = self._make_grid(parent_node, name="<VPIC_Gridt {0}>".format(i))
                _grid.set_crds(crds)

                for fs in [_gfile.fields] + _gfile.species:
                    bin_fname = ("{0}/T.{1}/{2}.{3}.{4}"
                                 "".format(fs.dir, time, fs.basename, time, i))
                    # file_wrapper is a lazy proxy for the whole binary file...
                    # ie, something that knows how to seek etc.
                    bin_fname = os.path.join(self.dirname, bin_fname)
                    file_wrapper = file_wrapper_cls(bin_fname)

                    for descr in fs.descr:
                        fld_name = descr.name
                        if fs.basename != 'fields':
                            fld_name += " ({0})".format(fs.basename)

                        for icomp in range(descr.n_comps):
                            if descr.n_comps == 1:
                                _fld_name = fld_name
                            else:
                                if descr.n_comps == 3:
                                    comps = 'XYZ'
                                elif descr.n_comps == 6:
                                    comps = ['XX', 'YY', 'ZZ', 'YZ', 'ZX', 'XY']
                                else:
                                    comps = 'ABCDEFGHIJKLMN'
                                _fld_name = '{0} {1}'.format(fld_name, comps[icomp])
                            # data is a lazy proxy for an ndarray
                            # FIXME: the data type is in descr?
                            data = data_wrapper_cls(file_wrapper, _fld_name,
                                                    _gfile.ldims,
                                                    icomp + descr.off,
                                                    np.dtype(np.float32))
                            _fld = self._make_field(_grid, "Scalar", _fld_name,
                                                    crds, data, time=time,
                                                    center='node')
                            _grid.add_field(_fld)

                data_spatial.add(_grid)

            _amr_grid, is_amr = amr_grid.dataset_to_amr_grid(data_spatial,
                                                             _last_amr_skeleton)
            if is_amr:
                _last_amr_skeleton = _amr_grid.skeleton

            data_temporal.add(_amr_grid)

        self.add(data_temporal)
        self.activate(0)


class VPIC_BinFileWrapper(object):
    """A File-like object for interfacing with VPIC binary files"""
    _file = None

    filename = None

    def __init__(self, filename):
        self.filename = filename

    def __del__(self):
        self.close()

    def read_field(self, comp, slicex=slice(None), slicey=slice(None),
                   slicez=slice(None), dtype=np.float32):
        with self as _:
            f_in = self._file
            fmt_1 = struct.Struct('=5b')
            CHAR_BITS, sz_short_int, sz_int, sz_float, \
                sz_double = fmt_1.unpack(f_in.read(fmt_1.size))
            #logger.debug(CHAR_BITS, sz_short_int, sz_int, sz_float, sz_double)
            assert CHAR_BITS == 8 and sz_short_int == 2
            assert sz_int == 4 and sz_float == 4 and sz_double == 8

            fmt_2 = struct.Struct('=HIfd')
            cafe, deadbeef, float_1, double_1 = fmt_2.unpack(f_in.read(fmt_2.size))
            #logger.debug("{:x} {:x}".format(cafe, deadbeef))
            assert cafe == 0xcafe and deadbeef == 0xdeadbeef
            assert float_1 == 1 and double_1 == 1

            fmt_3 = struct.Struct('=6i10f')
            version, dump_type, step, nx_out, ny_out, nz_out, \
                dt, dx, dy, dz, x0, y0, z0, \
                cvac, eps0, damp = fmt_3.unpack(f_in.read(fmt_3.size))
            logger.debug('version {0} dump_type {1}'.format(version, dump_type))
            logger.debug('step {0} n_out {1}'.format(step, (nx_out, ny_out, nz_out)))
            logger.debug('dt {0} dx {1} x0 {2}'.format(dt, (dx, dy, dz), (x0, y0, z0)))
            logger.debug('cvac {0} eps0 {1} damp {2}'.format(cvac, eps0, damp))

            fmt_4 = struct.Struct('=3i1f5i')
            rank, nproc, sp_id, q_m, sz_data, ndim, ghost_size_x, \
               ghost_size_y, ghost_size_z = fmt_4.unpack(f_in.read(fmt_4.size))
            logger.debug("rank {0} nproc {1}".format(rank, nproc))
            logger.debug("sp_id {0} q_m {1}".format(sp_id, q_m))
            logger.debug("sz_data {0} ndim {1} ghost_size {2}"
                         "".format(sz_data, ndim, (ghost_size_x, ghost_size_y,
                                                   ghost_size_z)))

            dim = (ghost_size_x, ghost_size_y, ghost_size_z)
            # The material data are originally short ints
            # but are written as uint_32_t (of size same
            # as float32; see dump.cc:field_dump); this is
            # perhaps for convnience.
            # TODO: slice when reading?
            HEADER_SIZE = 123
            RECORD_SIZE = 4
            assert f_in.tell() == HEADER_SIZE

            f_in.seek(HEADER_SIZE + RECORD_SIZE * comp * np.prod(dim))
            data = np.fromfile(f_in, count=np.prod(dim), dtype=dtype)

            data = data.reshape(*dim, order='F')
            data = data[1:-1, 1:-1, 1:-1]
            data_sliced = data[slicez, slicey, slicex]

            return data_sliced

    def _read_file_header(self):
        """load up the file's meta data"""
        # TODO: move the header stuff here so we only do it once
        return None

    def read_header(self):
        if self._endian is None:
            with self as _:
                # just opening the file makes it read the header
                pass

    def open(self):
        if self._file is None:
            self._file = open(self.filename, 'rb')
            try:
                pass
                # TODO: trigger _read_file_header()
                # if self._endian is None:
                #     self._read_file_header()
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


class VPIC_DataWrapper(vfile.DataWrapper):
    file_wrapper = None
    filename = None
    fld_name = None
    expected_shape = None
    fld_number = None

    def __init__(self, file_wrapper, fld_name, expected_shape, fld_number,
                 dtype):
        """Lazy wrapper for a field in a Fortbin file

        Parameters:
            expected_shape (tuple): shape of data in the file (zyx)
        """
        super(VPIC_DataWrapper, self).__init__()
        self.file_wrapper = file_wrapper
        self.filename = file_wrapper.filename
        self.fld_name = fld_name
        self.expected_shape = expected_shape
        self.fld_number = fld_number
        self._dtype = dtype

    @property
    def shape(self):
        """
        Returns:
            zyx shape since that's the shape __array__ returns
        """
        return self.expected_shape

    @property
    def dtype(self):
        return self._dtype

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


def _main():
    import viscid
    from viscid.plot import vpyplot as vlt
    logger.setLevel(viscid.logging.DEBUG)

    f = viscid.load_file('~/Downloads/sample_data/global.vpc')
    vlt.plot(f['bx'], show=True)

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
