from __future__ import print_function
import os

import numpy as np

import viscid
from viscid.compat import OrderedDict
from viscid import logger
from viscid.readers import vfile


try:
    import h5py
except ImportError as e:
    h5py = viscid.UnimportedModule(e)


class H5pyDataWrapper(vfile.DataWrapper):
    """  """
    _hypersliceable = True  # can read slices from disk

    fname = None
    loc = None
    comp_dim = None
    comp_idx = None
    transpose = None

    _shape = None
    _dtype = None

    def __init__(self, fname, loc, comp_dim=None, comp_idx=None,
                 transpose=False):
        super(H5pyDataWrapper, self).__init__()
        self.fname = fname
        self.loc = loc
        self.comp_dim = comp_dim
        self.comp_idx = comp_idx
        self.transpose = transpose

    def _read_info(self):
        try:
            with h5py.File(self.fname, 'r') as f:
                dset = self._resolve_loc(f)
                self._shape = list(dset.shape)
                if self.comp_dim is not None:
                    self._shape.pop(self.comp_dim)
                self._shape = tuple(self._shape)
                self._dtype = dset.dtype
        except IOError:
            logger.error("Problem opening hdf5 file, '%s'", self.fname)
            raise

    @property
    def shape(self):
        """ only ask for this if you really need it; can be a speed problem
        for large temporal datasets over sshfs """
        if self._shape is None:
            self._read_info()
        if self.transpose:
            return self._shape[::-1]
        else:
            return self._shape

    @property
    def dtype(self):
        """ only ask for this if you really need it; can be a speed problem
        for large temporal datasets over sshfs """
        if self._dtype is None:
            self._read_info()
        return self._dtype

    def __len__(self):
        if self.transpose:
            return self.shape[-1]
        else:
            return self.shape[0]

    def __array__(self, *args, **kwargs):
        arr = np.empty(self.shape, dtype=self.dtype)
        self.read_direct(arr)
        return arr

    def _inject_comp_slice(self, slc):
        if self.comp_dim is not None:
            new_slc = []
            if slc is not None:
                try:
                    new_slc = list(slc)
                except TypeError:
                    new_slc = [slc]
                new_slc += [slice(None)] * (len(self.shape) - len(new_slc))
            else:
                new_slc = [slice(None)] * len(self.shape)

            if self.comp_dim < 0:
                self.comp_dim += len(self.shape) + 1
            if self.comp_dim < 0:
                raise ValueError("comp_dim can't be < -len(self.shape)")

            new_slc.insert(self.comp_dim, self.comp_idx)
            slc = tuple(new_slc)
        return slc

    def read_direct(self, arr, **kwargs):
        source_sel = kwargs.pop("source_sel", None)
        source_sel = self._inject_comp_slice(source_sel)
        with h5py.File(self.fname, 'r') as f:
            fill_arr = arr
            if self.transpose:
                # FIXME: the temp array here isn't pretty, but transposing
                # the array is kind of a hack anyway. it is fixed by the
                # ability for fields to specify their xyz/zyx order in the
                # xyz branch, but that branch isn't fully tested yet
                fill_arr = np.empty(arr.shape[::-1], dtype=arr.dtype)
            self._resolve_loc(f).read_direct(fill_arr, source_sel=source_sel,
                                             **kwargs)
            if self.transpose:
                arr[...] = fill_arr.T

    def __getitem__(self, item):
        item = self._inject_comp_slice(item)
        with h5py.File(self.fname, 'r') as f:
            arr = self._resolve_loc(f)[item]
            if self.transpose:
                return np.transpose(arr)
            else:
                return arr

    def _resolve_loc(self, open_file):
        ret, self.loc = viscid.resolve_path(open_file, self.loc, first=True)
        return ret


class FileLazyHDF5(vfile.VFile):
    """ This is for lazy wrapping an h5 file referred to by an xdmf file,
    or anything else where you want to have a single file instance for a single
    hdf5 file, but you don't want any parsing because you already know what's
    going to be in the file. """
    _detector = None

    def __init__(self, fname, **kwargs):
        super(FileLazyHDF5, self).__init__(fname, **kwargs)

    def _parse(self):
        pass

    def get_data(self, handle):
        return H5pyDataWrapper(self.fname, handle)

    def resolve_path(self, path, first=False):
        with h5py.File(self.fname, 'r') as f:
            return viscid.resolve_path(f, path, first=first)

    def find_items(self, item):
        return self.resolve_path(item)[0]

    def find_item(self, item):
        return self.resolve_path(item, first=True)[0]


class FileHDF5(vfile.VFile):
    """ this is an abstract-ish class from which other types of hdf5 readers
    should derrive """
    _detector = r".*\.h5\s*$"

    SAVE_ONLY = True

    _CRDS_GROUP = "/crds"
    _FLD_GROUPS = {"node": "/flds_nc",
                   "cell": "/flds_cc",
                   "face": "/flds_fc",
                   "edge": "/flds_ec"}

    _XDMF_TEMPLATE_BEGIN = """<?xml version='1.0' ?>
<Xdmf xmlns:xi='http://www.w3.org/2001/XInclude' Version='2.0'>
<Domain>
<Grid GridType="Collection" CollectionType="Spatial">
  <Time Type="Single" Value="{time}" />
"""
    _XDMF_TEMPLATE_RECTILINEAR_GRID_BEGIN = """
  <Grid Name="{grid_name}" GridType="Uniform">
    <Topology TopologyType="3DRectMesh" Dimensions="{crd_dims}"/>
    <Geometry GeometryType="VXVYVZ">
    <DataItem Name="VX" DataType="Float" Dimensions="{xdim}" Format="HDF">
       {h5fname}:{xloc}
    </DataItem>
    <DataItem Name="VY" DataType="Float" Dimensions="{ydim}" Format="HDF">
       {h5fname}:{yloc}
    </DataItem>
    <DataItem Name="VZ" DataType="Float" Dimensions="{zdim}" Format="HDF">
       {h5fname}:{zloc}
    </DataItem>
    </Geometry>
"""
    _XDMF_TEMPLATE_ATTRIBUTE = """
    <Attribute Name="{fld_name}" AttributeType="{fld_type}" Center="{center}">
      <DataItem Dimensions="{fld_dims}" NumberType="{dtype}"
                Precision="{precision}" Format="HDF">
        {h5fname}:{fld_loc}
      </DataItem>
    </Attribute>
"""

    _XDMF_INFO_TEMPLATE = '<Information Name="{name}" Value="{value}" />\n'

    _XDMF_TEMPLATE_GRID_END = """  </Grid>
"""
    _XDMF_TEMPLATE_END = """</Grid>
</Domain>
</Xdmf>
"""

    def __init__(self, fname, **kwargs):
        super(FileHDF5, self).__init__(fname, **kwargs)

    def _parse(self):
        raise NotImplementedError("Please load using the xdmf file")

    def save(self, fname=None, **kwargs):
        """ save an instance of VFile, fname defaults to the name
        of the file object as read """
        if fname is None:
            fname = self.fname
        self.save_fields(fname, self.field_dict())

    @classmethod
    def save_fields(cls, fname, flds, complevel=0, compression='gzip',
                    compression_opts=None, **kwargs):
        """ save some fields using the format given by the class """
        # FIXME: this is only good for writing cartesian rectilnear flds
        # FIXME: axes are renamed if flds[0] is 1D or 2D
        assert len(flds) > 0
        fname = os.path.expanduser(os.path.expandvars(fname))

        if complevel and compression == 'gzip' and compression_opts is None:
            compression_opts = complevel
        # TODO: what if compression != 'gzip'
        do_compression = compression_opts is not None

        if isinstance(flds, list):
            if isinstance(flds[0], (list, tuple)):
                flds = OrderedDict(flds)
            else:
                flds = OrderedDict([(fld.name, fld) for fld in flds])

        # FIXME: all coordinates are saved as non-uniform, the proper
        #        way to do this is to have let coordinate format its own
        #        hdf5 / xdmf / numpy binary output
        fld0 = next(iter(flds.values()))
        clist = fld0.crds.get_clist(full_arrays=True)
        crd_arrs = [np.array([0.0])] * 3
        crd_names = ["x", "y", "z"]
        for i, c in enumerate(clist):
            crd_arrs[i] = c[1]
        crd_shape = [len(arr) for arr in crd_arrs]
        time = fld0.time

        # write arrays to the hdf5 file
        with h5py.File(fname, 'w') as f:
            for axis_name, arr in zip(crd_names, crd_arrs):
                loc = cls._CRDS_GROUP + '/' + axis_name
                if do_compression:
                    f.create_dataset(loc, data=arr, compression=compression,
                                     compression_opts=compression_opts)
                else:
                    f[loc] = arr

            for name, fld in flds.items():
                loc = cls._FLD_GROUPS[fld.center.lower()] + '/' + name
                # xdmf files use kji ordering
                if do_compression:
                    f.create_dataset(loc, data=fld.data.T, compression=compression,
                                     compression_opts=compression_opts)
                else:
                    f[loc] = fld.data.T

            # big bad openggcm time_str hack to put basetime into hdf5 file
            for fld in flds.values():
                try:
                    tfmt = "%Y:%m:%d:%H:%M:%S.%f"
                    sec_td = viscid.as_timedelta64(fld.time, 's')
                    dtime = viscid.as_datetime(fld.basetime + sec_td).strftime(tfmt)
                    epoch = viscid.readers.openggcm.GGCM_EPOCH
                    ts = viscid.as_timedelta(fld.basetime - epoch).total_seconds()
                    ts += fld.time
                    timestr = "time= {0} {1:.16e} {2} 300c".format(fld.time, ts, dtime)
                    f.create_group('openggcm')
                    f['openggcm'].attrs['time_str'] = np.string_(timestr)
                    break
                except viscid.NoBasetimeError:
                    pass

        # now write an xdmf file
        xdmf_fname = os.path.splitext(fname)[0] + ".xdmf"
        relh5fname = "./" + os.path.basename(fname)
        with open(xdmf_fname, 'w') as f:
            xloc = cls._CRDS_GROUP + '/' + crd_names[0]
            yloc = cls._CRDS_GROUP + '/' + crd_names[1]
            zloc = cls._CRDS_GROUP + '/' + crd_names[2]
            dim_str = " ".join([str(l) for l in crd_shape][::-1])
            f.write(cls._XDMF_TEMPLATE_BEGIN.format(time=time))
            s = cls._XDMF_TEMPLATE_RECTILINEAR_GRID_BEGIN.format(
                grid_name="vgrid", crd_dims=dim_str, h5fname=relh5fname,
                xdim=crd_shape[0], ydim=crd_shape[1], zdim=crd_shape[2],
                xloc=xloc, yloc=yloc, zloc=zloc)
            f.write(s)

            for fld in flds.values():
                _crd_system = viscid.as_crd_system(fld, None)
                if _crd_system:
                    f.write(cls._XDMF_INFO_TEMPLATE.format(name="crd_system",
                                                           value=_crd_system))
                    break

            for name, fld in flds.items():
                fld = fld.as_flat().T
                dt = fld.dtype.name.rstrip("0123456789").title()
                precision = fld.dtype.itemsize
                fld_dim_str = " ".join([str(l) for l in fld.shape])
                loc = cls._FLD_GROUPS[fld.center.lower()] + '/' + name
                s = cls._XDMF_TEMPLATE_ATTRIBUTE.format(
                    fld_name=name,
                    fld_type=fld.fldtype, center=fld.center.title(),
                    dtype=dt, precision=precision, fld_dims=fld_dim_str,
                    h5fname=relh5fname, fld_loc=loc)
                f.write(s)

            f.write(cls._XDMF_TEMPLATE_GRID_END)
            f.write(cls._XDMF_TEMPLATE_END)

##
## EOF
##
