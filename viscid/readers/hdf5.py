from __future__ import print_function
import os

import numpy as np

from viscid import logger
from viscid.readers import vfile

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    logger.warn("h5py library not found, no hdf5 support.")

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
        assert HAS_H5PY
        super(H5pyDataWrapper, self).__init__()
        self.fname = fname
        self.loc = loc
        self.comp_dim = comp_dim
        self.comp_idx = comp_idx
        self.transpose = transpose

    def _read_info(self):
        try:
            with h5py.File(self.fname, 'r') as f:
                dset = f[self.loc]
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

    def len(self):
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
            f[self.loc].read_direct(fill_arr, source_sel=source_sel,
                                    **kwargs)
            if self.transpose:
                arr[...] = fill_arr.T

    def __getitem__(self, item):
        item = self._inject_comp_slice(item)
        with h5py.File(self.fname, 'r') as f:
            arr = f[self.loc][item]
            if self.transpose:
                return np.transpose(arr)
            else:
                return arr


class FileLazyHDF5(vfile.VFile):
    """ This is for lazy wrapping an h5 file referred to by an xdmf file,
    or anything else where you want to have a single file instance for a single
    hdf5 file, but you don't want any parsing because you already know what's
    going to be in the file. """
    _detector = None

    def __init__(self, fname, **kwargs):
        assert HAS_H5PY
        super(FileLazyHDF5, self).__init__(fname, **kwargs)

    def _parse(self):
        pass

    def get_data(self, handle):
        return H5pyDataWrapper(self.fname, handle)


class FileHDF5(vfile.VFile): #pylint: disable=R0922
    """ this is an abstract-ish class from which other types of hdf5 readers
    should derrive """
    _detector = r".*\.h5\s*$"

    SAVE_ONLY = True

    _CRDS_GROUP = "/crds"
    _FLD_GROUPS = {"node": "/flds_nc",
                   "cell": "/flds_cc",
                   "face": "/flds_fc",
                   "edge": "/flds_ec"}

    _XDMF_TEMPLATE_BEGIN = \
"""<?xml version='1.0' ?>
<Xdmf xmlns:xi='http://www.w3.org/2001/XInclude' Version='2.0'>
<Domain>
<Grid GridType="Collection" CollectionType="Spatial">
  <Time Type="Single" Value="{time}" />
"""
    _XDMF_TEMPLATE_RECTILINEAR_GRID_BEGIN = \
"""  <Grid Name="{grid_name}" GridType="Uniform">
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
    _XDMF_TEMPLATE_ATTRIBUTE = \
"""    <Attribute Name="{fld_name}" AttributeType="{fld_type}" Center="{center}">
      <DataItem Dimensions="{fld_dims}" NumberType="{dtype}" Precision="{precision}" Format="HDF">
        {h5fname}:{fld_loc}
      </DataItem>
    </Attribute>
"""
    _XDMF_TEMPLATE_GRID_END = \
"""  </Grid>
"""
    _XDMF_TEMPLATE_END = \
"""</Grid>
</Domain>
</Xdmf>
"""

    def __init__(self, fname, **kwargs):
        assert(HAS_H5PY)
        super(FileHDF5, self).__init__(fname, **kwargs)

    def _parse(self):
        raise NotImplementedError("Please load using the xdmf file")

    def save(self, fname=None, **kwargs):
        """ save an instance of VFile, fname defaults to the name
        of the file object as read """
        if fname is None:
            fname = self.fname
        flds = list(self.iter_fields)
        self.save_fields(fname, flds)

    @classmethod
    def save_fields(cls, fname, flds, **kwargs):
        """ save some fields using the format given by the class """
        # FIXME: this is only good for writing cartesian rectilnear flds
        # FIXME: axes are renamed if flds[0] is 1D or 2D
        assert len(flds) > 0
        fname = os.path.expanduser(os.path.expandvars(fname))

        # FIXME: all coordinates are saved as non-uniform, the proper
        #        way to do this is to have let coordinate format its own
        #        hdf5 / xdmf / numpy binary output
        clist = flds[0].crds.get_clist(full_arrays=True)
        crd_arrs = [np.array([0.0])] * 3
        crd_names = ["z", "y", "x"]
        for i, c in enumerate(clist):
            crd_arrs[i] = c[1]
        crd_shape = [len(arr) for arr in crd_arrs]
        time = flds[0].time

        # write arrays to the hdf5 file
        with h5py.File(fname, 'w') as f:
            for axis_name, arr in zip(crd_names, crd_arrs):
                loc = cls._CRDS_GROUP + '/' + axis_name
                f[loc] = arr

            for fld in flds:
                loc = cls._FLD_GROUPS[fld.center.lower()] + '/' + fld.name
                f[loc] = fld.data

        # now write an xdmf file
        xdmf_fname = os.path.splitext(fname)[0] + ".xdmf"
        relh5fname = "./" + os.path.basename(fname)
        with open(xdmf_fname, 'w') as f:
            xloc = cls._CRDS_GROUP + '/' + crd_names[2]
            yloc = cls._CRDS_GROUP + '/' + crd_names[1]
            zloc = cls._CRDS_GROUP + '/' + crd_names[0]
            dim_str = " ".join([str(l) for l in crd_shape])
            f.write(cls._XDMF_TEMPLATE_BEGIN.format(time=time))
            s = cls._XDMF_TEMPLATE_RECTILINEAR_GRID_BEGIN.format(
                grid_name="vgrid", crd_dims=dim_str, h5fname=relh5fname,
                xdim=crd_shape[2], ydim=crd_shape[1], zdim=crd_shape[0],
                xloc=xloc, yloc=yloc, zloc=zloc)
            f.write(s)

            for fld in flds:
                dt = fld.dtype.name.rstrip("0123456789").title()
                precision = fld.dtype.itemsize
                fld_dim_str = " ".join([str(l) for l in fld.shape])
                loc = cls._FLD_GROUPS[fld.center.lower()] + '/' + fld.name
                s = cls._XDMF_TEMPLATE_ATTRIBUTE.format(
                    fld_name=fld.name,
                    fld_type=fld.fldtype, center=fld.center.title(),
                    dtype=dt, precision=precision, fld_dims=fld_dim_str,
                    h5fname=relh5fname, fld_loc=loc)
                f.write(s)

            f.write(cls._XDMF_TEMPLATE_GRID_END)
            f.write(cls._XDMF_TEMPLATE_END)

##
## EOF
##
