from __future__ import print_function
import os
import logging

import numpy as np
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    logging.warn("h5py library not found, no hdf5 support.")

from . import vfile

class H5pyDataWrapper(vfile.DataWrapper):
    """  """
    fname = None
    loc = None

    _shape = None
    _dtype = None

    def __init__(self, fname, loc):
        super(H5pyDataWrapper, self).__init__()
        self.fname = fname
        self.loc = loc

    def _get_info(self):
        # this takes super long when reading 3 hrs worth of ggcm data
        # over sshfs
        # import pdb; pdb.set_trace()
        try:
            with h5py.File(self.fname, 'r') as f:
                dset = f[self.loc]
                self._shape = dset.shape
                self._dtype = dset.dtype
        except IOError as e:
            logging.error("Problem opening hdf5 file, '{0}'".format(self.fname))
            raise e

    @property
    def shape(self):
        """ only ask for this if you really need it; can be a speed problem
        for large temporal datasets over sshfs """
        if self._shape is None:
            self._get_info()
        return self._shape

    @property
    def dtype(self):
        """ only ask for this if you really need it; can be a speed problem
        for large temporal datasets over sshfs """
        if self._dtype is None:
            self._get_info()
        return self._dtype

    def wrap_func(self, func_name, *args, **kwargs):
        with h5py.File(self.fname, 'r') as f:
            return getattr(f[self.loc], func_name)(*args, **kwargs)

    def __array__(self, *args, **kwargs):
        return self.wrap_func("__array__", *args, **kwargs)

    def read_direct(self, *args, **kwargs):
        return self.wrap_func("read_direct", *args, **kwargs)

    def len(self):
        return self.wrap_func("len")

    def __getitem__(self, item):
        return self.wrap_func("__getitem__", item)


class FileLazyHDF5(vfile.VFile):
    """ This is for lazy wrapping an h5 file referred to by an xdmf file,
    or anything else where you want to have a single file instance for a single
    hdf5 file, but you don't want any parsing because you already know what's
    going to be in the file. """
    _detector = None

    def __init__(self, fname, **kwargs):
        assert(HAS_H5PY)
        super(FileLazyHDF5, self).__init__(fname, **kwargs)

    def _parse(self):
        pass

    def get_data(self, handle):
        return H5pyDataWrapper(self.fname, handle)


class FileHDF5(vfile.VFile): #pylint: disable=R0922
    """ this is an abstract-ish class from which other types of hdf5 readers
    should derrive """
    _detector = r".*\.h5\s*$"

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
        assert(len(flds) > 0)
        clist = flds[0].crds.get_clist()
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
        with open(xdmf_fname, 'w') as f:
            xloc = cls._CRDS_GROUP + '/' + crd_names[2]
            yloc = cls._CRDS_GROUP + '/' + crd_names[1]
            zloc = cls._CRDS_GROUP + '/' + crd_names[0]
            dim_str = " ".join([str(l) for l in crd_shape])
            f.write(cls._XDMF_TEMPLATE_BEGIN.format(time=time))
            f.write(cls._XDMF_TEMPLATE_RECTILINEAR_GRID_BEGIN.format(
                    grid_name="vgrid", crd_dims=dim_str, h5fname=fname,
                    xdim=crd_shape[2], ydim=crd_shape[1], zdim=crd_shape[0],
                    xloc=xloc, yloc=yloc, zloc=zloc))

            for fld in flds:
                dt = fld.dtype.name.rstrip("0123456789").title()
                precision = fld.dtype.itemsize
                fld_dim_str = " ".join([str(l) for l in fld.shape])
                loc = cls._FLD_GROUPS[fld.center.lower()] + '/' + fld.name
                f.write(cls._XDMF_TEMPLATE_ATTRIBUTE.format(fld_name=fld.name,
                        fld_type=fld.type, center=fld.center.title(),
                        dtype=dt, precision=precision, fld_dims=fld_dim_str,
                        h5fname=fname, fld_loc=loc))

            f.write(cls._XDMF_TEMPLATE_GRID_END)
            f.write(cls._XDMF_TEMPLATE_END)

##
## EOF
##
