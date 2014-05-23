# pylint: disable=too-many-lines
"""Fields are the basis of Viscid's data abstration

Fields belong in grids, or by themselves as a result of a calculation.
They can belong to a :class:`Grid` as the result of a file load, or
by themselves as the result of a calculation. This module has some
convenience functions for creating fields similar to `Numpy`.
"""

from __future__ import print_function
from six import string_types
import warnings
import logging
from inspect import isclass

import numpy as np

from viscid import coordinate
from viscid import vutil

LAYOUT_DEFAULT = "none"  # do not translate
LAYOUT_INTERLACED = "interlaced"
LAYOUT_FLAT = "flat"
LAYOUT_SCALAR = "scalar"
LAYOUT_OTHER = "other"

def empty(typ, name, crds, nr_comps=0, layout=LAYOUT_FLAT, center="Cell",
          dtype="float64", **kwargs):
    """Analogous to `numpy.empty` (uninitialized array)

    Parameters:
        typ (str): 'Scaler' / 'Vector'
        name (str): a way to refer to the field programatically
        crds (Coordinates): coordinates that describe the shape / grid
            of the field
        nr_comps (int, optional): for vector fields, nr of components
        layout (str, optional): how data is stored, is in "flat" or
            "interlaced" (interlaced == AOS)
        center (str, optional): cell or node, there really isn't
            support for edge / face yet
        dtype (optional): some way to describe numpy dtype of data
        kwargs: passed through to Field constructor
    """
    if center.lower() == "cell":
        sshape = crds.shape_cc
    elif center.lower() == "node":
        sshape = crds.shape_nc
    else:
        sshape = crds.shape_nc

    if ScalarField.istype(typ):
        shape = sshape
    else:
        if layout.lower() == LAYOUT_INTERLACED:
            shape = sshape + [nr_comps]
        else:
            shape = [nr_comps] + sshape

    dat = np.empty(shape, dtype=dtype)

    return wrap_field(typ, name, crds, dat, center=center, **kwargs)

def empty_like(name, fld, **kwargs):
    """Analogous to `numpy.empty_like`

    Makes a new, unitilialized :class:`Field`. Copies as much meta data
    as it can from `fld`.

    Parameters:
        name: name for this field
        fld: field to get coordinates / metadata from
        kwargs: passed through to :class:`Field` constructor

    Returns:
        new uninitialized :class:`Field`
    """
    dat = np.empty(fld.shape, dtype=fld.dtype)
    c = fld.center
    t = fld.time
    return wrap_field(fld.type, name, fld.crds, dat, center=c, time=t, **kwargs)

def zeros_like(name, fld, **kwargs):
    """Analogous to `numpy.zeros_like`

    Returns:
        new :class:`Field` initialized to 0

    See Also: :meth:`empty_like`
    """
    dat = np.zeros(fld.shape, dtype=fld.dtype)
    c = fld.center
    t = fld.time
    return wrap_field(fld.type, name, fld.crds, dat, center=c, time=t, **kwargs)

def ones_like(name, fld, **kwargs):
    """Analogous to `numpy.ones_like`

    Returns:
        new :class:`Field` initialized to 1

    See Also: :meth:`empty_like`
    """
    dat = np.ones(fld.shape, dtype=fld.dtype)
    c = fld.center
    t = fld.time
    return wrap_field(fld.type, name, fld.crds, dat, center=c, time=t, **kwargs)

def scalar_fields_to_vector(name, fldlist, **kwargs):
    """Convert scaler fields to a vector field

    Parameters:
        name (str): name for the vector field
        fldlist: list of :class:`ScalarField`
        kwargs: passed to :class:`VectorField` constructor

    Returns:
        A new :class:`VectorField`.
    """
    if not name:
        name = fldlist[0].name
    center = fldlist[0].center
    crds = fldlist[0].crds
    time = fldlist[0].time
    # shape = fldlist[0].data.shape

    vfield = VectorField(name, crds, fldlist, center=center, time=time,
                         info=fldlist[0].info, **kwargs)
    return vfield

def field_type(typ):
    """Lookup a Field type

    The magic lookup happens when typ is a string, if typ is a class
    then just return the class for convenience.

    Parameters:
        typ: python class object or string describing a field type in
            some way

    Returns:
        a :class:`Field` subclass
    """
    if isclass(typ) and issubclass(typ, Field):
        return typ
    else:
        for cls in vutil.subclass_spider(Field):
            if cls.istype(typ):
                return cls
    logging.warn("Field type {0} not understood".format(typ))
    return None

def wrap_field(typ, name, crds, data, **kwargs):
    """Convenience script for wrapping ndarrays

    Parameters:
        typ (str): 'Scaler' / 'Vector'
        name (str): a way to refer to the field programatically
        crds (Coordinates): coordinates that describe the shape / grid
            of the field
        data: Some data container, most likely a ``numpy.ndarray``
        kwargs: passed through to :class:`Field` constructor

    Returns:
        A :class:`Field` instance.
    """
    #
    #len(clist), clist[0][0], len(clist[0][1]), type)
    cls = field_type(typ)
    if cls is not None:
        return cls(name, crds, data, **kwargs)
    else:
        raise NotImplementedError("can not decipher field")

def rewrap_field(fld):
    return type(fld)(fld.name, fld.crds, fld.data, center=fld.center,
                     forget_source=True, _copy=True)

class Field(object):
    _TYPE = "none"
    _CENTERING = ['node', 'cell', 'grid', 'face', 'edge']

    # set on __init__
    # NOTE: _src_data is allowed by be a list to support creating vectors from
    # some scalar fields without necessarilly loading the data
    _center = "none"  # String in CENTERING
    _src_data = None  # numpy-like object (h5py too), or list of these objects
    name = None  # String
    crds = None  # Coordinate object
    time = None  # float
    # dict, this stuff will be copied by self.wrap
    info = None  #
    # dict, used for stuff that won't be blindly copied by self.wrap
    deep_meta = None
    pretty_name = None  # String

    pre_reshape_transform_func = None
    post_reshape_transform_func = None

    # these get reset when data is set
    _layout = None
    _nr_comps = None
    _nr_comp = None
    _dtype = None

    # set when data is retrieved
    _cache = None  # this will always be a numpy array

    def __init__(self, name, crds, data, center="Node", time=0.0, info=None,
                 deep_meta=None, forget_source=False, pretty_name=None,
                 pre_reshape_transform_func=None,
                 post_reshape_transform_func=None,
                 **kwargs):
        self.name = name
        self.center = center
        self.time = time
        self.crds = crds
        self.data = data

        if pretty_name is None:
            self.pretty_name = self.name
        else:
            self.pretty_name = pretty_name

        if pre_reshape_transform_func is not None:
            self.pre_reshape_transform_func = pre_reshape_transform_func
        if post_reshape_transform_func is not None:
            self.post_reshape_transform_func = post_reshape_transform_func

        self.info = {} if info is None else info
        self.deep_meta = {} if deep_meta is None else deep_meta
        for k, v in kwargs.items():
            if k.startswith("_"):
                self.deep_meta[k[1:]] = v
            else:
                self.info[k] = v

        if not "force_layout" in self.deep_meta:
            if "force_layout" in self.info:
                warnings.warn("deprecated force_layout syntax: kwarg should "
                              "be given as _force_layout")
                self.deep_meta["force_layout"] = self.info["force_layout"]
            else:
                self.deep_meta["force_layout"] = LAYOUT_DEFAULT
        self.deep_meta["force_layout"] = self.deep_meta["force_layout"].lower()

        if not "copy" in self.deep_meta:
            self.deep_meta["copy"] = False

        if forget_source:
            self.forget_source()

    @property
    def type(self):
        return self._TYPE

    @property
    def center(self):
        return self._center
    @center.setter
    def center(self, new_center):
        new_center = new_center.lower()
        assert new_center in self._CENTERING
        self._center = new_center

    @property
    def layout(self):
        """ get the layout type, 'interlaced' (AOS) | 'flat (SOA)'. This at most
        calls _detect_layout(_src_data) which tries to be lazy """
        if self._layout is None:
            if self.deep_meta["force_layout"] != LAYOUT_DEFAULT:
                self._layout = self.deep_meta["force_layout"]
            else:
                self._layout = self._detect_layout(self._src_data)
        return self._layout

    @layout.setter
    def layout(self, new_layout):
        """ UNTESTED, unload data if layout changes """
        new_layout = new_layout.lower()
        if self.is_loaded():
            current_layout = self.layout
            if new_layout != current_layout:
                self.unload()
        self.deep_meta["force_layout"] = new_layout
        self._layout = None

    @property
    def nr_dims(self):
        """ returns number of dims, this should be the number of dims of
        the underlying data, but is assumed a priori so no data load is
        done here """
        return self.nr_sdims + 1

    @property
    def nr_sdims(self):
        """ number of spatial dims, same as crds.nr_dims, does not
        explicitly load the data """
        return self.crds.nr_dims

    @property
    def nr_comps(self):
        """ how many components are there? Only inspects _src_data """
        # this gets # of comps from src_data, so layout is given as layout
        # of src_data since it might not be loaded yet
        if self._nr_comps is None:
            layout = self._detect_layout(self._src_data)
            if layout == LAYOUT_INTERLACED:
                if isinstance(self._src_data, (list, tuple)):
                    # this is an awkward way to get the data in here...
                    self._nr_comps = self._src_data[0].shape[-1]
                else:
                    self._nr_comps = self._src_data.shape[-1]
            elif layout == LAYOUT_FLAT:
                # length of 1st dim... this works for ndarrays && lists
                self._nr_comps = len(self._src_data)
            elif layout == LAYOUT_SCALAR:
                self._nr_comps = 1
            else:
                raise RuntimeError("Could not detect data layout; "
                                   "can't give nr_comps")
                # return self._src_data.shape[-1]
        return self._nr_comps

    @property
    def nr_comp(self):
        """ dimension of the components of the vector, loads the data if
        self.layout does """
        if self._nr_comp is None:
            layout = self.layout
            if layout == LAYOUT_FLAT:
                self._nr_comp = 0
            elif layout == LAYOUT_INTERLACED:
                self._nr_comp = self.crds.nr_dims
            elif layout == LAYOUT_SCALAR:
                # use same as interlaced for slicing convenience, note
                # this is only for a one component vector, for scalars
                # nr_comp is None
                self._nr_comp = self.crds.nr_dims
            else:
                raise RuntimeError("Could not detect data layout; "
                                   "can't give nr_comp")
        return self._nr_comp

    @property
    def shape(self):
        """ returns the shape of the underlying data, does not explicitly load
        the data """
        s = self.sshape
        try:
            s.insert(self.nr_comp, self.nr_comps)
        except TypeError:
            pass

        return s

    @property
    def sshape(self):
        """ shape of spatial dimensions, does not include comps, and is not
        the same shape as underlying array, does not load the data """
        # it is enforced that the cached data has a shape that agrees with
        # the coords by _reshape_ndarray_to_crds... actually, that method
        # requires this method to depend on the crd shape
        if self.iscentered("node"):
            return list(self.crds.shape_nc)
        elif self.iscentered("cell"):
            return list(self.crds.shape_cc)
        else:
            logging.warn("edge/face vectors not implemented, assuming "
                         "node shape")
            return self.crds.shape

    @property
    def size(self):
        """ how many values are in underlying data """
        return np.product(self.shape)

    @property
    def ssize(self):
        """ how many values make up one component of the underlying data """
        return np.product(self.sshape)

    @property
    def dtype(self):
        # print(type(self._src_data))
        # dtype.name is for pruning endianness out of dtype
        if self._dtype is None:
            if isinstance(self._src_data, (list, tuple)):
                dt = self._src_data[0].dtype
            else:
                dt = self._src_data.dtype

            if isinstance(dt, np.dtype):
                self._dtype = np.dtype(dt.name)
            else:
                self._dtype = np.dtype(dt)
        return self._dtype

    @property
    def data(self):
        """ if you want to fill the cache, this will do it, note that
        to empty the cache later you can always use unload """
        if self._cache is None:
            self._fill_cache()
        return self._cache
    @data.setter
    def data(self, dat):
        # clean up
        self._purge_cache()
        self._layout = None
        self._nr_comp = None
        self._nr_comps = None
        self._dtype = None
        self._src_data = dat
        self._translate_src_data()
        # do some sort of lazy pre-setup _src_data inspection?

    def is_loaded(self):
        return self._cache is not None

    def _purge_cache(self):
        """ does not guarentee that the memory will be freed """
        self._cache = None

    def _fill_cache(self):
        """ actually load data into the cache """
        self._cache = self._src_data_to_ndarray()

    def _translate_src_data(self):
        pass

    def _src_data_to_ndarray(self):
        """ prep the src data into something usable and enforce a layout """
        # some magic may need to happen here to accept more than np/h5 data
        # override if a type does something fancy (eg, interlacing)
        # and dat.flags["OWNDATA"]  # might i want this?
        src_data_layout = self._detect_layout(self._src_data)
        force_layout = self.deep_meta["force_layout"]

        # we will preserve layout or we already have the correct layout,
        # do no translation
        if force_layout == LAYOUT_DEFAULT or \
           force_layout == src_data_layout:
            return self._dat_to_ndarray(self._src_data)

        # if layout is found to be other, i cant do anything with that
        elif src_data_layout == LAYOUT_OTHER:
            logging.warn("Cannot auto-detect layout; not translating; "
                         "performance may suffer")
            return self._dat_to_ndarray(self._src_data)

        # ok, we demand FLAT arrays, make it so
        elif force_layout == LAYOUT_FLAT:
            if src_data_layout != LAYOUT_INTERLACED:
                raise RuntimeError("I should not be here")

            nr_comps = self.nr_comps
            data_dest = np.empty(self.shape, dtype=self.dtype)
            for i in range(nr_comps):
                # NOTE: I wonder if this is the fastest way to reorder
                data_dest[i, ...] = self._src_data[..., i]
                # NOTE: no special case for lists, they are not
                # interpreted this way
            return self._dat_to_ndarray(data_dest)

        # ok, we demand INTERLACED arrays, make it so
        elif force_layout == LAYOUT_INTERLACED:
            if src_data_layout != LAYOUT_FLAT:
                raise RuntimeError("I should not be here")

            nr_comps = self.nr_comps
            dtype = self.dtype
            data_dest = np.empty(self.shape, dtype=dtype)
            for i in range(nr_comps):
                data_dest[..., i] = self._src_data[i]

            self._layout = LAYOUT_INTERLACED
            return self._dat_to_ndarray(data_dest)

        # catch the remaining cases
        elif self.deep_meta["force_layout"] == LAYOUT_OTHER:
            raise ValueError("How should I know how to force other layout?")
        else:
            raise ValueError("Bad argument for layout forcing")

        raise RuntimeError("I should not be here")

    def _dat_to_ndarray(self, dat):
        """ This should be the last thing called for all data that gets put
        into the cache. It makes dimensions jive correctly. This will translate
        non-ndarray data structures to a flat ndarray. Also, this function
        makes damn sure that the dimensionality of the coords matches the
        dimensionality of the array, which is to say, if the array is only 2d,
        but the coords have a 3rd dimension with length 1, reshape the array
        to include that extra dimension.
        """
        # dtype.name is for pruning endianness out of dtype
        if isinstance(dat, np.ndarray):
            arr = np.array(dat, dtype=dat.dtype.name, copy=self.deep_meta["copy"])
        elif isinstance(dat, (list, tuple)):
            dt = dat[0].dtype.name
            arr = np.array([np.array(d, dtype=dt, copy=self.deep_meta["copy"])
                            for d in dat], dtype=dt)
        # elif isinstance(dat, Field):
        #     arr = dat.data  # not the way
        else:
            arr = np.array(dat, dtype=dat.dtype.name, copy=self.deep_meta["copy"])

        if self.pre_reshape_transform_func is not None:
            arr = self.pre_reshape_transform_func(self, arr)

        arr = self._reshape_ndarray_to_crds(arr)

        if self.post_reshape_transform_func is not None:
            arr = self.post_reshape_transform_func(self, arr)

        return arr

    def _reshape_ndarray_to_crds(self, arr):
        """ enforce same dimensionality as coords here!
        self.shape better still be using the crds shape corrected for
        node / cell centering
        """
        if list(arr.shape) == self.shape:
            return arr
        else:
            ret = arr.reshape(self.shape)
            return ret

    def _detect_layout(self, dat):
        """ returns LAYOUT_XXX, this just looks at len(dat) or dat.shape """
        # if i receive a list, then i suppose i have a list of
        # arrays, one for each component... this is a flat layout
        sshape = list(self.sshape)
        if isinstance(dat, (list, tuple)):
            # Make sure the list makes sense... this strictly speaking
            # doesn't need to happen, but it's a bad habbit to allow
            # shapes through that I don't explicitly plan for, since
            # this is just the door to the rabbit hole
            # BUT, this has the side effect that one can't create vector
            # fields with vx = Xcc, one has to use
            # vx = Xcc + 0 * Ycc + 0 * Zcc, which is rather cumbersome
            # for d in dat:
            #     assert(list(d.shape) == list(sshape))
            return LAYOUT_FLAT

        if list(dat.shape) == sshape:
            return LAYOUT_SCALAR

        # if the crds shape has more values than the dat.shape
        # then try trimming the directions that have 1 element
        # this can happen when crds are 3d, but field is only 2d
        ## I'm not sure why this needs to happen... but tests fail
        ## without it
        while len(sshape) > len(dat.shape) - 1:
            try:
                sshape.remove(1)
            except ValueError:
                break

        # check which dims match the shape of the crds
        if list(dat.shape) == sshape:
            layout = LAYOUT_SCALAR
        elif list(dat.shape[1:]) == sshape:
            layout = LAYOUT_FLAT
        elif list(dat.shape[:-1]) == sshape:
            layout = LAYOUT_INTERLACED
        elif dat.shape[0] == np.prod(sshape):
            layout = LAYOUT_INTERLACED
        elif dat.shape[-1] == np.prod(sshape):
            layout = LAYOUT_FLAT
        else:
            # if this happens, don't ignore it even if it happens to work
            logging.warn("could not detect layout for '{0}': shape = {1} "
                         "target shape = {2}"
                         "".format(self.name, dat.shape, sshape))
            layout = LAYOUT_OTHER

        return layout

    def _prepare_slice(self, selection):
        """ if selection has a slice for component dimension, set it aside """
        comp_slc = slice(None)
        if isinstance(selection, tuple):
            selection = list(selection)
        if isinstance(selection, list):
            if self.nr_comps > 0 and len(selection) == self.nr_dims:
                comp_slc = selection.pop[self.nr_comp]
        return selection, comp_slc

    def _finalize_slice(self, slices, crdlst, reduced, comp_slc):
        # no slice necessary, just pass the field through
        if list(slices) == [slice(None)] * len(slices):
            return self

        # coord transforms are not copied on purpose
        crds = coordinate.wrap_crds(self.crds.type, crdlst)

        try:
            slices.insert(self.nr_comp, comp_slc)
        except TypeError:
            pass

        # if we sliced the hell out of the array, just
        # return the value that's left, ndarrays have the same behavior
        slced_dat = self.data[tuple(slices)]
        if len(reduced) == len(slices) or slced_dat.size == 1:
            return slced_dat
        else:
            fld = self.wrap(slced_dat,
                            {"crds": crds})
            # if there are reduced dims, put them into the deep_meta dict
            if len(reduced) > 0:
                fld.deep_meta["reduced"] = reduced
            return fld

    def forget_source(self):
        self._src_data = self.data

    def slice(self, selection):
        """ Slice the field using a string like "y=3i:6i:2,z=0" or a standard
        list of slice objects like one would give to numpy. In a string, i
        means by index, and bare numbers mean by the index closest to that
        value; see Coordinate.make_slice docs for an example. The semantics
        for keeping / droping dimensions are the same as for numpy arrays.
        This means selections that leave one crd in a given dimension reduce
        that dimension out. For other behavior see
        slice_reduce and slice_keep
        """
        cc = self.iscentered("Cell")
        selection, comp_slc = self._prepare_slice(selection)
        slices, crdlst, reduced = self.crds.make_slice(selection, cc=cc)
        return self._finalize_slice(slices, crdlst, reduced, comp_slc)

    def slice_reduce(self, selection):
        """ Slice the field, then go through all dims and look for dimensions
        with only one coordinate. Reduce those dimensions out of the new
        field """
        cc = self.iscentered("Cell")
        selection, comp_slc = self._prepare_slice(selection)
        slices, crdlst, reduced = self.crds.make_slice_reduce(selection,
                                                              cc=cc)
        return self._finalize_slice(slices, crdlst, reduced, comp_slc)

    def slice_and_keep(self, selection):
        """ Slice the field, then go through dimensions that would be reduced
        by a normal numpy slice (like saying 'z=0') and keep those dimensions
        in the new field """
        cc = self.iscentered("Cell")
        selection, comp_slc = self._prepare_slice(selection)
        slices, crdlst, reduced = self.crds.make_slice_keep(selection,
                                                            cc=cc)
        return self._finalize_slice(slices, crdlst, reduced, comp_slc)

    def set_slice(self, selection, value):
        cc = self.iscentered("Cell")
        selection = self._prepare_slice(selection)[0]
        slices, _ = self.crds.make_slice(selection, cc=cc)[:2]
        self.data[tuple(slices)] = value
        return None

    def unload(self):
        """ does not guarentee that the memory will be freed """
        self._purge_cache()

    @classmethod
    def istype(cls, type_str):
        return cls._TYPE == type_str.lower()

    def iscentered(self, center_str):
        return self.center == center_str.lower()

    def iter_points(self, center=None, **kwargs): #pylint: disable=W0613
        if center is None:
            center = self.center
        return self.crds.iter_points(center=center)

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        """ unload the data """
        self.unload()
        return None

    def __iter__(self):
        """ iterate though all values in the data, raveled """
        for val in self.data.ravel():
            yield val

    ##################################
    ## Utility methods to get at crds
    # these are the same as something like self.crds['xnc']
    # or self.crds.get_crd()
    def get_crd(self, axis, shaped=False):
        """ return crd along axis with same centering as field
        axis can be crd name as string, or index, as in x==2, y==1, z==2 """
        return self.crds.get_crd(axis, center=self.center, shaped=shaped)

    def get_crd_nc(self, axis, shaped=False):
        """ returns a flat ndarray of coordinates along a given axis
        axis can be crd name as string, or index, as in x==2, y==1, z==2 """
        return self.crds.get_nc(axis, shaped=shaped)

    def get_crd_cc(self, axis, shaped=False):
        """ returns a flat ndarray of coordinates along a given axis
        axis can be crd name as string, or index, as in x==2, y==1, z==2 """
        return self.crds.get_cc(axis, shaped=shaped)

    def get_crd_ec(self, axis, shaped=False):
        """ returns a flat ndarray of coordinates along a given axis
        axis can be crd name as string, or index, as in x==2, y==1, z==2 """
        return self.crds.get_ec(axis, shaped=shaped)

    def get_crd_fc(self, axis, shaped=False):
        """ returns a flat ndarray of coordinates along a given axis
        axis can be crd name as string, or index, as in x==2, y==1, z==2 """
        return self.crds.get_fc(axis, shaped=shaped)

    ## these return all crd dimensions
    # these are the same as something like self.crds.get_crds()
    def get_crds(self, axes=None, shaped=False):
        """ return all crds as list of ndarrays with same centering as field """
        return self.crds.get_crds(axes=axes, center=self.center, shaped=shaped)

    def get_crds_nc(self, axes=None, shaped=False):
        """ returns all node centered coords as a list of ndarrays, flat if
        shaped==False, or shaped if shaped==True """
        return self.crds.get_crds_nc(axes=axes, shaped=shaped)

    def get_crds_cc(self, axes=None, shaped=False):
        """ returns all cell centered coords as a list of ndarrays, flat if
        shaped==False, or shaped if shaped==True """
        return self.crds.get_crds_cc(axes=axes, shaped=shaped)

    def get_crds_fc(self, axes=None, shaped=False):
        """ returns all face centered coords as a list of ndarrays, flat if
        shaped==False, or shaped if shaped==True """
        return self.crds.get_crds_fc(axes=axes, shaped=shaped)

    def get_crds_ec(self, axes=None, shaped=False):
        """ returns all edge centered coords as a list of ndarrays, flat if
        shaped==False, or shaped if shaped==True """
        return self.crds.get_crds_ec(axes=axes, shaped=shaped)

    def is_spherical(self):
        return self.crds.is_spherical()

    #######################
    ## emulate a container

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        return self.slice(item)

    def __setitem__(self, key, value):
        """ just act as if you setitem on underlying data """
        self.data.__setitem__(key, value)

    def __delitem__(self, item):
        """ just act as if you delitem on underlying data, probably raises a
        ValueError """
        self.data.__delitem__(item)

    ##########################
    ## emulate a numeric type

    def __array__(self, dtype=None):
        # dtype = None is ok, datatype won't change
        return np.array(self.data, dtype=dtype, copy=False)

    def wrap(self, arr, context=None, typ=None):
        """ arr is the data to wrap... context is exta deep_meta to pass
        to the constructor. The return is just a number if arr is a
        1 element ndarray, this is for ufuncs that reduce to a scalar """
        if arr is NotImplemented:
            return NotImplemented
        # if just 1 number wrappen in an array, unpack the value and
        # return it... this is more ufuncy behavior
        if isinstance(arr, np.ndarray) and arr.size == 1:
            return np.ravel(arr)[0]
        if context is None:
            context = {}
        name = context.pop("name", self.name)
        pretty_name = context.pop("pretty_name", self.pretty_name)
        crds = context.pop("crds", self.crds)
        center = context.pop("center", self.center)
        time = context.pop("time", self.time)
        # should it always return the same type as self?
        if typ is None:
            typ = type(self)
        else:
            typ = field_type(typ)
        return typ(name, crds, arr, time=time, center=center,
                   info=self.info, deep_meta=context, pretty_name=pretty_name)

    def __array_wrap__(self, out_arr, context=None): #pylint: disable=W0613
        # print("wrapping")
        return self.wrap(out_arr)

    # def __array_finalize__(self, *args, **kwargs):
    #     print("attempted call to field.__array_finalize__")

    def __add__(self, other):
        return self.wrap(self.data.__add__(other))
    def __sub__(self, other):
        return self.wrap(self.data.__sub__(other))
    def __mul__(self, other):
        return self.wrap(self.data.__mul__(other))
    def __div__(self, other):
        return self.wrap(self.data.__div__(other))
    def __truediv__(self, other):
        return self.wrap(self.data.__truediv__(other))
    def __floordiv__(self, other):
        return self.wrap(self.data.__floordiv__(other))
    def __mod__(self, other):
        return self.wrap(self.data.__mod__(other))
    def __divmod__(self, other):
        return self.wrap(self.data.__divmod__(other))
    def __pow__(self, other):
        return self.wrap(self.data.__pow__(other))
    def __lshift__(self, other):
        return self.wrap(self.data.__lshift__(other))
    def __rshift__(self, other):
        return self.wrap(self.data.__rshift__(other))
    def __and__(self, other):
        return self.wrap(self.data.__rshift__(other))
    def __xor__(self, other):
        return self.wrap(self.data.__rshift__(other))
    def __or__(self, other):
        return self.wrap(self.data.__rshift__(other))

    def __radd__(self, other):
        return self.wrap(self.data.__radd__(other))
    def __rsub__(self, other):
        return self.wrap(self.data.__rsub__(other))
    def __rmul__(self, other):
        return self.wrap(self.data.__rmul__(other))
    def __rdiv__(self, other):
        return self.wrap(self.data.__rdiv__(other))
    def __rtruediv__(self, other):
        return self.wrap(self.data.__rtruediv__(other))
    def __rfloordiv__(self, other):
        return self.wrap(self.data.__rfloordiv__(other))
    def __rmod__(self, other):
        return self.wrap(self.data.__rmod__(other))
    def __rdivmod__(self, other):
        return self.wrap(self.data.__rdivmod__(other))
    def __rpow__(self, other):
        return self.wrap(self.data.__rpow__(other))

    # inplace operations are not implemented since the data
    # can up and disappear (due to unload)... this could cause
    # confusion
    def __iadd__(self, other):
        return NotImplemented
    def __isub__(self, other):
        return NotImplemented
    def __imul__(self, other):
        return NotImplemented
    def __idiv__(self, other):
        return NotImplemented
    def __itruediv__(self, other): #pylint: disable=R0201,W0613
        return NotImplemented
    def __ifloordiv__(self, other): #pylint: disable=R0201,W0613
        return NotImplemented
    def __imod__(self, other):
        return NotImplemented
    def __ipow__(self, other):
        return NotImplemented

    def __neg__(self):
        return self.wrap(self.data.__neg__())
    def __pos__(self):
        return self.wrap(self.data.__pos__())
    def __abs__(self):
        return self.wrap(self.data.__abs__())
    def __invert__(self):
        return self.wrap(self.data.__invert__())

    def any(self):
        return self.data.any()
    def all(self):
        return self.data.all()

    def __lt__(self, other):
        return self.wrap(self.data.__lt__(other))
    def __le__(self, other):
        return self.wrap(self.data.__le__(other))
    def __eq__(self, other):
        return self.wrap(self.data.__eq__(other))
    def __ne__(self, other):
        return self.wrap(self.data.__ne__(other))
    def __gt__(self, other):
        return self.wrap(self.data.__gt__(other))
    def __ge__(self, other):
        return self.wrap(self.data.__ge__(other))


class ScalarField(Field):
    _TYPE = "scalar"

    @property
    def nr_dims(self):
        return self.nr_sdims

    # FIXME: there is probably a better way to deal with scalars not
    # having a component dimension
    @property
    def nr_comp(self):
        raise TypeError("Scalars have no components")

    @property
    def nr_comps(self):
        return 0

    # no downsample / transpose / swap axes for vectors yet since that would
    # have the added layer of checking the layout
    def downsample(self):
        """ downsample the spatial dimensions by a factor of 2 """
        # FIXME: this implementation assumes a lot about the field
        end = np.array(self.sshape) // 2
        dat = self.data

        if self.nr_sdims == 1:
            downdat = 0.5 * (dat[:end[0]:2] + dat[1:end[0]:2])
        elif self.nr_sdims == 2:
            downdat = 0.25 * (dat[:end[0]:2, :end[1]:2] +
                              dat[1:end[0]:2, :end[1]:2] +
                              dat[:end[0]:2, 1:end[1]:2] +
                              dat[1:end[0]:2, 1:end[1]:2])
        elif self.nr_sdims == 3:
            downdat = 0.125 * (dat[:end[0]:2, :end[1]:2, :end[2]:2] +
                               dat[1:end[0]:2, :end[1]:2, :end[2]:2] +
                               dat[:end[0]:2, 1:end[1]:2, :end[2]:2] +
                               dat[:end[0]:2, :end[1]:2, 1:end[2]:2] +
                               dat[1:end[0]:2, 1:end[1]:2, :end[2]:2] +
                               dat[1:end[0]:2, :end[1]:2, 1:end[2]:2] +
                               dat[:end[0]:2, 1:end[1]:2, 1:end[2]:2] +
                               dat[1:end[0]:2, 1:end[1]:2, 1:end[2]:2])

        downclist = self.crds.get_clist(np.s_[::2])
        downcrds = coordinate.wrap_crds("nonuniform_cartesian", downclist)
        return self.wrap(downdat, {"crds": downcrds})

    def transpose(self, *axes):
        """ same behavior as numpy transpose, alse accessable
        using np.transpose(fld) """
        if axes == (None, ) or len(axes) == 0:
            axes = range(self.nr_dims - 1, -1, -1)
        if len(axes) != self.nr_dims:
            raise ValueError("transpose can not change number of axes")
        clist = self.crds.get_clist()
        new_clist = [clist[ax] for ax in axes]
        t_crds = coordinate.wrap_crds(self.crds.type, new_clist)
        t_data = self.data.transpose(axes)
        return self.wrap(t_data, {"crds": t_crds})

    def swap_axes(self, a, b):
        new_clist = self.crds.get_clist()
        new_clist[a], new_clist[b] = new_clist[b], new_clist[a]
        new_crds = coordinate.wrap_crds(self.crds.type, new_clist)
        new_data = self.data.swap_axes(a, b)
        return self.wrap(new_data, {"crds": new_crds})


class VectorField(Field):
    _TYPE = "vector"
    _COMPONENT_NAMES = "xyzuvw"

    def component_views(self):
        """ return numpy views to components individually, memory layout
        of the original field is maintained """
        nr_comps = self.nr_comps
        # comp_slc = [slice(None)] * self.nr_dims
        if self.layout == LAYOUT_FLAT:
            return [self.data[i, ...] for i in range(nr_comps)]
        elif self.layout == LAYOUT_INTERLACED:
            return [self.data[..., i] for i in range(nr_comps)]
        else:
            return [self.data[..., i] for i in range(nr_comps)]

    def component_fields(self):
        n = self.name
        crds = self.crds
        c = self.center
        t = self.time
        views = self.component_views()
        lst = [None] * len(views)
        for i, v in enumerate(views):
            lst[i] = ScalarField("{0}{1}".format(n, self._COMPONENT_NAMES[i]),
                                 crds, v, center=c, time=t, info=self.info)
        return lst

    def __getitem__(self, item):
        if isinstance(item, string_types) and item in self._COMPONENT_NAMES:
            i = self._COMPONENT_NAMES.index(item)
            if self.layout == LAYOUT_FLAT:
                dat = self.data[i, ...]
            else:
                dat = self.data[..., i]
            return self.wrap(dat, typ="Scalar",
                             context={"name": self.name + item})
        else:
            return super(VectorField, self).__getitem__(item)

class MatrixField(Field):
    _TYPE = "matrix"


class TensorField(Field):
    _TYPE = "tensor"


##
## EOF
##
