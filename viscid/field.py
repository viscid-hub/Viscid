# pylint: disable=too-many-lines
"""Fields are the basis of Viscid's data abstration

Fields belong in grids, or by themselves as a result of a calculation.
They can belong to a :class:`Grid` as the result of a file load, or
by themselves as the result of a calculation. This module has some
convenience functions for creating fields similar to `Numpy`.
"""

from __future__ import print_function
import warnings
from itertools import count, islice
from inspect import isclass

import numpy as np

from viscid import logger
from viscid.compat import string_types, izip_longest
from viscid import coordinate
from viscid import vutil
from viscid import tree

LAYOUT_DEFAULT = "none"  # do not translate
LAYOUT_INTERLACED = "interlaced"
LAYOUT_FLAT = "flat"
LAYOUT_SCALAR = "scalar"
LAYOUT_OTHER = "other"


def arrays2field(dat_arr, crd_arrs, name="NoName", center=None,
                 crd_names="zyxwvu"):
    """Turn arrays into fields so they can be used in viscid.plot, etc.

    This is a convenience function that takes care of making coordnates
    and the like. If the default behavior doesn't work for you, you'll
    need to make your own coordnates and call
    :py:func:`viscid.field.wrap_field`.

    Args:
        dat_arr (ndarray): data with len(crd_arrs) or len(crd_arrs) + 1
            dimensions
        crd_arrs (list of ndarrays): zyx list of ndarrays that
            describe the node centered coordnates of the field
        name (str): some name
        center (str, None): If not None, translate field to this
            centering (node or cell)
    """
    crds = coordinate.arrays2crds(crd_arrs, crd_names=crd_names)

    # discover what kind of data was given
    crds_shape_nc = list(crds.shape_nc)
    crds_shape_cc = list(crds.shape_cc)
    dat_arr_shape = list(dat_arr.shape)

    if len(dat_arr.shape) == len(crds.shape_nc):
        discovered_type = "scalar"
        discovered_layout = LAYOUT_FLAT
        if crds_shape_nc == dat_arr_shape:
            discovered_center = "node"
        elif crds_shape_cc == dat_arr_shape:
            discovered_center = "cell"
        else:
            raise ValueError("Can't detect centering for scalar dat_arr")
    elif len(dat_arr.shape) + 1 == len(crds.shape_nc):
        discovered_type = "vector"
        if crds_shape_nc == dat_arr_shape[:-1]:
            discovered_layout = LAYOUT_INTERLACED
            discovered_center = "node"
        elif crds_shape_cc == dat_arr_shape[:-1]:
            discovered_layout = LAYOUT_INTERLACED
            discovered_center = "cell"
        elif crds_shape_nc == dat_arr_shape[1:]:
            discovered_layout = LAYOUT_FLAT
            discovered_center = "node"
        elif crds_shape_cc == dat_arr_shape[1:]:
            discovered_layout = LAYOUT_FLAT
            discovered_center = "cell"
        else:
            raise ValueError("Can't detect centering for vector dat_arr")
    else:
        raise ValueError("crds and data have incompatable dimensions: {0} {1}"
                         "".format(dat_arr.shape, crds.shape_nc))

    fld = wrap_field(dat_arr, crds, name=name, fldtype=discovered_type,
                     center=discovered_center, layout=discovered_layout)
    fld = fld.as_centered(center)
    return fld

def dat2field(dat_arr, name="NoName", fldtype="scalar", center=None,
              layout=LAYOUT_FLAT):
    """Makes np.arange coordnate arrays and calls arrays2field

    Args:
        dat_arr (ndarray): data
        name (str): name of field
        fldtype (str, optional): 'scalar' / 'vector'
        center (str, None): If not None, translate field to this
            centering (node or cell)
        layout (TYPE, optional): Description
    """
    sshape = []
    if fldtype.lower() == "scalar":
        sshape = dat_arr.shape
    elif fldtype.lower() == "vector":
        if layout == LAYOUT_FLAT:
            sshape = dat_arr.shape[1:]
        elif layout == LAYOUT_FLAT:
            sshape = dat_arr.shape[1:]
        else:
            raise ValueError("Unknown layout: {0}".format(layout))
    else:
        raise ValueError("Unknown type: {0}".format(fldtype))

    crd_arrs = [np.arange(s).astype(dat_arr.dtype) for s in sshape]
    return arrays2field(name, crd_arrs, dat_arr, center=center)

def empty(crds, dtype="f8", name="NoName", center="cell", layout=LAYOUT_FLAT,
          nr_comps=0, crd_names="zyxwvu", _initial_vals="empty", **kwargs):
    """Analogous to `numpy.empty` (uninitialized array)

    Parameters:
        crds (Coordinates, list, or tuple): Can be a coordinates
            object. Can also be a list of ndarrays describing
            coordinate arrays. Or, if it's just a list or tuple of
            integers, those integers are taken to be the nz,ny,nx shape
            and the coordinates will be fill with :py:func:`np.arange`.
        dtype (optional): some way to describe numpy dtype of data
        name (str): a way to refer to the field programatically
        center (str, optional): cell or node, there really isn't
            support for edge / face yet
        layout (str, optional): how data is stored, is in "flat" or
            "interlaced" (interlaced == AOS)
        nr_comps (int, optional): for vector fields, nr of components
        **kwargs: passed through to Field constructor
    """
    if not isinstance(crds, coordinate.Coordinates):
        # if crds is a list/tuple of integers, then make coordinate
        # arrays using arange
        if not isinstance(crds, (list, tuple, np.ndarray)):
            try:
                crds = list(crds)
            except TypeError:
                crds = [crds]
        if all([isinstance(c, int) for c in crds]):
            crds = [np.arange(c).astype(dtype) for c in crds]
        # now assume that crds is a list of coordinate arrays that arrays2crds
        # can understand
        crds = coordinate.arrays2crds(crds, crd_names=crd_names)

    if center.lower() == "cell":
        sshape = crds.shape_cc
    elif center.lower() == "node":
        sshape = crds.shape_nc
    else:
        sshape = crds.shape_nc

    if nr_comps == 0:
        fldtype = "scalar"
        shape = sshape
    else:
        fldtype = "vector"
        if layout.lower() == LAYOUT_INTERLACED:
            shape = list(sshape) + [nr_comps]
        else:
            shape = [nr_comps] + list(sshape)

    if _initial_vals == "empty":
        dat = np.empty(shape, dtype=dtype)
    elif _initial_vals == "zeros":
        dat = np.zeros(shape, dtype=dtype)
    elif _initial_vals == "ones":
        dat = np.ones(shape, dtype=dtype)
    else:
        raise ValueError("_initial_vals only accepts empty, zeros, ones; not {0}"
                         "".format(_initial_vals))
    return wrap_field(dat, crds, name=name, fldtype=fldtype, center=center,
                      **kwargs)

def zeros(crds, dtype="f8", name="NoName", center="cell", layout=LAYOUT_FLAT,
          nr_comps=0, **kwargs):
    """Analogous to `numpy.zeros`

    Returns:
        new :class:`Field` initialized to 0

    See Also: :meth:`empty`
    """
    return empty(crds, dtype=dtype, name=name, center=center, layout=layout,
                 nr_comps=nr_comps, _initial_vals="zeros", **kwargs)

def ones(crds, dtype="f8", name="NoName", center="cell", layout=LAYOUT_FLAT,
         nr_comps=0, **kwargs):
    """Analogous to `numpy.ones`

    Returns:
        new :class:`Field` initialized to 1

    See Also: :meth:`empty`
    """
    return empty(crds, dtype=dtype, name=name, center=center, layout=layout,
                 nr_comps=nr_comps, _initial_vals="ones", **kwargs)

def empty_like(fld, name="NoName", **kwargs):
    """Analogous to `numpy.empty_like`

    Makes a new, unitilialized :class:`Field`. Copies as much meta data
    as it can from `fld`.

    Parameters:
        fld: field to get coordinates / metadata from
        name: name for this field
        **kwargs: passed through to :class:`Field` constructor

    Returns:
        new uninitialized :class:`Field`
    """
    dat = np.empty(fld.shape, dtype=fld.dtype)
    c = fld.center
    t = fld.time
    return wrap_field(dat, fld.crds, name=name, fldtype=fld.fldtype, center=c,
                      time=t, parents=[fld], **kwargs)

def zeros_like(fld, name="NoName", **kwargs):
    """Analogous to `numpy.zeros_like`

    Returns:
        new :class:`Field` initialized to 0

    See Also: :meth:`empty_like`
    """
    dat = np.zeros(fld.shape, dtype=fld.dtype)
    c = fld.center
    t = fld.time
    return wrap_field(dat, fld.crds, name=name, fldtype=fld.fldtype, center=c,
                      time=t, parents=[fld], **kwargs)

def ones_like(fld, name="NoName", **kwargs):
    """Analogous to `numpy.ones_like`

    Returns:
        new :class:`Field` initialized to 1

    See Also: :meth:`empty_like`
    """
    dat = np.ones(fld.shape, dtype=fld.dtype)
    c = fld.center
    t = fld.time
    return wrap_field(dat, fld.crds, name=name, fldtype=fld.fldtype, center=c,
                      time=t, parents=[fld], **kwargs)

def scalar_fields_to_vector(fldlist, name="NoName", **kwargs):
    """Convert scalar fields to a vector field

    Parameters:
        name (str): name for the vector field
        fldlist: list of :class:`ScalarField`
        **kwargs: passed to :class:`VectorField` constructor

    Returns:
        A new :class:`VectorField`.
    """
    if not name:
        name = fldlist[0].name
    center = fldlist[0].center
    crds = fldlist[0]._src_crds
    time = fldlist[0].time
    # shape = fldlist[0].data.shape

    _crds = type(crds)(crds.get_clist())

    vfield = VectorField(name, _crds, fldlist, center=center, time=time,
                         meta=fldlist[0].meta, parents=[fldlist[0]],
                         **kwargs)
    return vfield

def field_type(fldtype):
    """Lookup a Field type

    The magic lookup happens when fldtype is a string, if fldtype is a class
    then just return the class for convenience.

    Parameters:
        fldtype: python class object or string describing a field type in
            some way

    Returns:
        a :class:`Field` subclass
    """
    if isclass(fldtype) and issubclass(fldtype, Field):
        return fldtype
    else:
        for cls in vutil.subclass_spider(Field):
            if cls.istype(fldtype):
                return cls
    logger.warn("Field type {0} not understood".format(fldtype))
    return None

def wrap_field(data, crds, name="NoName", fldtype="scalar", **kwargs):
    """Convenience script for wrapping ndarrays

    Parameters:
        data: Some data container, most likely a ``numpy.ndarray``
        crds (Coordinates): coordinates that describe the shape / grid
            of the field
        fldtype (str): 'scalar' / 'Vector'
        name (str): a way to refer to the field programatically
        **kwargs: passed through to :class:`Field` constructor

    Returns:
        A :class:`Field` instance.
    """
    #
    #len(clist), clist[0][0], len(clist[0][1]), type)
    cls = field_type(fldtype)
    if cls is not None:
        return cls(name, crds, data, **kwargs)
    else:
        raise NotImplementedError("can not decipher field")

def rewrap_field(fld):
    ret = type(fld)(fld.name, fld.crds, fld.data, center=fld.center,
                    forget_source=True, _copy=True, parents=[fld])
    return ret

class Field(tree.Leaf):
    _TYPE = "none"
    _CENTERING = ['node', 'cell', 'grid', 'face', 'edge']
    _COMPONENT_NAMES = ""

    # set on __init__
    # NOTE: _src_data is allowed by be a list to support creating vectors from
    # some scalar fields without necessarilly loading the data
    _center = "none"  # String in CENTERING
    _src_data = None  # numpy-like object (h5py too), or list of these objects

    _src_crds = None  # Coordinate object
    _crds = None  # Coordinate object

    # dict, this stuff will be copied by self.wrap
    meta = None  #
    # dict, used for stuff that won't be blindly copied by self.wrap
    deep_meta = None
    pretty_name = None  # String

    post_reshape_transform_func = None
    transform_func_kwargs = None

    # _parent_field is a hacky way to keep a 'parent' field in sync when data
    # is loaded... this is used for converting cell centered data ->
    # node centered and back... this only works because _fill_cache
    # ONLY sets self._cache... if other meta data were set by
    # _fill_cache, that would need to propogate upstream too
    _parent_field = None

    # these get reset when data is set
    _layout = None
    _nr_comps = None
    _nr_comp = None
    _dtype = None

    # set when data is retrieved
    _cache = None  # this will always be a numpy array

    def __init__(self, name, crds, data, center="Node", time=0.0, meta=None,
                 deep_meta=None, forget_source=False, pretty_name=None,
                 post_reshape_transform_func=None,
                 transform_func_kwargs=None,
                 info=None, parents=None, _parent_field=None,
                 **kwargs):
        """
        Args:
            parents: Dataset, Grid, Field, or list of any of
                those. These parents are the sources for find_info, and
                and monkey-pached methods.
            _parent_field (Field): special parent where data can be taken
                for lazy loading. This field is added to parents
                automatically

        Other Parameters:
            kwargs with a leading underscore (like _copy) are added to
            the deep_meta dict without the leading _. Everything else
            is added to the meta dict
        """
        super(Field, self).__init__(name, time, info, parents)
        self.center = center
        self._src_crds = crds
        self.data = data

        if pretty_name is None:
            self.pretty_name = self.name
        else:
            self.pretty_name = pretty_name

        # # i think this is a mistake
        # if isinstance(data, (list, tuple)) and isinstance(data[0], Field):
        #     if post_reshape_transform_func is None:
        #         post_reshape_transform_func = data[0].post_reshape_transform_func
        #     if transform_func_kwargs is None:
        #         transform_func_kwargs = data[0].transform_func_kwargs

        if post_reshape_transform_func is not None:
            self.post_reshape_transform_func = post_reshape_transform_func
        if transform_func_kwargs:
            self.transform_func_kwargs = transform_func_kwargs
        else:
            self.transform_func_kwargs = {}

        self.meta = {} if meta is None else meta
        self.deep_meta = {} if deep_meta is None else deep_meta
        for k, v in kwargs.items():
            if k.startswith("_"):
                self.deep_meta[k[1:]] = v
            else:
                self.meta[k] = v

        if not "force_layout" in self.deep_meta:
            if "force_layout" in self.meta:
                warnings.warn("deprecated force_layout syntax: kwarg should "
                              "be given as _force_layout")
                self.deep_meta["force_layout"] = self.meta["force_layout"]
            else:
                self.deep_meta["force_layout"] = LAYOUT_DEFAULT
        self.deep_meta["force_layout"] = self.deep_meta["force_layout"].lower()

        if not "copy" in self.deep_meta:
            self.deep_meta["copy"] = False

        self._parent_field = _parent_field
        if _parent_field is not None:
            self.parents.insert(0, _parent_field)

        if forget_source:
            self.forget_source()

    @property
    def fldtype(self):
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
        return self._src_crds.nr_dims

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
                self._nr_comp = self._src_crds.nr_dims
            elif layout == LAYOUT_SCALAR:
                # use same as interlaced for slicing convenience, note
                # this is only for a one component vector, for scalars
                # nr_comp is None
                self._nr_comp = self._src_crds.nr_dims
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
            return list(self._src_crds.shape_nc)
        elif self.iscentered("cell"):
            return list(self._src_crds.shape_cc)
        else:
            logger.warn("edge/face vectors not implemented, assuming "
                        "node shape")
            return self._src_crds.shape

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
    def crds(self):
        if self._crds is None:
            self._crds = self._src_crds.apply_reflections()
        return self._crds

    @crds.setter
    def crds(self, val):
        self._crds = None
        self._src_crds = val

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
        # self._translate_src_data()  # um, what's this for? looks dangerous
        # do some sort of lazy pre-setup _src_data inspection?

    @property
    def blocks(self):
        return [self]

    @property
    def nr_blocks(self):  # pylint: disable=no-self-use
        return 1

    def is_loaded(self):
        return self._cache is not None

    def _purge_cache(self):
        """ does not guarentee that the memory will be freed """
        self._cache = None
        if self._parent_field is not None:
            self._parent_field._cache = self._cache
    def _fill_cache(self):
        """ actually load data into the cache """
        self._cache = self._src_data_to_ndarray()
        if self._parent_field is not None:
            self._parent_field._cache = self._cache

    # um, what was this for? looks dangerous
    # def _translate_src_data(self):
    #     pass

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
            logger.warn("Cannot auto-detect layout; not translating; "
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
            tmp = [np.array(d, dtype=dt, copy=self.deep_meta["copy"]) for d in dat]
            _shape = tmp[0].shape
            arr = np.empty([len(tmp)] + list(_shape), dtype=dt)
            for i, t in enumerate(tmp):
                arr[i] = t
        # elif isinstance(dat, Field):
        #     arr = dat.data  # not the way
        else:
            arr = np.array(dat, dtype=dat.dtype.name, copy=self.deep_meta["copy"])

        arr = self._reshape_ndarray_to_crds(arr)
        try:
            nr_comp = self.nr_comp
            nr_comps = self.nr_comps
        except TypeError:
            nr_comp = None
            nr_comps = None

        # FIXME, there should be a flag for whether or not this should
        # make a copy of the array if it's not contiguous in memory
        arr = self._src_crds.reflect_fld_arr(arr, self.iscentered("Cell"),
                                             nr_comp, nr_comps)

        if self.post_reshape_transform_func is not None:
            arr = self.post_reshape_transform_func(self, self._src_crds, arr,
                                                   **self.transform_func_kwargs)

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
            logger.warn("could not detect layout for '{0}': shape = {1} "
                        "target shape = {2}"
                        "".format(self.name, dat.shape, sshape))
            layout = LAYOUT_OTHER

        return layout

    def _prepare_slice(self, selection):
        """ if selection has a slice for component dimension, set it aside """
        comp_slc = None

        # try to look for a vector component slice
        if self.nr_comps > 0:
            try:
                if isinstance(selection, (list, tuple)):
                    sel_lst = list(selection)
                    _isstr = False
                elif isinstance(selection, string_types):
                    _ = selection.replace('_', ',')
                    sel_lst = [s for s in _.split(",")]  # pylint: disable=maybe-no-member
                    _isstr = True
                else:
                    raise TypeError()

                for i, s in enumerate(sel_lst):
                    try:
                        if len(s.strip()) == 1 and s in self._COMPONENT_NAMES:
                            # ok, this is asking for a component
                            comp_slc = s
                            sel_lst.pop(i)
                            break
                    except AttributeError:
                        continue
                if len(sel_lst) == self.nr_dims and comp_slc is None:
                    comp_slc = sel_lst.pop(self.nr_comp)

                if isinstance(comp_slc, string_types):
                    comp_slc = self._COMPONENT_NAMES.index(comp_slc)
                if _isstr:
                    selection = ",".join(sel_lst)
                else:
                    selection = sel_lst
            except TypeError:
                pass

        if comp_slc is None:
            comp_slc = slice(None)

        return selection, comp_slc

    def _finalize_slice(self, slices, crdlst, reduced, comp_slc):
        all_none = (list(slices) == [slice(None)] * len(slices))
        no_sslices = slices is None or all_none
        no_compslice = comp_slc is None or comp_slc == slice(None)

        # no slice necessary, just pass the field through
        if no_sslices and no_compslice:
            return self

        # if we're doing a component slice, and the underlying
        # data is a list/tuple of Field objects, we don't need
        # to do any more work
        src_is_fld_list = (isinstance(self._src_data, (list, tuple)) and
                           all([isinstance(f, Field) for f in self._src_data]))
        if no_sslices and src_is_fld_list:
            if self.post_reshape_transform_func is not None:
                raise NotImplementedError()
            return self._src_data[comp_slc]

        # coord transforms are not copied on purpose
        crds = coordinate.wrap_crds(self._src_crds.crdtype, crdlst)

        # be intelligent here, if we haven't loaded the data and
        # the source is an h5py-like source, we don't have to read
        # the whole field; h5py will deal with the hyperslicing for us
        slced_dat = None

        hypersliceable = getattr(self._src_data, "_hypersliceable", False)
        single_comp_slc = isinstance(comp_slc, (int, np.integer))
        cc = self.iscentered("Cell")

        if self._cache is None and src_is_fld_list:
            if single_comp_slc:
                # this may not work as advertised since slices may
                # not be complete?
                slced_dat = self._src_data[comp_slc][slices]
            else:
                comps = self._src_data[comp_slc]
                slced_dat = [c[slices] for c in comps]
        elif self._cache is None and hypersliceable:
            # we have to flip the slice, meaning: if the array looks like
            # ind     : 0    1     2    3    4    5    6
            # x       : -1.0 -0.5  0.0  0.5  1.0  1.5  2.0
            # -x[::-1]: -2.0 -1.5 -1.0 -0.5  0.0  0.5  1.0
            # if the slice is [-0.5:1.0], the crds figured out the slice
            # indices after doing -x[::-1], so the slice will be [3:7],
            # but in _src_dat, that data lives at indices [0:3]
            # so what we want is _src_data[::-1][3:6], but a more efficient
            # way to do this for large data sets is _src_data[0:3][::-1]
            # We will always read _src_data forward, then flip it since
            # h5py won't do a slice of [3:0:-1]
            first_slc, second_slc = self._src_crds.reflect_slices(slices, cc, False)

            # now put component slice back in
            try:
                nr_comp = self.nr_comp
                first_slc.insert(self.nr_comp, comp_slc)
                if not single_comp_slc:
                    second_slc.insert(self.nr_comp, slice(None))
            except TypeError:
                nr_comp = None

            # this is a bad hack for the fact that fields and the slices
            # have 3 spatial dimensions, but the src_data may have fewer
            _first, _second = first_slc, second_slc
            if len(self._src_data.shape) != len(_first):
                _first, _second = [], []
                j = 0  # trailing index
                it = izip_longest(count(), first_slc, second_slc,
                                  fillvalue=None)
                for i, a, b in islice(it, None, len(first_slc)):
                    if self._src_data.shape[j] == self.shape[i]:
                        _first.append(a)
                        _second.append(b)
                        j += 1

            # ok, now cull out from the second slice
            _second = [s for s in _second if s is not None]

            # only hyperslice _src_data if our slice has the right shape
            if len(self._src_data.shape) == len(_first):
                # do the hyper-slice
                slced_dat = self._src_data[tuple(_first)][tuple(_second)]

                # post-reshape-transform
                if self.post_reshape_transform_func is not None:
                    if cc:
                        target_shape = crds.shape_cc
                    else:
                        target_shape = crds.shape_nc
                    if nr_comp is not None:
                        target_shape.insert(nr_comp, -1)
                    slced_dat = self.post_reshape_transform_func(
                        self, crds, slced_dat.reshape(target_shape),
                        comp_slc=comp_slc, **self.transform_func_kwargs)

        # fallback: either not hypersliceable, or the shapes didn't match up
        if slced_dat is None:
            try:
                slices.insert(self.nr_comp, comp_slc)
            except TypeError:
                pass
            slced_dat = self.data[tuple(slices)]

        if len(reduced) == len(slices) or getattr(slced_dat, 'size', 0) == 1:
            # if we sliced the hell out of the array, just
            # return the value that's left, ndarrays have the same behavior
            ret = slced_dat
        else:
            ctx = dict(crds=crds)
            fldtype = None
            # if we sliced a vector down to one component
            if self.nr_comps is not None:
                if single_comp_slc:
                    comp_name = self._COMPONENT_NAMES[comp_slc]
                    ctx['name'] = self.name + comp_name
                    ctx['pretty_name'] = (self.pretty_name +
                                          "$_{0}$".format(comp_name))
                    fldtype = "Scalar"
            ret = self.wrap(slced_dat, ctx, fldtype=fldtype)

            # if there are reduced dims, put them into the deep_meta dict
            if len(reduced) > 0:
                ret.deep_meta["reduced"] = reduced
        return ret

    def forget_source(self):
        self._src_data = self.data

    def slice(self, selection):
        """ Slice the field using a string like "y=3:6:2,z=0" or a standard
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
        slices, crdlst, reduced = self._src_crds.make_slice(selection, cc=cc)
        return self._finalize_slice(slices, crdlst, reduced, comp_slc)

    def slice_reduce(self, selection):
        """ Slice the field, then go through all dims and look for dimensions
        with only one coordinate. Reduce those dimensions out of the new
        field """
        cc = self.iscentered("Cell")
        selection, comp_slc = self._prepare_slice(selection)
        slices, crdlst, reduced = self._src_crds.make_slice_reduce(selection,
                                                                   cc=cc)
        return self._finalize_slice(slices, crdlst, reduced, comp_slc)

    def slice_and_keep(self, selection):
        """ Slice the field, then go through dimensions that would be reduced
        by a normal numpy slice (like saying 'z=0') and keep those dimensions
        in the new field """
        cc = self.iscentered("Cell")
        selection, comp_slc = self._prepare_slice(selection)
        slices, crdlst, reduced = self._src_crds.make_slice_keep(selection,
                                                                 cc=cc)
        # print("??", type(self._src_crds), crdlst)
        return self._finalize_slice(slices, crdlst, reduced, comp_slc)

    def set_slice(self, selection, value):
        """Used for fld.__setitem__

        NOTE:
            This is only lightly tested
        """
        cc = self.iscentered("Cell")
        selection, comp_slc = self._prepare_slice(selection)
        slices, _ = self._src_crds.make_slice(selection, cc=cc)[:2]
        try:
            slices.insert(self.nr_comp, comp_slc)
        except TypeError:
            pass
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
        return self._src_crds.iter_points(center=center)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        """ unload the data """
        self.unload()
        return None

    def __iter__(self):
        """ iterate though all values in the data, raveled """
        for val in self.data.ravel():
            yield val

    ##################################
    ## Utility methods to get at crds
    # these are the same as something like self._src_crds['xnc']
    # or self._src_crds.get_crd()
    def get_crd(self, axis, shaped=False):
        """ return crd along axis with same centering as field
        axis can be crd name as string, or index, as in x==2, y==1, z==2 """
        return self._src_crds.get_crd(axis, center=self.center, shaped=shaped)

    def get_crd_nc(self, axis, shaped=False):
        """ returns a flat ndarray of coordinates along a given axis
        axis can be crd name as string, or index, as in x==2, y==1, z==2 """
        return self._src_crds.get_nc(axis, shaped=shaped)

    def get_crd_cc(self, axis, shaped=False):
        """ returns a flat ndarray of coordinates along a given axis
        axis can be crd name as string, or index, as in x==2, y==1, z==2 """
        return self._src_crds.get_cc(axis, shaped=shaped)

    def get_crd_ec(self, axis, shaped=False):
        """ returns a flat ndarray of coordinates along a given axis
        axis can be crd name as string, or index, as in x==2, y==1, z==2 """
        return self._src_crds.get_ec(axis, shaped=shaped)

    def get_crd_fc(self, axis, shaped=False):
        """ returns a flat ndarray of coordinates along a given axis
        axis can be crd name as string, or index, as in x==2, y==1, z==2 """
        return self._src_crds.get_fc(axis, shaped=shaped)

    ## these return all crd dimensions
    # these are the same as something like self._src_crds.get_crds()
    def get_crds(self, axes=None, shaped=False):
        """ return all crds as list of ndarrays with same centering as field """
        return self._src_crds.get_crds(axes=axes, center=self.center, shaped=shaped)

    def get_crds_nc(self, axes=None, shaped=False):
        """ returns all node centered coords as a list of ndarrays, flat if
        shaped==False, or shaped if shaped==True """
        return self._src_crds.get_crds_nc(axes=axes, shaped=shaped)

    def get_crds_cc(self, axes=None, shaped=False):
        """ returns all cell centered coords as a list of ndarrays, flat if
        shaped==False, or shaped if shaped==True """
        return self._src_crds.get_crds_cc(axes=axes, shaped=shaped)

    def get_crds_fc(self, axes=None, shaped=False):
        """ returns all face centered coords as a list of ndarrays, flat if
        shaped==False, or shaped if shaped==True """
        return self._src_crds.get_crds_fc(axes=axes, shaped=shaped)

    def get_crds_ec(self, axes=None, shaped=False):
        """ returns all edge centered coords as a list of ndarrays, flat if
        shaped==False, or shaped if shaped==True """
        return self._src_crds.get_crds_ec(axes=axes, shaped=shaped)

    def get_clist(self, *args, **kwargs):
        """I'm not sure anybody should use this since a clist is kind
        of an internal thing used for creating new coordinate instances"""
        return self._src_crds.get_clist(*args, **kwargs)

    def is_spherical(self):
        return self._src_crds.is_spherical()

    ######################
    def shell_copy(self, force=False, **kwargs):
        """Get a field just like this one with a new cache

        So, fields that belong to files are kept around for the
        lifetime of the file bucket, which is probably the lifetime
        of the main script. That means you have to explicitly unload
        to clear the cache (important if reading a timeseries of fields
        > 1GB). This function will return a new field instance with
        references to all the same internals as self, except for the
        cache. Effectively, this turns the parent field into a
        lightweight "field shell", from which a memory intensive field
        can be made on the fly.

        Parameters:
            force(bool): without force, only make a new field if the
                cache is not already loaded. If set to True, a field's
                data could be in ram twice since _cache will be filled
                for the new field when needed. This is probably not what
                you want.
            kwargs: additional keyword arguments to give to the Field
                constructor

        Returns:
            a field as described above (could be self)
        """
        if self._cache is not None and not force:
            return self

        # Note: this is fragile if a subclass takes additional parameters
        # in an overridden __init__; in that case, the developer MUST
        # override shell_copy and pass the extra kwargs in to here.
        f = type(self)(self.name, self._src_crds, self._src_data, center=self.center,
                       time=self.time, meta=self.meta, deep_meta=self.deep_meta,
                       forget_source=False,
                       pretty_name=self.pretty_name,
                       post_reshape_transform_func=self.post_reshape_transform_func,
                       transform_func_kwargs=self.transform_func_kwargs,
                       parents=[self],
                       **kwargs)
        return f

    #####################
    ## convert centering
    # Note: these are kind of specific to cartesian connected grids

    def as_centered(self, center):
        if not center:
            fld = self
        elif center.lower() == "cell":
            fld = self.as_cell_centered()
        elif center.lower() == "node":
            fld = self.as_node_centered()
        else:
            raise NotImplementedError("can only give field as cell or node")
        return fld

    def as_cell_centered(self):
        """Convert field to cell centered field without discarding
        any data; this goes through hacky pains to make sure the data
        is the same as self (including the state of cachedness)"""
        if self.iscentered('cell'):
            return self

        elif self.iscentered('node'):
            # construct new crds
            new_crds = self._src_crds.extend_by_half()

            # this is similar to a shell copy, but it's intimately
            # linked to self as a parent
            f = type(self)(self.name, new_crds, None, center="cell",
                           time=self.time, meta=self.meta,
                           deep_meta=self.deep_meta, forget_source=False,
                           pretty_name=self.pretty_name,
                           post_reshape_transform_func=self.post_reshape_transform_func,
                           transform_func_kwargs=self.transform_func_kwargs,
                           _parent_field=self)
            #FIME: this is such a hack to try our hardest to keep the
            # reference to the data the same
            f._src_data = self._src_data
            f._cache = self._cache

            return f

        else:
            raise NotImplementedError("can't yet move {0} to cell "
                                      "centers".format(self.center))


    def as_node_centered(self):
        """Convert field to node centered field without discarding
        any data;  this goes through hacky pains to make sure the data
        is the same as self (including the state of cachedness)"""
        if self.iscentered('node'):
            return self

        elif self.iscentered('cell'):
            # construct new crds
            # axes = self._src_crds.axes
            # crds_cc = self.get_crds_cc()
            new_clist = self._src_crds.get_clist(center="cell")
            new_crds = type(self._src_crds)(new_clist)

            # this is similar to a shell copy, but it's intimately
            # linked to self as a parent
            f = type(self)(self.name, new_crds, None, center="node",
                           time=self.time, meta=self.meta,
                           deep_meta=self.deep_meta, forget_source=False,
                           pretty_name=self.pretty_name,
                           post_reshape_transform_func=self.post_reshape_transform_func,
                           transform_func_kwargs=self.transform_func_kwargs,
                           _parent_field=self)
            #FIME: this is such a hack to try our hardest to keep the
            # reference to the data the same
            f._src_data = self._src_data
            f._cache = self._cache

            return f

        else:
            raise NotImplementedError("can't yet move {0} to node "
                                      "centers".format(self.center))

    def as_c_contiguous(self):
        """Return a Field with c-contiguous data

        Note:
            If the data is not already in memory (cached), this
            function will trigger a load.

        Returns:
            Field or self
        """
        was_loaded = self.is_loaded()
        ret = None

        if self.data.flags['C_CONTIGUOUS']:
            ret = self
        else:
            ret = self.wrap(np.ascontiguousarray(self.data))

        if not was_loaded and ret is not self:
            self.unload()
        return ret

    #######################
    ## emulate a container

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        return self.slice(item)

    def __setitem__(self, key, value):
        """ just act as if you setitem on underlying data """
        self.set_slice(key, value)

    def __delitem__(self, item):
        """ just act as if you delitem on underlying data, probably raises a
        ValueError """
        self.data.__delitem__(item)

    ##########################
    ## emulate a numeric type

    def __array__(self, dtype=None):
        # dtype = None is ok, datatype won't change
        return np.array(self.data, dtype=dtype, copy=False)

    def wrap(self, arr, context=None, fldtype=None):
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
        if fldtype is None:
            fldtype = type(self)
        else:
            fldtype = field_type(fldtype)
        # Transform functions are intentionally omitted. The idea being that
        # the transform was already applied when creating arr
        return fldtype(name, crds, arr, time=time, center=center,
                   meta=self.meta, deep_meta=context, parents=[self],
                   pretty_name=pretty_name)

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
        return self.wrap(self.data.__and__(other))
    def __xor__(self, other):
        return self.wrap(self.data.__xor__(other))
    def __or__(self, other):
        return self.wrap(self.data.__or__(other))

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

    @property
    def real(self):
        ret = self
        if self.dtype.kind == "c":
            ret = self.wrap(self.data.real)
        return ret

    @property
    def imag(self):
        return self.wrap(self.data.imag)

    def astype(self, dtype):
        ret = self
        if np.dtype(dtype) != self.dtype:
            ret = self.wrap(self.data.astype(dtype))
        return ret


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

        downclist = self._src_crds.get_clist(np.s_[::2])
        downcrds = coordinate.wrap_crds("nonuniform_cartesian", downclist)
        return self.wrap(downdat, {"crds": downcrds})

    def transpose(self, *axes):
        """ same behavior as numpy transpose, alse accessable
        using np.transpose(fld) """
        if axes == (None, ) or len(axes) == 0:
            axes = list(range(self.nr_dims - 1, -1, -1))
        if len(axes) != self.nr_dims:
            raise ValueError("transpose can not change number of axes")
        clist = self._src_crds.get_clist()
        new_clist = [clist[ax] for ax in axes]
        t_crds = coordinate.wrap_crds(self._src_crds.crdtype, new_clist)
        t_data = self.data.transpose(axes)
        return self.wrap(t_data, {"crds": t_crds})

    def swap_axes(self, a, b):
        new_clist = self._src_crds.get_clist()
        new_clist[a], new_clist[b] = new_clist[b], new_clist[a]
        new_crds = coordinate.wrap_crds(self._src_crds.crdtype, new_clist)
        new_data = self.data.swap_axes(a, b)
        return self.wrap(new_data, {"crds": new_crds})

    def as_interlaced(self, force_c_contiguous=True):
        if force_c_contiguous:
            return self.as_c_contiguous()
        else:
            return self


class VectorField(Field):
    _TYPE = "vector"
    _COMPONENT_NAMES = "xyzuvw"

    def component_views(self):
        """ return numpy views to components individually, memory layout
        of the original field is maintained """
        flds = self.component_fields()
        return [f.data for f in flds]

    def component_fields(self):
        if (self.layout == LAYOUT_FLAT and
            isinstance(self._src_data, (list, tuple)) and
            all([isinstance(f, Field) for f in self._src_data])):
            # if all elements are fields
            return self._src_data

        lst = [None] * self.nr_comps
        for i in range(self.nr_comps):
            slc = [slice(None)] * (len(self.shape))
            slc[self.nr_comp] = i
            lst[i] = self[slc]
        return lst

    def as_interlaced(self, force_c_contiguous=True):
        """Get an interlaced version of this field

        Note:
            This will trigger a data load if the data is not already
            in memory

        Args:
            force_c_contiguous: if data is not c contiguous, then wrap
                it in another np.array() call.

        Returns:
            self, or Field if
        """
        was_loaded = self.is_loaded

        ret = None
        if self.layout == LAYOUT_INTERLACED:
            if force_c_contiguous:
                if not self.data.flags['C_CONTIGUOUS']:
                    # print("calling np.ascontiguousarray")
                    ret = self.wrap(np.ascontiguousarray(self.data))
                else:
                    # print("returning self")
                    ret = self
        else:
            ctx = dict(force_layout=LAYOUT_INTERLACED)
            # the data load is going to wrap the array, i think it's
            # redundant to put an "ascontiguousarray" here
            ret = self.wrap(self.data, ctx)

        if not was_loaded and ret is not self:
            self.unload()
        return ret

class MatrixField(Field):
    _TYPE = "matrix"


class TensorField(Field):
    _TYPE = "tensor"


##
## EOF
##
