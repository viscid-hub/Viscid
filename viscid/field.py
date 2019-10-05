# pylint: disable=too-many-lines
"""Fields are the basis of Viscid's data abstration

Fields belong in grids, or by themselves as a result of a calculation.
They can belong to a :class:`Grid` as the result of a file load, or
by themselves as the result of a calculation. This module has some
convenience functions for creating fields similar to `Numpy`.
"""

from __future__ import print_function
from itertools import count, islice
from inspect import isclass
import re

import numpy as np

import viscid
from viscid import logger
from viscid.compat import string_types, izip_longest
from viscid import coordinate
from viscid.cython import interp_trilin
from viscid import sliceutil
from viscid import tree
from viscid.vutil import subclass_spider


LAYOUT_DEFAULT = "none"  # do not translate
LAYOUT_INTERLACED = "interlaced"
LAYOUT_FLAT = "flat"
LAYOUT_SCALAR = "scalar"
LAYOUT_OTHER = "other"

_DEFAULT_COMPONENT_NAMES = "xyzuvw"


__all__ = ['arrays2field', 'dat2field', 'full', 'empty', 'zeros', 'ones',
           'full_like', 'empty_like', 'zeros_like', 'ones_like',
           'mfield', 'mfield_cell', 'mfield_node',
           'scalar_fields_to_vector', 'wrap_field']


def arrays2field(crd_arrs, dat_arr, name="NoName", center=None,
                 crd_type=None, crd_names="xyzuvw"):
    """Turn arrays into fields so they can be used in viscid.plot, etc.

    This is a convenience function that takes care of making coordnates
    and the like. If the default behavior doesn't work for you, you'll
    need to make your own coordnates and call
    :py:func:`viscid.field.wrap_field`.

    Args:
        crd_arrs (list of ndarrays): xyz list of ndarrays that
            describe the node centered coordnates of the field
        dat_arr (ndarray): data with len(crd_arrs) or len(crd_arrs) + 1
            dimensions
        name (str): some name
        center (str, None): If not None, translate field to this
            centering (node or cell)
    """
    crds = coordinate.arrays2crds(crd_arrs, crd_type=crd_type, crd_names=crd_names)

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
    elif len(dat_arr.shape) == len(crds.shape_nc) + 1:
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
        elif layout == LAYOUT_INTERLACED:
            sshape = dat_arr.shape[:-1]
        else:
            raise ValueError("Unknown layout: {0}".format(layout))
    else:
        raise ValueError("Unknown type: {0}".format(fldtype))

    crd_arrs = [np.arange(s).astype(dat_arr.dtype) for s in sshape]
    return arrays2field(crd_arrs, dat_arr, name=name, center=center)

def full(crds, fill_value, dtype="f8", name="NoName", center="cell",
         layout=LAYOUT_FLAT, nr_comps=0, crd_type=None, crd_names="xyzuvw",
         **kwargs):
    """Analogous to `numpy.full`

    Parameters:
        crds (Coordinates, list, or tuple): Can be a coordinates
            object. Can also be a list of ndarrays describing
            coordinate arrays. Or, if it's just a list or tuple of
            integers, those integers are taken to be the nz,ny,nx shape
            and the coordinates will be fill with :py:func:`np.arange`.
        fill_value (number, None): Initial value of array. None
            indicates uninitialized (i.e., `numpy.empty`)
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
        crds = coordinate.arrays2crds(crds, crd_type=crd_type, crd_names=crd_names)

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

    if fill_value is None:
        dat = np.empty(shape, dtype=dtype)
    else:
        if hasattr(np, "full"):
            dat = np.full(shape, fill_value, dtype=dtype)
        elif hasattr(np, "filled"):
            dat = np.filled(shape, fill_value, dtype=dtype)
        else:
            raise RuntimeError("Please update Numpy; your version has neither "
                               "`numpy.full` nor `numpy.filled`")
    return wrap_field(dat, crds, name=name, fldtype=fldtype, center=center,
                      **kwargs)

def empty(crds, dtype="f8", name="NoName", center="cell", layout=LAYOUT_FLAT,
          nr_comps=0, **kwargs):
    """Analogous to `numpy.empty`

    Returns:
        new uninitialized :class:`Field`

    See Also: :meth:`full`
    """
    return full(crds, None, dtype=dtype, name=name, center=center,
                layout=layout, nr_comps=nr_comps, **kwargs)

def zeros(crds, dtype="f8", name="NoName", center="cell", layout=LAYOUT_FLAT,
          nr_comps=0, **kwargs):
    """Analogous to `numpy.zeros`

    Returns:
        new :class:`Field` initialized to 0

    See Also: :meth:`full`
    """
    return full(crds, 0.0, dtype=dtype, name=name, center=center,
                layout=layout, nr_comps=nr_comps, **kwargs)

def ones(crds, dtype="f8", name="NoName", center="cell", layout=LAYOUT_FLAT,
         nr_comps=0, **kwargs):
    """Analogous to `numpy.ones`

    Returns:
        new :class:`Field` initialized to 1

    See Also: :meth:`full`
    """
    return full(crds, 1.0, dtype=dtype, name=name, center=center,
                layout=layout, nr_comps=nr_comps, **kwargs)

def full_like(fld, fill_value, name="NoName", **kwargs):
    """Analogous to `numpy.full_like`

    Makes a new :class:`Field` initialized to fill_value. Copies as
    much meta data as it can from `fld`.

    Parameters:
        fld: field to get coordinates / metadata from
        fill_value (number, None): initial value, or None to leave
            data uninitialized
        name: name for this field
        **kwargs: passed through to :class:`Field` constructor

    Returns:
        new :class:`Field`
    """
    if fill_value is None:
        dat = np.empty(fld.shape, dtype=fld.dtype)
    else:
        if hasattr(np, "full"):
            dat = np.full(fld.shape, fill_value, dtype=fld.dtype)
        elif hasattr(np, "filled"):
            dat = np.filled(fld.shape, fill_value, dtype=fld.dtype)
        else:
            raise RuntimeError("Please update Numpy; your version has neither "
                               "`numpy.full` nor `numpy.filled`")
    c = kwargs.pop("center", fld.center)
    t = kwargs.pop("time", fld.time)
    return wrap_field(dat, fld.crds, name=name, fldtype=fld.fldtype, center=c,
                      time=t, parents=[fld], **kwargs)

def empty_like(fld, **kwargs):
    """Analogous to `numpy.empty_like`

    Returns:
        new uninitialized :class:`Field`

    See Also: :meth:`full_like`
    """
    return full_like(fld, None, **kwargs)

def zeros_like(fld, **kwargs):
    """Analogous to `numpy.zeros_like`

    Returns:
        new :class:`Field` filled with zeros

    See Also: :meth:`full_like`
    """
    return full_like(fld, 0.0, **kwargs)

def ones_like(fld, **kwargs):
    """Analogous to `numpy.ones_like`

    Returns:
        new :class:`Field` filled with ones

    See Also: :meth:`full_like`
    """
    return full_like(fld, 1.0, **kwargs)


class _mfield_factory(object):
    """This mimics Numpy's mgrid and ogrid functionality"""
    def __init__(self, center='node', dtype=np.float64):
        self.center = center
        self.dtype = dtype

    def __getitem__(self, slc):
        crd_arrs = [arr.reshape(-1) for arr in np.ogrid[slc]]
        return zeros(crd_arrs, center=self.center, dtype=self.dtype)


mfield_cell = _mfield_factory(center='cell', dtype=np.float64)
mfield_node = _mfield_factory(center='node', dtype=np.float64)
mfield = mfield_node

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

    # component fields will already be transposed when filling caches, so
    # the source will already be xyz
    if "zyx_native" not in kwargs:
        kwargs["zyx_native"] = False
    else:
        logger.warning("did you really want to do another transpose?")

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
        for cls in subclass_spider(Field):
            if cls.istype(fldtype):
                return cls
    logger.error("Field type {0} not understood".format(fldtype))
    return None

def wrap_field(data, crds, name="NoName", fldtype="scalar", center='node',
               **kwargs):
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
    # try to auto-detect vector fields
    try:
        if (center.strip().lower() == 'node' and
            data.size % np.prod(crds.shape_nc) == 0 and
            data.size // np.prod(crds.shape_nc) > 1):
            fldtype = "vector"
        elif (center.strip().lower() == 'cell' and
              data.size % np.prod(crds.shape_cc) == 0 and
              data.size // np.prod(crds.shape_cc) > 1):
            fldtype = "vector"
    except AttributeError:
        pass

    cls = field_type(fldtype)
    if cls is not None:
        return cls(name, crds, data, center=center, **kwargs)
    else:
        raise NotImplementedError("can not decipher field")

def rewrap_field(fld):
    # zyx_native is false b/c we're using fld.data
    ret = type(fld)(fld.name, fld.crds, fld.data, center=fld.center,
                    forget_source=True, _copy=True, zyx_native=False,
                    parents=[fld])
    return ret


class _FldSlcProxy(object):
    parent = None
    def __init__(self, parent, do_floatify=True):
        self.parent = parent
        self.do_floatify = do_floatify

    def _floatify(self, item):
        if isinstance(item, string_types):
            item = item.strip().lower()
            dt_re = r"(\s*{0}\s*)".format(viscid.sliceutil.RE_DTIME_SLC_GROUP)
            datetimes = re.findall(dt_re, item)
            item = re.sub(dt_re, '__DATETIME__', item)
            item = re.sub(r'([\d\.e]+(?![\d\.FfJj]))', r'\1j', item)
            item_split = item.split('__DATETIME__')
            item = [None] * (len(item_split) + len(datetimes))
            item[0::2] = item_split
            item[1::2] = datetimes
            item = ''.join(item)

        if item in (Ellipsis, '...'):
            item = Ellipsis
        elif item in (np.newaxis, ):
            item = np.newaxis
        elif item in (None, 'none'):
            item = None
        elif isinstance(item, (np.ndarray,)):
            if self.do_floatify:
                item = 1j * item
        elif isinstance(item, string_types):
            pass
        elif isinstance(item, (list,)):
            for i, _ in enumerate(item):
                # try:
                if self.do_floatify:
                    item[i] = 1j * float(item[i])
                # except (ValueError, TypeError):
                #     pass
        else:
            if self.do_floatify:
                item = 1j * float(item)
        return item

    def _xform(self, item):
        if not isinstance(item, tuple):
            item = (item, )
        sel = []
        for it in item:
            if isinstance(it, slice):
                start = self._floatify(it.start)
                stop = self._floatify(it.stop)
                step = it.step
                sel.append(slice(start, stop, step))
            else:
                sel.append(self._floatify(it))

        if self.parent.nr_comps and len(sel) >= self.parent.nr_comp:
            sel.insert(self.parent.nr_comp, slice(None))

        return tuple(sel)

    def __getitem__(self, item):
        return self.parent.__getitem__(self._xform(item))

    def __setitem__(self, item, val):
        return self.parent.__setitem__(self._xform(item), val)


class Field(tree.Leaf):
    _TYPE = "none"
    _CENTERING = ['node', 'cell', 'grid', 'face', 'edge']

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
    defer_wrapping = False

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
    _cached_xyz_src_view = None

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
        # if name == "b":
        #     print("initing b")
        #     import pdb; pdb.set_trace()
        super(Field, self).__init__(name, time, info, parents)
        self.center = center
        self._src_crds = crds
        self.data = data

        self.comp_names = ''

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

        self.meta = {} if meta is None else meta.copy()
        self.deep_meta = {} if deep_meta is None else deep_meta.copy()
        for k, v in kwargs.items():
            if k.startswith("_"):
                self.deep_meta[k[1:]] = v
            else:
                self.meta[k] = v

        if "force_layout" not in self.deep_meta:
            if "force_layout" in self.meta:
                logger.warning("deprecated force_layout syntax: kwarg should "
                               "be given as _force_layout")
                self.deep_meta["force_layout"] = self.meta["force_layout"]
            else:
                self.deep_meta["force_layout"] = LAYOUT_DEFAULT
        self.deep_meta["force_layout"] = self.deep_meta["force_layout"].lower()

        if "copy" not in self.deep_meta:
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
        """ UNTESTED, clear cache if layout changes """
        new_layout = new_layout.lower()
        if self.is_loaded:
            current_layout = self.layout
            if new_layout != current_layout:
                self.clear_cache()
        self.deep_meta["force_layout"] = new_layout
        self._layout = None

    @property
    def nr_dims(self):
        """ returns number of dims, this should be the number of dims of
        the underlying data, but is assumed a priori so no data load is
        done here """
        return self.nr_sdims + 1

    @property
    def ndim(self):
        return self.nr_dims

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
        s = list(self.sshape)
        try:
            s.insert(self.nr_comp, self.nr_comps)
        except TypeError:
            pass
        return tuple(s)

    @property
    def native_shape(self):
        """ returns the shape of the underlying data, does not explicitly load
        the data """
        s = list(self.native_sshape)
        try:
            s.insert(self.nr_comp, self.nr_comps)
        except TypeError:
            pass
        return tuple(s)

    @property
    def sshape(self):
        """ shape of spatial dimensions, does not include comps, and is not
        the same shape as underlying array, does not load the data """
        # it is enforced that the cached data has a shape that agrees with
        # the coords by _reshape_ndarray_to_crds... actually, that method
        # requires this method to depend on the crd shape
        if self.iscentered("node"):
            return tuple(self._src_crds.shape_nc)
        elif self.iscentered("cell"):
            return tuple(self._src_crds.shape_cc)
        else:
            # logger.warning("edge/face vectors not implemented, assuming "
            #                "cell shape")
            return tuple(self._src_crds.shape_cc)

    @property
    def native_sshape(self):
        sshape = self.sshape
        if self.meta.get("zyx_native", False):
            sshape = sshape[::-1]
        return tuple(sshape)

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
        to empty the cache later you can always use clear_cache """
        if self._cache is None:
            self._fill_cache()
        return self._cache
    @data.setter
    def data(self, dat):
        # clean up
        self.clear_cache()
        self._layout = None
        self._nr_comp = None
        self._nr_comps = None
        self._dtype = None
        self._src_data = dat
        if self.meta and self.meta.get('zyx_native', False):
            self.meta['zyx_native'] = False
        # self._translate_src_data()  # um, what's this for? looks dangerous
        # do some sort of lazy pre-setup _src_data inspection?

    @property
    def flat_data(self):
        return self.data.reshape(-1)

    @property
    def patches(self):
        return [self]

    @property
    def nr_patches(self):  # pylint: disable=no-self-use
        return 1

    @property
    def xl(self):
        return self.crds.xl_nc

    @property
    def xh(self):
        return self.crds.xh_nc

    def get_slice_extent(self, selection):
        extent = self.patches[0]._src_crds.get_slice_extent(selection)
        for i in range(3):
            if np.isnan(extent[0, i]):
                extent[0, i] = self.xl[i]
            if np.isnan(extent[1, i]):
                extent[1, i] = self.xh[i]
        return extent

    @property
    def is_loaded(self):
        return self._cache is not None

    def clear_cache(self):
        """ does not guarentee that the memory will be freed """
        self._cache = None
        self._cached_xyz_src_view = None
        if self._parent_field is not None:
            self._parent_field._cache = self._cache
            self._parent_field._cached_xyz_src_view = self._cached_xyz_src_view

    def _fill_cache(self):
        """ actually load data into the cache """
        self._cache = self._src_data_to_ndarray()
        if self._parent_field is not None:
            self._parent_field._cache = self._cache
            # self._parent_field._cached_xyz_src_view = self._cached_xyz_src_view

    def resolve(self):
        """Resolve all pending actions on a field like translations etc"""
        if self._cache is None:
            self._fill_cache()
        return self

    # um, what was this for? looks dangerous
    # def _translate_src_data(self):
    #     pass

    # @staticmethod
    # def _zyx_transpose(dat):
    #     if isinstance(dat, Field):
    #         pass
    #     else:
    #         return dat.T

    @property
    def _xyz_src_data(self):
        """Return a view of _src_data that is xyz ordered

        Note that this will always cause the cache to be filled
        """
        if self._cached_xyz_src_view is None:
            if self.meta.get("zyx_native", False):
                if isinstance(self._src_data, (list, tuple)):
                    # lst = [np.array(d).T for d in self._src_data]
                    # self._cached_xyz_src_view = lst

                    lst = []
                    for d in self._src_data:
                        lst.append(np.array(d).T)
                        # Slight memory hack: clear cache on src data, there
                        # probably exists a better way to be more lazy
                        if hasattr(d, "clear_cache"):
                            d.clear_cache()
                    self._cached_xyz_src_view = lst
                else:
                    # spatial_transpose = list(range(len(self.shape)))
                    spatial_transpose = list(range(len(self._src_data.shape)))
                    if self.nr_comps > 0:
                        nr_comp = self.nr_comp
                        spatial_transpose.remove(nr_comp)
                        spatial_transpose = spatial_transpose[::-1]
                        spatial_transpose.insert(nr_comp, nr_comp)
                    else:
                        spatial_transpose = spatial_transpose[::-1]
                    # transposed view
                    Tview = np.transpose(self._src_data.__array__(),
                                         spatial_transpose)
                    self._cached_xyz_src_view = np.array(Tview)

            else:
                self._cached_xyz_src_view = self._src_data
        return self._cached_xyz_src_view

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
            return self._dat_to_ndarray(self._xyz_src_data)

        # if layout is found to be other, i cant do anything with that
        elif src_data_layout == LAYOUT_OTHER:
            logger.warning("Cannot auto-detect layout; not translating; "
                           "performance may suffer")
            return self._dat_to_ndarray(self._xyz_src_data)

        # ok, we demand FLAT arrays, make it so
        elif force_layout == LAYOUT_FLAT:
            if src_data_layout != LAYOUT_INTERLACED:
                raise RuntimeError("I should not be here")

            nr_comps = self.nr_comps
            data_dest = np.empty(self.shape, dtype=self.dtype)
            for i in range(nr_comps):
                # NOTE: I wonder if this is the fastest way to reorder
                data_dest[i, ...] = self._xyz_src_data[..., i]
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
                data_dest[..., i] = self._xyz_src_data[i]

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
            arrfunc = np.array
            if isinstance(dat, np.ma.core.MaskedArray):
                arrfunc = np.ma.array
            arr = arrfunc(dat, dtype=dat.dtype.name, copy=self.deep_meta["copy"])
        elif isinstance(dat, (list, tuple)):
            dt = dat[0].dtype.name
            # tmp = [np.array(d, dtype=dt, copy=self.deep_meta["copy"]) for d in dat]
            tmp = []
            for d in dat:
                tmp.append(np.array(d, dtype=dt, copy=self.deep_meta["copy"]))
                # Slight memory hack: clear cache on src data, there
                # probably exists a better way to be more lazy
                if hasattr(d, "clear_cache"):
                    d.clear_cache()
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

    def _detect_layout(self, dat, native=True):
        """ returns LAYOUT_XXX, this just looks at len(dat) or dat.shape
        Note:
            dat should be in 'native layout', meaning zyx is _src_data is
            zyx, or xyz if _src_data is xyz
        """
        # if native, then compare dat's shape against the native shape
        if native:
            sshape = list(self.native_sshape)
        else:
            sshape = list(self.sshape)

        # if i receive a list, then i suppose i have a list of
        # arrays, one for each component... this is a flat layout
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

        dat_shape = list(dat.shape)

        if dat_shape == sshape:
            return LAYOUT_SCALAR

        # if the crds shape has more values than the dat.shape
        # then try trimming the directions that have 1 element
        # this can happen when crds are 3d, but field is only 2d
        ## I'm not sure why this needs to happen... but tests fail
        ## without it
        while len(sshape) > len(dat_shape) - 1:
            try:
                sshape.remove(1)
            except ValueError:
                break

        # check which dims match the shape of the crds
        # or if they match when disregarding length 1 axes.
        # This 2nd part happens when calling atleast_3d() on
        # a field that isn't already in memory
        dat_shape2 = [si for si in dat_shape if si > 1]
        sshape2 = [si for si in dat_shape if si > 1]

        if dat_shape == sshape:
            layout = LAYOUT_SCALAR
        elif dat_shape[1:] == sshape:
            layout = LAYOUT_FLAT
        elif dat_shape[:-1] == sshape or dat_shape2[:-1] == sshape2:
            layout = LAYOUT_INTERLACED
        elif dat_shape[0] == np.prod(sshape):
            layout = LAYOUT_INTERLACED
        elif dat_shape[-1] == np.prod(sshape):
            layout = LAYOUT_FLAT
        # the following are layouts that happen after a call to atleast_3d()
        elif dat_shape2 == sshape2:
            layout = LAYOUT_SCALAR
        elif dat_shape2[1:] == sshape2:
            layout = LAYOUT_FLAT
        elif dat_shape2[:-1] == sshape2:
            layout = LAYOUT_INTERLACED
        else:
            # if this happens, don't ignore it even if it happens to work
            # print("??", self, self.native_shape, self.shape, )
            # import pdb; pdb.set_trace()
            logger.error("could not detect layout for '{0}': shape = {1} "
                         "target shape = {2}"
                         "".format(self.name, dat_shape, sshape))
            layout = LAYOUT_OTHER

        return layout

    def _prepare_slice(self, selection):
        """ if selection has a slice for component dimension, set it aside """
        sel_list = sliceutil.raw_sel2sel_list(selection)
        sel_list, comp_slc = sliceutil.prune_comp_sel(sel_list, self.comp_names)

        if self.layout == LAYOUT_SCALAR:
            assert comp_slc == slice(None)

        return sel_list, comp_slc

    def _finalize_slice(self, slices, crdlst, reduced, crd_type, comp_slc):
        # if self.name == "b":
        #     import pdb; pdb.set_trace()
        all_none = sliceutil.all_slices_none(slices)
        no_sslices = slices is None or all_none
        no_compslice = comp_slc is None or sliceutil.all_slices_none([comp_slc])

        # no slice necessary, just pass the field through
        if no_sslices and no_compslice:
            return self

        # if we're doing a component slice, and the underlying
        # data is a list/tuple of Field objects, we don't need
        # to do any more work
        src_is_fld_list = (isinstance(self._src_data, (list, tuple)) and
                           all([isinstance(f, Field) for f in self._src_data]))
        if self._cache is None and no_sslices and src_is_fld_list:
            if self.post_reshape_transform_func is not None:
                raise NotImplementedError()
            return self._src_data[comp_slc]

        # IMPORTANT NOTE: from this point on, crds of the result will be
        # "transformed", and the data will be xyz ordered. This is the end
        # of the line for lazily keeping track of zyx native arrays b/c
        # the slice will have to load the data

        # coord transforms are not copied on purpose
        cunits = self._src_crds.get_units((c[0] for c in crdlst), allow_invalid=1)
        crds = coordinate.wrap_crds(crd_type, crdlst, units=cunits,
                                    full_arrays=True, quiet_init=True,
                                    **self._src_crds.meta)

        # be intelligent here, if we haven't loaded the data and
        # the source is an h5py-like source, we don't have to read
        # the whole field; h5py will deal with the hyperslicing for us
        slced_dat = None

        hypersliceable = getattr(self._src_data, "_hypersliceable", False)
        single_comp_slc = hasattr(comp_slc, "__index__")
        cc = (self.iscentered("Cell") or self.iscentered("Face") or
              self.iscentered("Edge"))

        zyx_native = self.meta.get("zyx_native", False)
        # if zyx_native:
        #     native_slices = slices[::-1]
        # else:
        #     native_slices = slices

        if self._cache is None and src_is_fld_list:
            if single_comp_slc:
                # this may not work as advertised since slices may
                # not be complete?
                # using xyz slices because it's calling Field.__getitem__
                slced_dat = self._src_data[comp_slc][tuple(slices)]
            else:
                comps = self._src_data[comp_slc]
                # using xyz slices because it's calling Field.__getitem__
                slced_dat = [c[tuple(slices)] for c in comps]
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

            # using xyz slices b/c that's the order of the crds
            first_slc, second_slc = self._src_crds.reflect_slices(slices,
                                                                  cc, False)

            if zyx_native:
                native_first_slc = first_slc[::-1]
                native_second_slc = second_slc[::-1]
            else:
                native_first_slc = first_slc
                native_second_slc = second_slc

            # now put component slice back in
            try:
                if self.nr_comp == 0:
                    nr_comp = 0
                else:
                    nr_comp = len(first_slc)
                first_slc.insert(nr_comp, comp_slc)
                native_first_slc.insert(nr_comp, comp_slc)
                if not single_comp_slc:
                    second_slc.insert(nr_comp, slice(None))
                    native_second_slc.insert(nr_comp, slice(None))
            except TypeError:
                nr_comp = None

            # this is a bad hack for the fact that fields and the slices
            # have 3 spatial dimensions, but the src_data may have fewer
            _first, _second = native_first_slc, native_second_slc
            if len(self._src_data.shape) != len(_first):
                _first, _second = [], []
                j = 0  # trailing index
                it = izip_longest(count(), native_first_slc, native_second_slc,
                                  fillvalue=None)
                for i, a, b in islice(it, None, len(first_slc)):
                    if self._src_data.shape[j] == self.native_shape[i]:
                        _first.append(a)
                        _second.append(b)
                        j += 1

            # ok, now cull out from the second slice
            _second = [s for s in _second if s is not None]

            # only hyperslice _src_data if our slice has the right shape
            if len(self._src_data.shape) == len(_first):
                # do the hyper-slice
                # these sliced have to be native b/c they're slicing the
                # _src_data directly which means the data could be stored
                # zyx in a lazy hdf5 file
                slced_dat = self._src_data[tuple(_first)][tuple(_second)]
                if zyx_native:
                    if comp_slc is not None and self.nr_comps and not single_comp_slc:
                        _n = len(slced_dat.shape)
                        if self.nr_comp == 0:
                            _t_axes = [0] + list(range(1, _n))[::-1]
                        else:
                            # this is slightly fragile if they layout isn't
                            # strictly interlaced, but that would be rather
                            # pathalogical
                            _t_axes = list(range(_n - 1))[::-1] + [_n - 1]
                        slced_dat = np.array(np.transpose(slced_dat, _t_axes))
                    else:
                        slced_dat = np.array(slced_dat.T)

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
                if self.nr_comp == 0:
                    nr_comp = 0
                else:
                    nr_comp = len(slices)
                slices.insert(nr_comp, comp_slc)
            except TypeError:
                nr_comp = None
            # use xyz slices cause self.data is guarenteed to be xyz
            slced_dat = self.data[tuple(slices)]

        if len(reduced) == len(slices) or getattr(slced_dat, 'size', 0) == 1:
            # if we sliced the hell out of the array, just
            # return the value that's left, ndarrays have the same behavior
            ret = slced_dat.item()
        else:
            ctx = dict(crds=crds, zyx_native=False)
            fldtype = None
            # if we sliced a vector down to one component
            if self.nr_comps:
                if single_comp_slc:
                    ############
                    comp_name = self.comp_names[comp_slc]
                    ctx['name'] = self.name + comp_name
                    ctx['pretty_name'] = (self.pretty_name +
                                          "$_{0}$".format(comp_name))
                    fldtype = "Scalar"
            ret = self.wrap(slced_dat, ctx, fldtype=fldtype)

            # if there are reduced dims, put them into the deep_meta dict
            if len(reduced) > 0:
                ret.meta["reduced"] = reduced
        return ret

    def forget_source(self):
        self._src_data = self.data
        self._cached_xyz_src_view = self.data

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
        cc = self.iscentered("Cell") or self.iscentered("Face") or self.iscentered("Edge")
        sel_list, comp_slc = self._prepare_slice(selection)
        crd_slice_info = self._src_crds.make_slice(tuple(sel_list), cc=cc)
        slices, crdlst, reduced, crd_type = crd_slice_info
        return self._finalize_slice(slices, crdlst, reduced, crd_type, comp_slc)

    def slice_and_reduce(self, selection):
        """ Slice the field, then go through all dims and look for dimensions
        with only one coordinate. Reduce those dimensions out of the new
        field """
        cc = self.iscentered("Cell") or self.iscentered("Face") or self.iscentered("Edge")
        sel_list, comp_slc = self._prepare_slice(selection)
        crd_slice_info = self._src_crds.make_slice_reduce(tuple(sel_list), cc=cc)
        slices, crdlst, reduced, crd_type = crd_slice_info
        return self._finalize_slice(slices, crdlst, reduced, crd_type, comp_slc)

    def slice_and_keep(self, selection):
        """ Slice the field, then go through dimensions that would be reduced
        by a normal numpy slice (like saying 'z=0') and keep those dimensions
        in the new field """
        cc = self.iscentered("Cell") or self.iscentered("Face") or self.iscentered("Edge")
        sel_list, comp_slc = self._prepare_slice(selection)
        crd_slice_info = self._src_crds.make_slice_keep(tuple(sel_list), cc=cc)
        slices, crdlst, reduced, crd_type = crd_slice_info
        # print("??", type(self._src_crds), crdlst)
        return self._finalize_slice(slices, crdlst, reduced, crd_type, comp_slc)

    slice_reduce = slice_and_reduce
    slice_keep = slice_and_keep

    @property
    def loc(self):
        """Easily slice by value (flaot), like in pandas

        Eample:
            >>> subset1 = field["13j", "14j":]
            >>> subset2 = field.loc[13, 14.0:]
            >>> # subset1 and subset2 should be identical
        """
        return _FldSlcProxy(self, do_floatify=True)

    @property
    def iloc(self):
        """Easily slice by value (flaot), like in pandas

        Eample:
            >>> subset1 = field["13j", "14j":]
            >>> subset2 = field.loc[13, 14.0:]
            >>> # subset1 and subset2 should be identical
        """
        return _FldSlcProxy(self, do_floatify=False)

    def interpolated_slice(self, selection):
        seeds = self.crds.slice_interp(selection, cc=self.iscentered('cell'))
        seed_pts = seeds.get_points(center=self.center)
        fld_dat = interp_trilin(self, seed_pts, wrap=False)
        new_fld = self.wrap(fld_dat, context=dict(crds=seeds))
        new_fld = new_fld.slice_reduce("")
        return new_fld

    def set_slice(self, selection, value):
        """Used for fld.__setitem__

        NOTE:
            This is only lightly tested
        """
        cc = (self.iscentered("Cell") or self.iscentered("Face") or
              self.iscentered("Edge"))
        sel_list, comp_slc = self._prepare_slice(selection)
        slices, _ = self._src_crds.make_slice(tuple(sel_list), cc=cc)[:2]
        try:
            nr_comp = self.nr_comp
            nr_comp += slices[:nr_comp].count(np.newaxis)
            slices.insert(nr_comp, comp_slc)
        except TypeError:
            pass
        self.data[tuple(slices)] = value
        return None

    @classmethod
    def istype(cls, type_str):  # noqa
        return cls._TYPE == type_str.lower()

    def iscentered(self, center_str):
        return self.center == center_str.lower()

    def to_seeds(self):
        return viscid.to_seeds(self.get_points())

    @property
    def nr_points(self):
        return self.get_nr_points()

    def get_nr_points(self, center=None, **kwargs):  # pylint: disable=unused-argument
        if center is None:
            center = self.center
        return self._src_crds.get_nr_points(center=center)

    def points(self, center=None, **kwargs):
        return self.get_points(center=center, **kwargs)

    def get_points(self, center=None, **kwargs):  # pylint: disable=unused-argument
        if center is None:
            center = self.center
        return self._src_crds.get_points(center=center)

    def as_mesh(self, center=None, **kwargs):  # pylint: disable=unused-argument
        if center is None:
            center = self.center
        return self._src_crds.as_mesh(center=center)

    def iter_points(self, center=None, **kwargs):  # pylint: disable=W0613
        if center is None:
            center = self.center
        return self._src_crds.iter_points(center=center)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        """clear cached data"""
        self.clear_cache()
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
        # FIXME: work out how get_crd should act on face/edge fields since
        #        in this case there are 3 possibilities
        center = 'cell' if self.center in ('face', 'edge') else self.center
        return self._src_crds.get_crd(axis, center=center, shaped=shaped)

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
    def get_crds_vector(self, axes=None, shaped=False):
        """ returns all face centered coords as a list of ndarrays, flat if
        shaped==False, or shaped if shaped==True """
        center = self.center.lower()
        if center == 'face':
            return self.get_crds_fc(axes=axes, shaped=shaped)
        elif center == 'edge':
            return self.get_crds_ec(axes=axes, shaped=shaped)
        else:
            if center == 'cell':
                c = self._src_crds.get_crds_cc(axes=axes, shaped=shaped)
            elif center == 'node':
                c = self._src_crds.get_crds_nc(axes=axes, shaped=shaped)
            else:
                raise RuntimeError()
            c = [[ci] * 3 for ci in c]
            return c

    def get_crds(self, axes=None, shaped=False):
        """ return all crds as list of ndarrays with same centering as field """
        # FIXME: work out how get_crds should act on face/edge fields since
        #        in this case there are 3 possibilities
        center = 'cell' if self.center in ('face', 'edge') else self.center
        return self._src_crds.get_crds(axes=axes, center=center, shaped=shaped)

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

    def meshgrid(self, axes=None, prune=False):
        crds = list(self.get_crds(axes=axes, shaped=True))
        if prune:
            poplist = []
            for i, nxi in reversed(list(enumerate(self.shape))):
                if nxi <= 1:
                    poplist.append(i)
                    crds.pop(i)
            for pi in poplist:
                for i in range(len(crds)):
                    slc = [slice(None)] * len(crds[i].shape)
                    slc[pi] = 0
                    crds[i] = crds[i][tuple(slc)]
        return np.broadcast_arrays(*crds)

    def meshgrid_flat(self, axes=None, prune=False):
        arrs = self.meshgrid(axes=axes, prune=prune)
        return [a.reshape(-1) for a in arrs]

    ######################
    def shell_copy(self, force=False, crds=None, **kwargs):
        """Get a field just like this one with a new cache

        So, fields that belong to files are kept around for the
        lifetime of the file bucket, which is probably the lifetime
        of the main script. That means you have to explicitly clear the
        cache (important if reading a timeseries of fields > 1GB).
        This function will return a new field instance with
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
        if crds is None and self._cache is not None and not force:
            return self

        if crds is None:
            crds = self._src_crds

        # Note: this is fragile if a subclass takes additional parameters
        # in an overridden __init__; in that case, the developer MUST
        # override shell_copy and pass the extra kwargs in to here.
        # note, zyx_native of the child should be the same as self since we're
        # passing src_data here
        f = type(self)(self.name, crds, self._src_data, center=self.center,
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

    def as_centered(self, center, default_width=1e-5):
        if not center or self.iscentered(center):
            fld = self
        elif center.lower() == "cell":
            fld = self.as_cell_centered(default_width=default_width)
        elif center.lower() == "node":
            fld = self.as_node_centered()
        else:
            raise NotImplementedError("can only give field as cell or node")
        return fld

    def as_cell_centered(self, default_width=1e-5):
        """Convert field to cell centered field without discarding
        any data; this goes through hacky pains to make sure the data
        is the same as self (including the state of cachedness)"""
        if self.iscentered('cell'):
            return self

        elif self.iscentered('node'):
            # construct new crds
            new_crds = self._src_crds.nc2cc(default_width=default_width)

            # this is similar to a shell copy, but it's intimately
            # linked to self as a parent
            f = type(self)(self.name, new_crds, None, center="cell",
                           time=self.time, meta=self.meta,
                           deep_meta=self.deep_meta, forget_source=False,
                           pretty_name=self.pretty_name,
                           post_reshape_transform_func=self.post_reshape_transform_func,
                           transform_func_kwargs=self.transform_func_kwargs,
                           _parent_field=self)
            # FIME: this is such a hack to try our hardest to keep the
            # reference to the data the same
            f._src_data = self._src_data
            f._cache = self._cache

            return f

        elif not self.nr_comps:
            viscid.logger.warning("Converting ECFC scalar to cell center has no "
                                  "effect on data. Only coordinates have changed. "
                                  "The result will appear staggered strangely.")
            return self.wrap_field(self.data, name=self.name, center='cell')

        elif self.iscentered('face'):
            return viscid.fc2cc(self)

        elif self.iscentered('edge'):
            return viscid.ec2cc(self)

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
            # FIXME: this is such a hack to try our hardest to keep the
            # reference to the data the same
            f._src_data = self._src_data
            f._cached_xyz_src_view = self._cached_xyz_src_view
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
        was_loaded = self.is_loaded
        ret = None

        if self.data.flags['C_CONTIGUOUS']:
            ret = self
        else:
            ret = self.wrap(np.ascontiguousarray(self.data))

        if not was_loaded and ret is not self:
            self.clear_cache()
        return ret

    def atleast_3d(self, xl=-1, xh=1):
        """return self, but with nr_sdims >= 3"""
        nr_sdims = self.nr_sdims

        if nr_sdims >= 3:
            return self
        elif nr_sdims < 3:
            _cc = (self.iscentered('cell') or self.iscentered('face') or
                   self.iscentered('edge'))
            newcrds = self.crds.atleast_3d(xl, xh, cc=_cc)
            ctx = {'crds': newcrds}

            if self.is_loaded:
                if _cc:
                    new_shape = newcrds.shape_cc
                else:
                    new_shape = newcrds.shape_nc
                if self.nr_comps:
                    new_shape = list(new_shape)
                    new_shape.insert(self.nr_comp, self.nr_comps)

                return self.wrap(self.data.reshape(new_shape), context=ctx)
            else:
                return self.wrap(self._src_data, context=ctx)

    def atmost_nd(self, n):
        axes = list(self.crds.axes)
        removed_axes = []
        remaining_axes = list(axes)
        slc = []

        for i, ni in enumerate(self.sshape):
            if ni == 1:
                slc.append('{0}=0'.format(axes[i]))
                removed_axes.append(axes[i])
                remaining_axes.remove(axes[i])
            if len(self.sshape) - len(slc) <= n:
                break

        while len(self.sshape) - len(slc) > n:
            slc.append('{0}=0j'.format(remaining_axes[-1]))
            removed_axes.append(remaining_axes[-1])
            remaining_axes.pop(-1)

        if not slc:
            return self
        else:
            return self[','.join(slc)]

    def adjust_crds(self, adjustments, name_map=None):
        """Return shell copy with adjusted coordinates

        Args:
            adjustments (dict, number): Value to scale all coordinates
                by, or dict of numbers or functions to adjust specific
                axes separately. Functions should take a single argument,
                the coordinates as a :py:class:`numpy.ndarray`.
            name_map (dict, None): map to change crd names

        Returns:
            Field: Shell copy of self with adjusted coordinates

        Example:
            >>> fld = viscid.zeros([16, 24])
            >>> fld.adjust_crds({'x': 2.0, 'y': lambda x: 0.5 * x})
        """
        if not isinstance(adjustments, dict):
            adj = adjustments
            adjustments = {}
            for ax in self.crds.axes:
                adjustments[ax] = adj

        axes = self.crds.axes
        crd_arrs_nc = self.get_crds_nc()
        # crd_arrs_cc = self.get_crds_cc()

        for i, ax, arr_nc in zip(count(), axes, crd_arrs_nc):
            if ax in adjustments:
                adj = adjustments[ax]
                if hasattr(adj, '__call__'):
                    crd_arrs_nc[i] = adj(arr_nc)
                    # crd_arrs_cc[i] = adj(arr_cc)
                else:
                    crd_arrs_nc[i] = adj * arr_nc
                    # crd_arrs_cc[i] = adj * arr_cc

        if name_map:
            for _a, _b in name_map.items():
                if _a in axes:
                    axes[axes.index(_a)] = _b

        crd_type = self.crds._TYPE
        crd_type = crd_type.replace('nonuniform', 'AUTO')
        crd_type = crd_type.replace('uniform', 'AUTO')
        new_crds = viscid.arrays2crds(crd_arrs_nc, crd_type=crd_type, crd_names=axes)
        ret = self.shell_copy(crds=new_crds)

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

    def wrap(self, arr, context=None, fldtype=None, npkwargs=None, other=None):
        """ arr is the data to wrap... context is exta deep_meta to pass
        to the constructor. The return is just a number if arr is a
        1 element ndarray, this is for ufuncs that reduce to a scalar """

        if self.defer_wrapping and hasattr(other, "wrap"):
            return other.wrap(arr, context=context, fldtype=fldtype,
                              npkwargs=npkwargs)

        if arr is NotImplemented:
            return NotImplemented
        # if just 1 number wrappen in an array, unpack the value and
        # return it... this is more ufuncy behavior
        try:
            if arr.size == 1:
                return np.ravel(arr)[0]
        except AttributeError:
            pass

        if context is None:
            context = {}
        context = dict(context)
        name = context.pop("name", self.name)
        pretty_name = context.pop("pretty_name", self.pretty_name)
        crds = context.pop("crds", self.crds)
        center = context.pop("center", self.center).lower()
        time = context.pop("time", self.time)
        # should it always return the same type as self?

        # hack for reduction operations (ops that have npkwargs['axis'])
        defer_wrapping = self.defer_wrapping
        if npkwargs:
            axis = npkwargs.get('axis', None)
            if axis is not None:
                if self.nr_comps > 0 and axis == self.nr_comp:
                    # reducing vector -> scalar, no need to play with crds
                    pass
                else:
                    reduce_axis = crds.axes[axis]
                    crd_slc = "{0}=0".format(reduce_axis)
                    default_keepdims = len(self.shape) == len(crds.shape)
                    iscc = self.iscentered('cell')

                    if npkwargs.get("keepdims", default_keepdims):
                        crds = self.crds.slice_keep(crd_slc, cc=iscc)
                    else:
                        crds = self.crds.slice_reduce(crd_slc, cc=iscc)
                    defer_wrapping = True

        # little hack for broadcasting vectors and scalars together
        crd_shape = crds.shape_nc if center == "node" else crds.shape_cc
        crd_shape = list(crd_shape)
        try:
            arr_shape = list(arr.shape)
        except AttributeError:
            arr_shape = [len(arr)] + list(arr[0].shape)

        if fldtype is None and arr_shape == crd_shape:
            fldtype = "scalar"
        if fldtype is None and (arr_shape[1:] == crd_shape or
                                arr_shape[:-1] == crd_shape):
            fldtype = "vector"

        if fldtype is None:
            fldtype = type(self)
        else:
            fldtype = field_type(fldtype)

        # Transform functions are intentionally omitted. The idea being that
        # the transform was already applied when creating arr
        # same idea with zyx_native... whatever created arr should have done
        # so using the natural xyz order since that's the shape of Field.data
        fld = fldtype(name, crds, arr, time=time, center=center,
                      meta=self.meta, deep_meta=context, parents=[self],
                      zyx_native=False, pretty_name=pretty_name)
        # if the operation reduced a vector to something with 1 component,
        # then turn the result into a ScalarField
        if fld.nr_comps == 1:
            fld = fld.component_fields()[0]
        return fld

    def wrap_field(self, data, name="NoName", fldtype=None, **kwargs):
        """Wrap an ndarray into a field in the local representation"""
        center = kwargs.pop('center', self.center)
        if fldtype is None:
            if len(data.shape) == len(self.shape):
                fldtype = self.fldtype
            else:
                fldtype = "scalar"
        return viscid.wrap_field(data, self.crds, name=name, fldtype=fldtype,
                                 center=center, **kwargs)

    def __array_wrap__(self, out_arr, context=None):  # pylint: disable=W0613
        return self.wrap(out_arr)

    def _ow(self, other):
        # hack because Fields don't broadcast correctly after a ufunc?
        try:
            return other.__array__()
        except AttributeError:
            return other

    def __add__(self, other):
        return self.wrap(self.data.__add__(self._ow(other)), other=other)
    def __sub__(self, other):
        return self.wrap(self.data.__sub__(self._ow(other)), other=other)
    def __mul__(self, other):
        return self.wrap(self.data.__mul__(self._ow(other)), other=other)
    def __div__(self, other):
        return self.wrap(self.data.__div__(self._ow(other)), other=other)
    def __truediv__(self, other):
        return self.wrap(self.data.__truediv__(self._ow(other)), other=other)
    def __floordiv__(self, other):
        return self.wrap(self.data.__floordiv__(self._ow(other)), other=other)
    def __mod__(self, other):
        return self.wrap(self.data.__mod__(self._ow(other)), other=other)
    def __divmod__(self, other):
        return self.wrap(self.data.__divmod__(self._ow(other)), other=other)
    def __pow__(self, other):
        return self.wrap(self.data.__pow__(self._ow(other)), other=other)
    def __lshift__(self, other):
        return self.wrap(self.data.__lshift__(self._ow(other)), other=other)
    def __rshift__(self, other):
        return self.wrap(self.data.__rshift__(self._ow(other)), other=other)
    def __and__(self, other):
        return self.wrap(self.data.__and__(self._ow(other)), other=other)
    def __xor__(self, other):
        return self.wrap(self.data.__xor__(self._ow(other)), other=other)
    def __or__(self, other):
        return self.wrap(self.data.__or__(self._ow(other)), other=other)

    def __radd__(self, other):
        return self.wrap(self.data.__radd__(self._ow(other)), other=other)
    def __rsub__(self, other):
        return self.wrap(self.data.__rsub__(self._ow(other)), other=other)
    def __rmul__(self, other):
        return self.wrap(self.data.__rmul__(self._ow(other)), other=other)
    def __rdiv__(self, other):
        return self.wrap(self.data.__rdiv__(self._ow(other)), other=other)
    def __rtruediv__(self, other):
        return self.wrap(self.data.__rtruediv__(self._ow(other)), other=other)
    def __rfloordiv__(self, other):
        return self.wrap(self.data.__rfloordiv__(self._ow(other)), other=other)
    def __rmod__(self, other):
        return self.wrap(self.data.__rmod__(self._ow(other)), other=other)
    def __rdivmod__(self, other):
        return self.wrap(self.data.__rdivmod__(self._ow(other)), other=other)
    def __rpow__(self, other):
        return self.wrap(self.data.__rpow__(self._ow(other)), other=other)

    # inplace operations are not implemented since the data
    # can up and disappear (due to clear_cache)... this could cause
    # confusion
    def __iadd__(self, other):
        return NotImplemented
    def __isub__(self, other):
        return NotImplemented
    def __imul__(self, other):
        return NotImplemented
    def __idiv__(self, other):
        return NotImplemented
    def __itruediv__(self, other):  # pylint: disable=R0201,W0613
        return NotImplemented
    def __ifloordiv__(self, other):  # pylint: disable=R0201,W0613
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

    def __lt__(self, other):
        return self.wrap(self.data.__lt__(self._ow(other)), other=other)
    def __le__(self, other):
        return self.wrap(self.data.__le__(self._ow(other)), other=other)
    def __eq__(self, other):
        return self.wrap(self.data.__eq__(self._ow(other)), other=other)
    def __ne__(self, other):
        return self.wrap(self.data.__ne__(self._ow(other)), other=other)
    def __gt__(self, other):
        # print("::__gt__::", self.data.shape, other.shape,
        #       self.data.__gt__(other).shape, "ndim", other.ndim)
        return self.wrap(self.data.__gt__(self._ow(other)), other=other)
    def __ge__(self, other):
        return self.wrap(self.data.__ge__(self._ow(other)), other=other)

    def any(self, **kwargs):
        return self.wrap(self.data.any(**kwargs), npkwargs=kwargs)
    def all(self, **kwargs):
        return self.wrap(self.data.all(**kwargs), npkwargs=kwargs)
    def argmax(self, axis=None, out=None, **kwargs):
        kwargs.update(axis=axis, out=out)
        return self.wrap(self.data.argmax(**kwargs), npkwargs=kwargs)
    def argmin(self, axis=None, out=None, **kwargs):
        kwargs.update(axis=axis, out=out)
        return self.wrap(self.data.argmin(**kwargs), npkwargs=kwargs)
    def argpartition(self, kth, axis=-1, **kwargs):
        kwargs.update(kth=kth, axis=axis)
        return self.data.argpartition(**kwargs)
    def argsort(self, axis=-1, kind='quicksort', order=None, **kwargs):
        kwargs.update(axis=axis, kind=kind, order=order)
        return self.wrap(self.data.argsort(**kwargs), npkwargs=kwargs)
    def clip(self, min=None, max=None, out=None, **kwargs):
        kwargs.update(min=min, max=max, out=out)
        return self.wrap(self.data.clip(**kwargs), npkwargs=kwargs)
    def conj(self, **kwargs):
        return self.wrap(self.data.conj(**kwargs), npkwargs=kwargs)
    def conjugate(self, **kwargs):
        return self.wrap(self.data.conjugate(**kwargs), npkwargs=kwargs)
    def cumprod(self, axis=None, dtype=None, order=None, **kwargs):
        kwargs.update(axis=axis, dtype=dtype, order=order)
        return self.wrap(self.data.cumprod(**kwargs), npkwargs=kwargs)
    def cumsum(self, axis=None, dtype=None, order=None, **kwargs):
        kwargs.update(axis=axis, dtype=dtype, order=order)
        return self.wrap(self.data.cumsum(**kwargs), npkwargs=kwargs)
    def max(self, **kwargs):
        return self.wrap(self.data.max(**kwargs), npkwargs=kwargs)
    def mean(self, **kwargs):
        return self.wrap(self.data.mean(**kwargs), npkwargs=kwargs)
    def min(self, **kwargs):
        return self.wrap(self.data.min(**kwargs), npkwargs=kwargs)
    def nonzero(self, **kwargs):
        return self.data.nonzero(**kwargs)
    def partition(self, kth, axis=-1, **kwargs):
        kwargs.update(kth=kth, axis=axis)
        return self.wrap(self.data.partition(**kwargs), npkwargs=kwargs)
    def prod(self, **kwargs):
        return self.wrap(self.data.prod(**kwargs), npkwargs=kwargs)
    def ptp(self, axis=None, out=None, **kwargs):
        kwargs.update(axis=axis, out=out)
        return self.wrap(self.data.ptp(**kwargs), npkwargs=kwargs)
    def round(self, decimals=0, out=None, **kwargs):
        kwargs.update(decimals=decimals, out=out)
        return self.wrap(self.data.round(**kwargs), npkwargs=kwargs)
    def std(self, **kwargs):
        return self.wrap(self.data.std(**kwargs), npkwargs=kwargs)
    def sum(self, **kwargs):
        return self.wrap(self.data.sum(**kwargs), npkwargs=kwargs)

    def trapz(self, axis=-1, fudge_factor=None):
        """Integrate field over a single axis

        Args:
            axis (str, int): axis name or index
            fudge_factor (callable): function that is called with
                func(data, crd_arr), where crd_arr is shaped. This is
                useful for including parts of the jacobian, like
                sin(theta) dtheta.

        Returns:
            Field or float
        """
        if axis in self.crds.axes:
            axis = self.crds.axes.index(axis)
        assert isinstance(axis, (int, np.integer))
        crd_arr = self.get_crd(axis, shaped=True)

        try:
            crd_arr = np.expand_dims(crd_arr, axis=self.nr_comp)
            if self.nr_comp > axis:
                fld_axis = axis
            else:
                fld_axis = axis + 1
        except TypeError:
            fld_axis = axis

        if fudge_factor is None:
            arr = self.data
        else:
            arr = self.data * fudge_factor(self.data, crd_arr)

        if self.nr_sdims > 1:
            slc = [slice(None) for _ in self.sshape]
            slc[axis] = 0
            ret = viscid.empty_like(self[tuple(slc)])
            ret.data = np.trapz(arr, crd_arr.reshape(-1), axis=fld_axis)
        else:
            ret = np.trapz(arr, crd_arr.reshape(-1), axis=fld_axis)
        return ret

    def cumtrapz(self, axis=-1, fudge_factor=None, initial=0):
        """Cumulatively integrate field over a single axis

        Args:
            axis (str, int): axis name or index
            fudge_factor (callable): function that is called with
                func(data, crd_arr), where crd_arr is shaped. This is
                useful for including parts of the jacobian, like
                sin(theta) dtheta.
            initial (float): Initial value

        Returns:
            Field
        """
        ret = None

        try:
            from scipy.integrate import cumtrapz as _sp_cumtrapz

            if axis in self.crds.axes:
                axis = self.crds.axes.index(axis)
            assert isinstance(axis, (int, np.integer))
            crd_arr = self.get_crd(axis, shaped=True)

            try:
                crd_arr = np.expand_dims(crd_arr, axis=self.nr_comp)
                if self.nr_comp > axis:
                    fld_axis = axis
                else:
                    fld_axis = axis + 1
            except TypeError:
                fld_axis = axis

            if fudge_factor is None:
                arr = self.data
            else:
                arr = self.data * fudge_factor(self.data, crd_arr)

            ret = viscid.empty_like(self)
            ret.data = _sp_cumtrapz(arr, crd_arr.reshape(-1), axis=fld_axis,
                                    initial=initial)
        except ImportError:
            viscid.logger.error("Scipy is required to perform cumtrapz")
            raise

        return ret

    @property
    def real(self):
        ret = self
        if self.dtype.kind == "c":
            ret = self.wrap(self.data.real)
        return ret

    @property
    def imag(self):
        return self.wrap(self.data.imag)

    def angle(self, deg=False):
        # hack: np.angle casts to ndarray... that's annoying
        return self.wrap(np.angle(self.data, deg=deg))

    def transpose(self, *axes):
        """ same behavior as numpy transpose, alse accessable
        using np.transpose(fld) """
        if len(axes) == 1 and axes[0]:
            axes = axes[0]
        if axes == (None, ) or len(axes) == 0:
            axes = list(range(self.nr_dims - 1, -1, -1))
        if len(axes) != self.nr_dims:
            raise ValueError("transpose can not change number of axes")
        clist = self._src_crds.get_clist()
        caxes = list(axes)
        if self.nr_comps:
            caxes.remove(self.nr_comp)
            caxes = [i - 1 if i > self.nr_comp else i for i in caxes]
        new_clist = [clist[i] for i in caxes]
        cunits = self._src_crds.get_units((c[0] for c in new_clist), allow_invalid=1)
        t_crds = coordinate.wrap_crds(self._src_crds.crdtype, new_clist,
                                      units=cunits, **self._src_crds.meta)
        t_data = self.data.transpose(axes)

        context = dict(crds=t_crds)
        # i think the layout should be taken care of automagically
        return self.wrap(t_data, context=context)

    def transpose_crds(self, *axes):
        if axes == (None, ) or len(axes) == 0:
            axes = list(range(self.nr_sdims - 1, -1, -1))
        ax_inds = [self.crds.ind(ax) for ax in axes]
        if self.nr_comps:
            ax_inds = [i + 1 if i >= self.nr_comp else i for i in ax_inds]
            ax_inds.insert(self.nr_comp, self.nr_comp)
        return self.transpose(*ax_inds)

    @property
    def T(self):
        return self.transpose()

    @property
    def TC(self):
        return self.transpose_crds()

    spatial_transpose = transpose_crds
    ST = TC

    def swapaxes(self, a, b):
        axes = list(range(self.nr_dims))
        axes[a], axes[b] = b, a
        return self.transpose(*axes)

    def swap_crd_axes(self, a, b):
        a, b = [self.crds.ind(s) for s in (a, b)]
        axes = list(range(self.nr_sdims))
        axes[a], axes[b] = b, a
        return self.transpose_crds(*axes)

    def astype(self, dtype):
        ret = self
        if np.dtype(dtype) != self.dtype:
            ret = self.wrap(self.data.astype(dtype))
        return ret

    def as_layout(self, to_layout, force_c_contiguous=True):
        raise NotImplementedError("abstract method")

    def as_interlaced(self, force_c_contiguous=True):
        return self.as_layout(LAYOUT_INTERLACED,
                              force_c_contiguous=force_c_contiguous)

    def as_flat(self, force_c_contiguous=True):
        return self.as_layout(LAYOUT_FLAT,
                              force_c_contiguous=force_c_contiguous)

    @property
    def __crd_system__(self):
        if self.find_info("crd_system", None):
            crd_system = self.find_info("crd_system")
        else:
            crd_system = NotImplemented
        return crd_system


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
        cunits = self._src_crds.get_units((c[0] for c in downclist), allow_invalid=1)
        downcrds = coordinate.wrap_crds("nonuniform_cartesian", downclist,
                                        units=cunits, **self._src_crds.meta)
        return self.wrap(downdat, {"crds": downcrds})

    def as_layout(self, to_layout, force_c_contiguous=True):
        if force_c_contiguous:
            return self.as_c_contiguous()
        else:
            return self


class VectorField(Field):
    _TYPE = "vector"

    def __init__(self, *args, **kwargs):
        """
        Keyword Arguments:
            comp_names (sequence): sequence of component names

        See Also:
            :py:class:`viscid.Field`
        """
        if 'comp_names' in kwargs:
            comp_names = kwargs.pop('comp_names')
        else:
            comp_names = _DEFAULT_COMPONENT_NAMES

        super(VectorField, self).__init__(*args, **kwargs)

        self.comp_names = comp_names

    def component_views(self):
        """ return numpy views to components individually, memory layout
        of the original field is maintained """
        flds = self.component_fields()
        return [f.data for f in flds]

    def component_fields(self):
        if (self._cache is None and
            self.layout == LAYOUT_FLAT and
            isinstance(self._src_data, (list, tuple)) and
            all([isinstance(f, Field) for f in self._src_data])):
            # if all elements are fields
            return self._src_data

        lst = [None] * self.nr_comps
        for i in range(self.nr_comps):
            slc = [slice(None)] * (len(self.shape))
            slc[self.nr_comp] = self.comp_names[i]
            lst[i] = self[tuple(slc)]
        return lst

    def as_layout(self, to_layout, force_c_contiguous=True):
        """Get an interlaced version of this field

        Note:
            This will trigger a data load if the data is not already
            in memory

        Args:
            to_layout: either LAYOUT_INTERLACED or LAYOUT_FLAT
            force_c_contiguous: if data is not c contiguous, then wrap
                it in another np.array() call.

        Returns:
            self, or Field if
        """
        was_loaded = self.is_loaded

        ret = None
        if self.layout == to_layout:
            if force_c_contiguous:
                if not self.data.flags['C_CONTIGUOUS']:
                    # print("calling np.ascontiguousarray")
                    ret = self.wrap(np.ascontiguousarray(self.data))
                else:
                    # print("returning self")
                    ret = self
        else:
            ctx = dict(force_layout=to_layout)
            # the data load is going to wrap the array, i think it's
            # redundant to put an "ascontiguousarray" here
            ret = self.wrap(self.data, ctx)

        if not was_loaded and ret is not self:
            self.clear_cache()
        return ret

    def _ow(self, other):
        # - hack because Fields don't broadcast correctly after a ufunc?
        # - Vector Hack if other is a scalar field to promote it to a vector
        #   field so numpy can broadcast interlaced vector flds w/ scalar flds
        try:
            if isinstance(other, ScalarField):
                if self.layout == 'interlaced':
                    return other.__array__()[..., np.newaxis]
                elif self.layout == 'flat':
                    return other.__array__()[np.newaxis, ...]
            return other.__array__()
        except AttributeError:
            return other


class MatrixField(Field):
    _TYPE = "matrix"


class TensorField(Field):
    _TYPE = "tensor"


##
## EOF
##
