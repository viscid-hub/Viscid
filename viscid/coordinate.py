# pylint: disable=too-many-lines
""" Container for grid coordinates

Coordinates primarily go into Field objects. The order of coords in
the clist should mirror the data layout, as in if data[ix, iy, iz]
(C-order, ix is fastest varying index) then list should go x, y, z...
this is the default order

types:
     "Structured":
         "nonuniform_cartesian"
         "Cylindrical"
         "Spherical"
     "Unstructured":
         -> Not Implemented <-

FIXME: uniform coordinates are generally unsupported, but they're just
       a special case of nonuniform coordinates, so the functionality
       is still there... it's just unnatural to use full crd arrays
       with uniform coordinates
"""

from __future__ import print_function, division
# from timeit import default_timer as time
import itertools
import sys

import numpy as np

import viscid
from viscid.compat import string_types, izip
from viscid.vutil import subclass_spider
from viscid import sliceutil


__all__ = ['NonuniformFullArrayError', 'arrays2crds', 'wrap_crds', 'extend_arr']


class NonuniformFullArrayError(ValueError):
    '''trying to make uniform crds from nonuniform full arrays'''
    pass


def arrays2crds(crd_arrs, crd_type='AUTO', crd_names="xyzuvw", **kwargs):
    """make either uniform or nonuniform coordnates given full arrays

    Args:
        crd_arrs (array-like): for n-dimensional crds, supply a list
            of n ndarrays
        crd_type (str): passed to wrap_crds, should uniquely specify a
            type of coordinate object
        crd_names (iterable): names of coordinates in the same order
            as crd_arrs. Should always be in xyz order.
    """
    clist = []
    uniform_clist = []
    is_uniform = True
    try:
        _ = len(crd_arrs[0])
    except TypeError:
        crd_arrs = [crd_arrs]

    for crd_name, arr in zip(crd_names, crd_arrs):
        arr = viscid.asarray_datetime64(arr, conservative=True)

        clist.append((crd_name, arr))
        try:
            rtol = 4 * np.finfo(arr.dtype).eps
        except ValueError:
            rtol = 0

        if len(arr) > 1:
            diff = arr[1:] - arr[:-1]
        else:
            diff = [1, 1]

        if viscid.is_timedelta_like(diff, conservative=True):
            diff = diff / np.ones((1,), dtype=diff.dtype)

        if np.allclose(diff[0], diff, rtol=rtol):
            uniform_clist.append((crd_name, [arr[0], arr[-1], len(arr)]))
        else:
            is_uniform = False

    # uniform crds don't play nice with datetime axes
    if any(viscid.is_time_like(arr, conservative=True) for arr in crd_arrs):
        is_uniform = False

    if not crd_type:
        crd_type = 'AUTO'

    if crd_type.startswith('AUTO'):
        prefix = 'uniform' if is_uniform else 'nonuniform'
        crd_type = crd_type.replace('AUTO', prefix)
    cl = uniform_clist if crd_type.startswith('uniform') else clist
    crds = wrap_crds(crd_type, cl, **kwargs)

    return crds

def lookup_crds_type(type_name):
    for cls in subclass_spider(Coordinates):
        if cls.istype(type_name):
            return cls
    return None

def wrap_crds(crdtype, clist, **kwargs):
    """  """
    # print(len(clist), clist[0][0], len(clist[0][1]), crytype)
    try:
        try:
            crdtype = lookup_crds_type(crdtype)
        except AttributeError:
            pass
        return crdtype(clist, **kwargs)
    except TypeError:
        raise NotImplementedError("can not decipher crds: {0}".format(crdtype))

def extend_arr(x, n=1, cell_fraction=1.0, full_arr=True, default_width=1e-5,
               width_arr=None):
    """TODO: docstring"""
    x = viscid.asarray_datetime64(x, conservative=True)

    try:
        n_low, n_high = n
    except TypeError:
        n_low, n_high = n, n

    if width_arr is None:
        width_arr = x

    if full_arr:
        if len(x) > 1:
            dxl = cell_fraction * (width_arr[1] - width_arr[0])
            dxh = cell_fraction * (width_arr[-1] - width_arr[-2])
        else:
            dxl = cell_fraction * default_width
            dxh = cell_fraction * default_width

        arr_low = [x[0] - (i + 1) * dxl for i in range(n_low)][::-1]
        arr_high = [x[-1] + (i + 1) * dxh for i in range(n_high)]
        ret = np.concatenate([arr_low, x, arr_high])
    else:
        if cell_fraction != 1.0:
            raise ValueError("Linspace arrays require extension with "
                             "cell_fraction == 1.0")
        xl, xh, nx = x
        nx = int(nx)
        if nx > 1:
            dx = (xh - xl) / (nx - 1)
        else:
            dx = default_width
        xl -= (n_low * cell_fraction) * dx
        xh += (n_high * cell_fraction) * dx
        nx += n_low + n_high
        ret = np.array([xl, xh, nx], dtype=x.dtype)
    return ret


class Coordinates(object):
    """Base class for all coordinate types

    Note:
        Subclasses should only call super(...).__init__ after setting
        up the axes
    """
    _TYPE = "none"

    _axes = ["x", "y", "z"]

    _Pcrds = None
    meta = None
    _units_validated = False
    _units = None

    def __init__(self, units=None, **kwargs):
        self.units = units
        self.meta = kwargs

    @property
    def axes(self):
        return self._axes

    @property
    def units(self):
        if not self._units_validated:
            units = self._units
            if not isinstance(units, (list, tuple)):
                units = [units] * self.nr_dims
            units += [None] * (len(units) - self.nr_dims)
            units = ["" if u is None else u.strip() for u in units]
            self._units = units
            self._units_validated = True
        return self._units

    @units.setter
    def units(self, val):
        self._units_validated = False
        self._units = val

    @property
    def crdtype(self):
        return self._TYPE

    @classmethod
    def istype(cls, type_str):  # noqa
        return cls._TYPE == type_str.lower()

    def is_spherical(self):
        return "spherical" in self._TYPE

    def is_uniform(self):
        return self._TYPE.startswith('uniform')

    def ind(self, axis):
        if isinstance(axis, int):
            if axis < len(self.axes):
                return axis
            else:
                raise ValueError()
        else:
            return self.axes.index(axis)

    def _axisvalid(self, ax):
        try:
            self.ind(ax)
            return True
        except ValueError:
            return False

    def get_units(self, axes, allow_invalid=False):
        ret = []
        for ax in axes:
            if self._axisvalid(ax):
                ret.append(self.units[self.ind(ax)])
            elif allow_invalid:
                ret.append('')
            else:
                raise ValueError("{0} not in axes ({1})".format(ax, self._axes))
        return ret

    def get_unit(self, axis):
        return self.get_units([axis])[0]

    def axis_name(self, axis):
        if isinstance(axis, string_types):
            if axis in self._crds:
                return axis
            else:
                raise KeyError(axis)
        else:
            return self.axes[axis]

    def as_local_coordinates(self):
        return self

    def as_uv_coordinates(self):
        # trying to get self._axes could raise an AttributeError for
        # subclasses != StructredCrds
        if len(self._axes) == 2:
            return self
        else:
            raise NotImplementedError()


class StructuredCrds(Coordinates):
    _TYPE = "structured"
    _CENTER = {"none": "", "node": "nc", "cell": "cc",
               "face": "fc", "edge": "ec"}
    SUFFIXES = list(_CENTER.values())

    _axes = ["x", "y", "z"]

    _src_crds_nc = None
    _src_crds_cc = None

    _dtype = None

    _reflect_axes = None
    has_cc = None

    def __init__(self, init_clist, has_cc=True, reflect_axes=None,
                 dtype=None, full_arrays=True, quiet_init=False, **kwargs):
        """ if caled with an init_clist, then the coordinate names
        are taken from this list """
        self.has_cc = has_cc

        self.reflect_axes = reflect_axes

        self.dtype = dtype

        if init_clist is not None:
            self._axes = [d[0].lower() for d in init_clist]
        self.clear_crds()
        if init_clist is not None:
            self._set_crds(init_clist)
        super(StructuredCrds, self).__init__(**kwargs)

    @property
    def xl_nc(self):
        return self.get_xl()
    @property
    def xh_nc(self):
        return self.get_xh()
    @property
    def L_nc(self):
        return self.get_L()
    @property
    def min_dx_nc(self):
        return self.get_min_dx()

    @property
    def xl_cc(self):
        return self.get_xl(center='cell')
    @property
    def xh_cc(self):
        return self.get_xh(center='cell')
    @property
    def L_cc(self):
        return self.get_L(center='cell')
    @property
    def min_dx_cc(self):
        return self.get_min_dx(center='cell')

    @property
    def nr_dims(self):
        """ number of spatial dimensions """
        return len(self._axes)

    @property
    def dtype(self):
        if self._dtype:
            return self._dtype
        else:
            if self._Pcrds is not None:
                return self[self.axes[0]].dtype
            elif self._src_crds_nc:
                return None

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype

    @property
    def shape(self):
        return self.shape_nc

    @property
    def shape_nc(self):
        return np.asarray([len(self[ax]) for ax in self.axes])

    @property
    def shape_cc(self):
        return np.asarray([len(self[ax + "cc"]) for ax in self.axes])

    @property
    def size(self):
        return self.size_nc

    @property
    def size_nc(self):
        return np.product(self.shape_nc)

    @property
    def size_cc(self):
        return np.product(self.shape_cc)

    @property
    def reflect_axes(self):
        return self._reflect_axes

    @reflect_axes.setter
    def reflect_axes(self, val):
        if not val:
            val = []
        self._reflect_axes = val

    @property
    def _crds(self):
        "!!! BIG NOTE: THIS PROPERTY IS STILL PRIVATE !!!"
        if self._Pcrds is None:
            self._fill_crds_dict()
        return self._Pcrds

    def clear_crds(self):
        self._src_crds_nc = {}
        self._src_crds_cc = {}
        self.clear_cache()
        # for d in self.axes:
        #     for sfx in self.SUFFIXES:
        #         self._Pcrds[d + sfx] = None
        #         self._Pcrds[d.upper() + sfx] = None

    def clear_cache(self):
        self._Pcrds = None

    def _set_crds(self, clist):
        """ called with a list of lists:
        (('x', ndarray), ('y',ndarray), ('z',ndarray))
        Note: input crds are assumed to be node centered
        """
        for ci in clist:
            axis = ci[0]
            dat_nc = ci[1]

            if axis not in self._axes:
                raise KeyError()

            self._src_crds_nc[axis.lower()] = ci[1]
            if len(ci) > 2:
                self._src_crds_cc[axis.lower()] = ci[2]

    def apply_reflections(self):
        """
        Returns:
            Coordinates with reflections applied
        """
        if len(self.reflect_axes) > 0:
            return type(self)(self.get_clist(), has_cc=self.has_cc,
                              dtype=self.dtype)
        else:
            return self

    def _reflect_axis_arr(self, arr):  # pylint: disable=no-self-use
        return np.array(-1.0 * arr[::-1])

    def reflect_fld_arr(self, arr, cc=False, nr_comp=None, nr_comps=None):
        fudge = 0 if nr_comp is None else 1
        assert len(arr.shape) == len(self._axes) + fudge

        rev = [True if ax in self.reflect_axes else False for ax in self._axes]
        if cc:
            shape = self.shape_cc
        else:
            shape = self.shape_nc

        if nr_comp is not None:
            shape = list(shape)
            shape.insert(nr_comp, nr_comps)
            rev.insert(nr_comp, False)
        first, second = sliceutil.make_fwd_slice(shape, [], rev)
        return arr[tuple(first)][tuple(second)]

    def reflect_slices(self, slices, cc=False, cull_second=True):
        """This is tricky, only use if you know what's going on"""
        # assert len(slices) == len(self._axes)

        rev = [True if ax in self.reflect_axes else False for ax in self._axes]
        if cc:
            shape = self.shape_cc
        else:
            shape = self.shape_nc
        first, second = sliceutil.make_fwd_slice(shape, slices, rev,
                                                 cull_second=cull_second)
        return first, second

    def _fill_crds_dict(self):
        """ do the math to calc node, cell, face, edge crds from
        the src crds """
        self._Pcrds = {}

        # get node centered crds from the src crds
        sfx = self._CENTER["node"]
        for axis, arr in self._src_crds_nc.items():
            # make axis into string representation
            # axis = self.axis_name(axis)
            ind = self.ind(axis)
            arr = np.array(arr, dtype=arr.dtype.name)

            if axis in self.reflect_axes:
                arr = self._reflect_axis_arr(arr)

            # ====================================================================
            # if self.transform_funcs is not None:
            #     if axis in self.transform_funcs:
            #         arr = self.transform_funcs[axis](self, arr,
            #                                          **self.transform_kwargs)
            #     elif ind in self.transform_funcs:
            #         arr = self.transform_funcs[ind](self, arr,
            #                                         **self.transform_kwargs)
            # ====================================================================

            flatarr, openarr = self._ogrid_single(ind, arr)
            self._Pcrds[axis.lower()] = flatarr
            self._Pcrds[axis.upper()] = openarr
            # now with suffix
            self._Pcrds[axis.lower() + sfx] = flatarr
            self._Pcrds[axis.upper() + sfx] = openarr

        # recalculate all cell centers, and refresh face / edges
        sfx = self._CENTER["cell"]
        for i, a in enumerate(self.axes):
            # a = self.axis_name(a)  # validate input
            if a in self._src_crds_cc:
                ccarr = self._src_crds_cc[a]
            else:
                # doing the cc math this way also works for datetime objects
                if len(self._Pcrds[a]) == 1:
                    ccarr = self._Pcrds[a]
                else:
                    ccarr = self._Pcrds[a][:-1] + 0.5 * (self._Pcrds[a][1:] -
                                                         self._Pcrds[a][:-1])
            flatarr, openarr = self._ogrid_single(a, ccarr)
            self._Pcrds[a + sfx] = flatarr
            self._Pcrds[a.upper() + sfx] = openarr

        # ok, so this is a little recursive, but it's ok since we set
        # _Pcrds above, note however that now we only have nc and cc
        # crds in _Pcrds
        crds_nc = self.get_crds_nc()
        crds_nc_shaped = self.get_crds_nc(shaped=True)
        crds_cc = self.get_crds_cc()
        crds_cc_shaped = self.get_crds_cc(shaped=True)

        # store references to face and edge centers while we're here
        sfx = self._CENTER["face"]
        for i, a in enumerate(self.axes):
            self._Pcrds[a + sfx] = [None] * len(self.axes)
            self._Pcrds[a.upper() + sfx] = [None] * len(self.axes)
            for j, d in enumerate(self.axes):  # pylint: disable=W0612
                if i == j:
                    self._Pcrds[a + sfx][j] = crds_nc[i][:-1]
                    self._Pcrds[a.upper() + sfx][j] = self._sm1(crds_nc_shaped[i])
                else:
                    self._Pcrds[a + sfx][j] = crds_cc[i]
                    self._Pcrds[a.upper() + sfx][j] = crds_cc_shaped[i]

        # same as face, but swap nc with cc
        sfx = self._CENTER["edge"]
        for i, a in enumerate(self.axes):
            self._Pcrds[a + sfx] = [None] * 3
            self._Pcrds[a.upper() + sfx] = [None] * 3
            for j in range(3):
                if i != j:
                    self._Pcrds[a + sfx][j] = crds_nc[i][:-1]
                    self._Pcrds[a.upper() + sfx][j] = self._sm1(crds_nc_shaped[i])
                else:
                    self._Pcrds[a + sfx][j] = crds_cc[i]
                    self._Pcrds[a.upper() + sfx][j] = crds_cc_shaped[i]

    @staticmethod
    def _sm1(a):
        n = a.shape
        slices = [slice(None) if ni <= 1 else slice(None, -1) for ni in n]
        return a[tuple(slices)]

    def _ogrid_single(self, axis, arr):
        """ returns (flat array, open array) """
        i = self.ind(axis)
        shape = [1] * self.nr_dims
        shape[i] = -1

        # print(arr.shape)

        if len(arr.shape) == 1:
            return arr, np.reshape(arr, shape)
        elif len(arr.shape) == self.nr_dims:
            return np.ravel(arr), arr
        else:
            raise ValueError()

    def get_slice_extent(self, selection, allow_read=False):
        """find value extent of a slice"""
        # IMPORTANT: This does not touch the actual crd arrays to maintain
        #            lazyness
        # _, selections, _ = self._parse_slice(selections)
        # extent = sliceutil.selections2values(None, selections, self.nr_dims)

        sel_list = sliceutil.raw_sel2sel_list(selection)
        full_sel_info = sliceutil.fill_nd_sel_list(sel_list, self.axes)
        full_sel_list, full_ax_names, full_newdim_flags = full_sel_info
        std_sel_list = sliceutil.standardize_sel_list(full_sel_list)

        # # FIXME: std_sel_list to extent
        # extent = selection2values(None, std_sel_list)

        extent = sliceutil.sel_list2values(None, std_sel_list)

        if np.any(np.isnan(extent)):
            if allow_read:
                raise NotImplementedError("Fallback to crd read to find slice "
                                          "extent is not yet implemented.")
            raise RuntimeError("Can't infer extent for selection [{0}] "
                               "without reading crds".format(selection))

        # enforce that the values are increasing
        for i, ext in enumerate(extent):
            if ext[1] < ext[0]:
                extent[i] = extent[i][::-1]

        extent = np.asarray(extent)

        return extent.T

    def nc2cc(self, default_width=1e-5):
        """Extend coordinates half a grid cell in all directions

        Used for turning node centered fields to cell centered without
        changing the data, just the coordinates.

        Returns:
            New coordinates instance with same type as self
        """
        axes = self.axes
        crds_nc = self.get_crds_nc()
        crds_cc = self.get_crds_cc()
        for i, x_cc in enumerate(crds_cc):
            crds_nc[i] = extend_arr(x_cc, default_width=default_width,
                                    width_arr=crds_nc[i])
        new_clist = [(ax, nc) for ax, nc in zip(axes, crds_nc)]
        return type(self)(new_clist)

    def _make_slice(self, selection, cc=False):
        """Turns a selection into finalized slices and coordinates

        selection is passed through
        :py:func:`viscid.sliceutil.raw_sel2sel_list` and
        :py:func:`viscid.sliceutil.standardize_sel_list`

        Parameters:
            selection (str, list): selection
            cc (bool): cell centered slice

        Returns:
            tuple (slices, slcrds, reduced)

            * **slices**: list of things that can go straight into an
                ndarray's __getitem__; one for each axis in self plus
                any new axes
            * **slcrds**: a clist for what the coords will be after the
                slice
            * **reduced**: a list of (axis, location) pairs of which
                axes are sliced out

        Note: cc is necessary for finding the closest plane, otherwise it
            might be off by half a grid cell
        """
        sel_list = sliceutil.raw_sel2sel_list(selection)
        full_sel_info = sliceutil.fill_nd_sel_list(sel_list, self.axes)
        full_sel_list, full_ax_names, full_newdim_flags = full_sel_info
        std_sel_list = sliceutil.standardize_sel_list(full_sel_list)

        crd_arrs_nc = []
        crd_arrs_cc = []
        for i, ax, sel in izip(itertools.count(), full_ax_names, std_sel_list):
            if ax in self.axes:
                assert not full_newdim_flags[i]
                crd_arrs_nc.append(self.get_nc(ax))
                crd_arrs_cc.append(self.get_cc(ax))
            else:
                assert sel == np.newaxis and full_newdim_flags[i]
                if cc:
                    crd_arrs_nc.append(np.array([-1e-1, 1e-1], dtype=self.dtype))
                    crd_arrs_cc.append(np.array([0.0], dtype=self.dtype))
                else:
                    crd_arrs_nc.append(np.array([0.0], dtype=self.dtype))
                    crd_arrs_cc.append(np.array([0.0], dtype=self.dtype))

        crd_arrs = crd_arrs_cc if cc else crd_arrs_nc
        idx_sel_list = list(sliceutil.std_sel_list2index(std_sel_list, crd_arrs))

        # Figure out what the sel_list is doing. If the slice reduces
        # out a dimension, put it in reduced. Also apply the slices to
        # the coordinate arrays so they can be used in to create new
        # fields later on.
        sliced_clist = []
        reduced = []

        for i, slc in enumerate(idx_sel_list):
            axis = full_ax_names[i]

            if isinstance(slc, slice):
                if not all(s is None or hasattr(s, "__index__")
                           for s in (slc.start, slc.stop, slc.step)):
                    raise TypeError("bad sss:", slc)

                sliced_clist_item = None

                # if using step on a cell centered field, we need to figure
                # stuff out, since the data && crd slices can't be the same,
                # and it's not as simple as just adding 1 b/c of the stride
                if cc and slc.step not in [None, 1, -1]:
                    crd_cc = crd_arrs_cc[i][slc]
                    crd_nc = 0.5 * (crd_cc[1:] + crd_cc[:-1])

                    allslc = slice(slc.start, slc.stop,
                                   slc.step // np.abs(slc.step))
                    allnc = crd_arrs_nc[i][allslc]
                    crd_nc = np.concatenate([[allnc[0]], crd_nc, [allnc[-1]]])

                    sliced_clist_item = [axis, crd_nc, crd_cc]

                elif cc and slc.stop is not None:
                    if slc.stop >= 0:
                        if slc.step is None or slc.step > 0:
                            newstop = slc.stop + 1
                        else:
                            newstop = slc.stop - 1
                    else:  # slc.stop < 0
                        # this will slice crds nc, so if we're going backward,
                        # the extra crd will be included
                        # print("am i used?", slc.step)
                        newstop = slc.stop
                    slc = slice(slc.start, newstop, slc.step)

                if sliced_clist_item is None:
                    cval = crd_arrs_nc[i][slc]
                    sliced_clist_item = [axis, cval]

                sliced_clist.append(sliced_clist_item)

            elif isinstance(slc, np.ndarray):
                if cc:
                    cval_cc = crd_arrs_cc[i][slc]
                    cval_nc = (cval_cc[1:] + cval_cc[:-1]) / 2
                    cval_nc = extend_arr(cval_nc)
                    sliced_clist.append([axis, cval_nc, cval_cc])
                else:
                    sliced_clist.append([axis, crd_arrs_nc[i][slc]])

            elif hasattr(slc, "__index__"):
                cval = crd_arrs_nc[i][slc]
                reduced.append([axis, cval])

            elif slc == np.newaxis:
                cval = crd_arrs_nc[i]
                sliced_clist.append([axis, cval])

            else:
                raise TypeError("{0};; {1}".format(type(slc), slc))

        # determine uniform / nonuniform-ness of the resulting crds
        if '_' in self._TYPE:
            uniform_type, more_type = self._TYPE.split('_', 1)
        else:
            uniform_type, more_type = self._TYPE, None

        any_non_uniform = False
        for i, clist_i in enumerate(sliced_clist):
            # int crds (including times/dates) are never store at uniform crds
            if isinstance(clist_i[1][0], (int, np.integer, np.datetime64)):
                any_non_uniform = True
                break

            diff_nc = np.diff(clist_i[1])
            if diff_nc.shape[0] > 0:
                # integers / dates / times should have been caught above
                rtol = 4 * np.finfo(diff_nc.dtype).eps
                if not np.allclose(diff_nc[0], diff_nc, rtol=rtol):
                    any_non_uniform = True
                    break

        if uniform_type == 'uniform' and any_non_uniform:
            uniform_type = 'nonuniform'

        if more_type is None:
            crds_type_name = uniform_type
        else:
            crds_type_name = '_'.join([uniform_type, more_type])
        crds_type = lookup_crds_type(crds_type_name)

        if uniform_type == 'uniform':
            for i, slcl in enumerate(sliced_clist):
                sliced_clist[i] = slcl[:2]

        # print("< slice made", slices)
        return idx_sel_list, sliced_clist, reduced, crds_type, full_ax_names

    def make_slice(self, selection, cc=False):
        slices, sliced_clist, reduced, crds_type, _ = \
            self._make_slice(selection, cc=cc)
        return slices, sliced_clist, reduced, crds_type

    def make_slice_reduce(self, selection, cc=False):
        """make slice, and reduce dims that were not explicitly sliced"""
        slices, crdlst, reduced, crds_type = self.make_slice(selection, cc=cc)
        # augment slices / reduced

        reduced_axes = [t[0] for t in reduced]

        for i, axis in enumerate(self.axes):
            reduce_axis = False
            if axis not in reduced_axes:
                if cc and self.shape_cc[i] == 1:
                    slices[i] = 0
                    reduced.insert(i, [axis, self.get_cc(axis)[0]])
                    reduce_axis = True
                elif not cc and self.shape_nc[i] == 1:
                    slices[i] = 0
                    reduced.insert(i, [axis, self.get_nc(axis)[0]])
                    reduce_axis = True
                # go find which element of crdlst to take out since there
                # may already be elements missing
                if reduce_axis:
                    for j, crd in enumerate(crdlst):
                        if crd is not None and crd[0] == axis:
                            crdlst[j] = None
                            break
        # remove the newly reduced crds
        crdlst = [crd for crd in crdlst if crd is not None]

        return slices, crdlst, reduced, crds_type

    def make_slice_keep(self, selection, cc=False):
        """make slice, but put back dims that were explicitly reduced"""
        slices, crdlst, reduced, crds_type, full_ax_names = \
            self._make_slice(selection, cc=cc)

        # put reduced dims back, reduced will be in the same order as self.axes
        # since make_slice loops over self.axes to do the slices; this enables
        # us to call insert in the loop

        new_ax_names = [c[0] for c in crdlst]

        for axis, loc in reduced:  # pylint: disable=W0612
            new_axis_ind = full_ax_names.index(axis)

            # slices[new_axis_ind] will be an int not a slice since it was
            # reduced
            loc_ind = slices[new_axis_ind]
            crd_nc = self.get_nc(axis)

            if cc:
                if loc_ind == -1:
                    crd = crd_nc[-2:]
                    slc = slice(-1, None)
                elif loc_ind == -2:
                    crd = crd_nc[-2:]
                    slc = slice(-2, -1)
                else:
                    crd = crd_nc[loc_ind:loc_ind + 2]
                    slc = slice(loc_ind, loc_ind + 1)
            else:
                if loc_ind == -1:
                    crd = crd_nc[-1:]
                    slc = slice(-1, None)
                else:
                    crd = crd_nc[loc_ind:loc_ind + 1]
                    slc = slice(loc_ind, loc_ind + 1)

            slices[new_axis_ind] = slc
            crdlst.insert(new_axis_ind, [axis, crd])

        # should be no more reduced crds
        reduced = []
        # print("MAKE SLICE KEEP : slices", slices, "crdlst", crdlst,
        #       "reduced", reduced)
        return slices, crdlst, reduced, crds_type

    def slice(self, selection, cc=False):
        """Get crds that describe a slice (subset) of this grid.
        Reduces dims the same way numpy / fields do. Chances are
        you want either slice_reduce or slice_keep
        """
        slices, crdlst, reduced, crds_type = self.make_slice(selection, cc=cc)
        # pass through if nothing happened
        if sliceutil.all_slices_none(slices):
            return self
        cunits = self.get_units((c[0] for c in crdlst), allow_invalid=True)
        return wrap_crds(crds_type, crdlst, dtype=self.dtype, units=cunits,
                         full_arrays=True, quiet_init=True, **self.meta)

    def slice_and_reduce(self, selection, cc=False):
        """Get crds that describe a slice (subset) of this grid. Go
        through, and if the slice didn't touch a dim with only one crd,
        reduce it
        """
        slices, crdlst, reduced, crds_type = self.make_slice_reduce(selection,
                                                                    cc=cc)
        # pass through if nothing happened
        if sliceutil.all_slices_none(slices):
            return self
        cunits = self.get_units((c[0] for c in crdlst), allow_invalid=True)
        return wrap_crds(crds_type, crdlst, dtype=self.dtype, units=cunits,
                         full_arrays=True, quiet_init=True, **self.meta)

    def slice_and_keep(self, selection, cc=False):
        slices, crdlst, reduced, crds_type = self.make_slice_keep(selection,
                                                                  cc=cc)
        # pass through if nothing happened
        if sliceutil.all_slices_none(slices):
            return self
        cunits = self.get_units((c[0] for c in crdlst), allow_invalid=True)
        return wrap_crds(crds_type, crdlst, dtype=self.dtype, units=cunits,
                         full_arrays=True, quiet_init=True, **self.meta)

    slice_reduce = slice_and_reduce
    slice_keep = slice_and_keep

    def slice_interp(self, selection, cc=False):
        _, crdlst, _, crds_type = self.make_slice_keep(selection, cc=cc)

        # axes, selection, _ = self._parse_slice(selection)
        sel_list = sliceutil.raw_sel2sel_list(selection)
        full_sel_info = sliceutil.fill_nd_sel_list(sel_list, self.axes)
        full_sel_list, full_ax_names, full_newdim_flags = full_sel_info
        std_sel_list = sliceutil.standardize_sel_list(full_sel_list)

        limits = sliceutil.sel_list2values(None, full_sel_list, len(crdlst))

        # for slices that were specified using a float, set that crd
        # to the desired float instead of the nearest crd as does slice_keep
        for i, slc in enumerate(std_sel_list):
            # slc = sliceutil.convert_deprecated_floats(slc, "slc")
            _slc_lims = limits[i]
            if _slc_lims[0] == _slc_lims[1]:
                val = _slc_lims[0]
                crdlst[i][1][0] = val
                if len(crdlst[i][1]) > 1:
                    # i'm not sure what this is for...
                    crdlst[i][1][1] = val
                if len(crdlst[i]) > 2:
                    crdlst[i][2][0] = val
        cunits = self.get_units((c[0] for c in crdlst), allow_invalid=True)
        return wrap_crds(crds_type, crdlst, dtype=self.dtype, units=cunits,
                         full_arrays=True, quiet_init=True, **self.meta)

    def get_crd(self, axis, shaped=False, center="none"):
        """if axis is not specified, return all coords,
        shaped makes axis capitalized and returns ogrid like crds
        shaped is only used if axis == None
        sfx can be none, node, cell, face, edge
        raises KeyError if axis not found
        """
        return self.get_crds([axis], shaped=shaped, center=center)[0]

    def get_crds(self, axes=None, shaped=False, center="none"):
        """Get coordinate arrays

        Parameters:
            axes: should be a list of names or integer indices, or None
                to return all axes
            shaped: boolean if you want a shaped or flat ndarray
                (True is the same as capitalizing axis)
            center: 'node' 'cell' 'edge' 'face', same as adding a suffix
                to axes
        Returns:
            list of coords as ndarrays
        """
        if axes is None:
            axes = list(self.axes)

        axes = [self.axes[a] if isinstance(a, (int, np.int)) else a for a in axes]
        axes = [a.upper() if shaped else a for a in axes]

        sfx = self._CENTER[center.lower()]
        return [self._crds[self.axis_name(a) + sfx] for a in axes]

    # def get_dcrd(self, axis, shaped=False, center="none"):
    #     """ if axis is not specified, return all coords,
    #     shaped makes axis capitalized and returns ogrid like crds
    #     shaped is only used if axis == None
    #     sfx can be none, node, cell, face, edge
    #     raises KeyError if axis not found """
    #     return self.get_dcrds([axis], shaped, center)[0]

    # def get_dcrds(self, axes=None, shaped=False, center="none"):
    #     """ axes: should be a list of names or integer indices, or None
    #               to return all axes
    #     shaped: boolean if you want a shaped or flat ndarray
    #             (True is the same as capitalizing axis)
    #     center: 'node' 'cell' 'edge' 'face', same as adding a suffix to axes
    #     returns list of coords as ndarrays """
    #     if axes == None:
    #         axes = [a.upper() if shaped else a for a in self.axes]
    #     if not isinstance(axes, (list, tuple)):
    #         try:
    #             axes = [a.upper() if shaped else a for a in axes]
    #         except TypeError:
    #             axes = [axes]
    #     sfx = self._CENTER[center.lower()]
    #     return [(self._crds[self.axis_name(a) + sfx][1:] -
    #              self._crds[self.axis_name(a) + sfx][:-1]) for a in axes]

    def get_culled_axes(self, ignore=2):
        """Get only good axes

        Discard axes whose coords have length <= ignore... useful for
        2d fields that only have 1 cell

        Returns
            list of axes names
        """
        return [name for name in self.axes if len(self[name]) > ignore]

    def get_clist(self, axes=None, slc=Ellipsis, full_arrays=True, center='node'):
        """??

        Returns:
            a clist of the coordinates sliced if you wish

        Note:
            I recommend using ``numpy.s_`` for making the slice
        """
        if not full_arrays:
            raise NotImplementedError("you need uniform crds for this")
        if axes is None:
            axes = self.axes

        ret = [[axis, self.get_crd(axis, center=center)[slc].copy()] for axis in axes]

        # # slc was never None, what was this for?
        # if slc is None and full_arrays and center == 'node' and self._src_crds_cc:
        #     for i, axis in enumerate(axes):
        #         if axis in self._src_crds_cc:
        #             ret[i].append(self._src_crds_cc[axis].copy())
        return ret

    ## These methods just return one crd axis
    def get_nc(self, axis, shaped=False):
        """returns a flat ndarray of coordinates along a given axis
        axis can be crd name as string, or index, as in x==2, y==1, z==2
        """
        return self.get_crd(axis, shaped=shaped, center="node")

    def get_cc(self, axis, shaped=False):
        """returns a flat ndarray of coordinates along a given axis
        axis can be crd name as string, or index, as in x==2, y==1, z==2
        """
        return self.get_crd(axis, shaped=shaped, center="cell")

    def get_fc(self, axis, shaped=False):
        """returns a flat ndarray of coordinates along a given axis
        axis can be crd name as string, or index, as in x==2, y==1, z==2
        """
        return self.get_crd(axis, shaped=shaped, center="face")

    def get_ec(self, axis, shaped=False):
        """returns a flat ndarray of coordinates along a given axis
        axis can be crd name as string, or index, as in x==2, y==1, z==2
        """
        return self.get_crd(axis, shaped=shaped, center="edge")

    ## These methods return all crd axes
    def get_crds_nc(self, axes=None, shaped=False):
        """returns all node centered coords as a list of ndarrays, flat if
        shaped==False, or shaped if shaped==True
        """
        return self.get_crds(axes=axes, center="node", shaped=shaped)

    def get_crds_cc(self, axes=None, shaped=False):
        """returns all cell centered coords as a list of ndarrays, flat if
        shaped==False, or shaped if shaped==True
        """
        return self.get_crds(axes=axes, center="cell", shaped=shaped)

    def get_crds_ec(self, axes=None, shaped=False):
        """return all edge centered coords as a list of ndarrays, flat if
        shaped==False, or shaped if shaped==True
        """
        return self.get_crds(axes=axes, center="edge", shaped=shaped)

    def get_crds_fc(self, axes=None, shaped=False):
        """returns all face centered coords as a list of ndarrays, flat if
        shaped==False, or shaped if shaped==True
        """
        return self.get_crds(axes=axes, center="face", shaped=shaped)

    def __array__(self, *args, **kwargs):
        return self.get_points()

    def points(self, center=None, **kwargs):
        return self.get_points(center=center, **kwargs)

    def get_points(self, center="none", **kwargs):
        """returns all points in a grid defined by crds as a
        nr_dims x nr_points ndarray
        """
        crds = self.get_crds(shaped=False, center=center)
        shape = [len(c) for c in crds]
        arr = np.empty([len(shape)] + [np.prod(shape)])
        for i, c in enumerate(crds):
            arr[i, :] = np.repeat(np.tile(c, int(np.prod(shape[:i]))),
                                  int(np.prod(shape[i + 1:])))
        return arr

    def as_mesh(self, center="none"):
        crds = self.get_crds(shaped=False, center=center)
        shape = [len(c) for c in crds]
        pts = self.get_points(center=center)
        while len(shape) > 2:
            try:
                shape.remove(1)
            except ValueError:
                raise ValueError("Only crds with 2 meaningful dimensions can "
                                 "create a surface mesh")
        return pts.reshape([3] + shape)

    def get_dx(self, axes=None, center='node'):
        """Get cell widths if center == 'node', or distances between cell
        centers if center == 'cell' """
        return [x[1:] - x[:-1] if len(x) > 1 else 1.0
                for x in self.get_crds(axes, center=center)]

    def get_min_dx(self, axes=None, center='node'):
        """Get a minimum cell width for each axis"""
        return np.array([np.min(dx) for dx in self.get_dx(axes, center=center)])

    def get_xl(self, axes=None, center='node'):
        _lst = [x[0] for x in self.get_crds(axes, center=center)]
        try:
            return np.array(_lst)
        except ValueError:
            return np.array(_lst, dtype='object')

    def get_xh(self, axes=None, center='node'):
        _lst = [x[-1] for x in self.get_crds(axes, center=center)]
        try:
            return np.array(_lst)
        except ValueError:
            return np.array(_lst, dtype='object')

    def get_L(self, axes=None, center='node'):
        """Get lengths"""
        return (self.get_xh(axes, center=center) -
                self.get_xl(axes, center=center))

    @property
    def nr_points(self):
        return self.get_nr_points()

    def get_nr_points(self, center="none", **kwargs):
        """returns the number of points in a grid defined by these crds"""
        return np.prod([len(crd) for crd in self.get_crds(center=center)])

    def iter_points(self, center="none", **kwargs):  # pylint: disable=W0613
        """returns an iterator over all the points in a grid, each
        nextitem will be a list of nr_dims numbers
        """
        return itertools.product(*self.get_crds(shaped=False, center=center))

    def _newax_cval(self, xl, xh, cc):
        raise NotImplementedError()

    def atleast_3d(self, xl=-1, xh=1, cc=False):
        if self.nr_dims >= 3:
            return self
        else:
            current_axes = self._axes
            if 'x' in current_axes or 'y' in current_axes or 'z' in current_axes:
                axes3 = ['x', 'y', 'z']
                target_idx = [0, 1, 2]
                for i, ax in reversed(list(enumerate(axes3))):
                    if ax in current_axes:
                        axes3.pop(i)
                        target_idx.pop(i)
            else:
                axes3 = ['fill_a', 'fill_b', 'fill_c']
                target_idx = [3, 3, 3]

            new_clist = self.get_clist()

            axes3 = axes3[:3 - len(current_axes)]
            for idx, ax in zip(target_idx, axes3):
                new_clist.insert(idx, (ax, self._newax_cval(xl, xh, cc)))
            return type(self)(new_clist)

    def print_tree(self):
        c = self.get_clist()
        for l in c:
            print(l[0])

    def __len__(self):
        return self.nr_dims

    def __getitem__(self, axis):
        """ returns coord identified by axis (shaped / centering encoded
        through capitalization of crd, and suffix), ex: 'Xcc' is shaped cell
        centered, 'znc' is flat node centered, 'x' is flat node centered,
        2 is self._axes[2]
        """
        return self.get_crd(axis)

    def __setitem__(self, axis, arr):
        """I recommend against doing this since there may be unintended
        side effects
        """
        raise RuntimeError("setting crds is deprecated - the constructor "
                           "does far too much transforming of the input "
                           "to assume that arr will be in the right form")
        # return self._set_crds((axis, arr))

    def __delitem__(self, item):
        raise ValueError("can not delete crd this way")

    def __contains__(self, item):
        if item[-2:] in list(self._CENTER.values()):
            item = item[:-2].lower()
        return item in self._crds


class UniformCrds(StructuredCrds):
    _TYPE = "uniform"

    _nc_linspace_args = None
    _cc_linspace_args = None

    xl_nc = None
    xh_nc = None
    L_nc = None
    shape_nc = None
    min_dx_nc = None

    xl_cc = None
    xh_cc = None
    shape_cc = None
    L_cc = None
    min_dx_cc = None

    def __init__(self, init_clist, full_arrays=False, dtype='f8',
                 quiet_init=False, **kwargs):
        """
        Args:
            init_clist: this should look something like
                [('y', [yl, yh, ny]), ('x', [xl, xh, nx])] unless
                full_arrays is True.
            full_arrays (bool): indicates that clist is given as
                full coordinate arrays, like for non-uniform crds

        Raises:
            NonuniformFullArrayError if full_arrays and crds are not uniform
        """
        self.dtype = dtype

        if full_arrays:
            if not quiet_init:
                s = ("DEPRECATION...\n"
                     "Full arrays for uniform crds shouldn't be used due to \n"
                     "finite precision errors")
                viscid.logger.warning(s)

            _nc_linspace_args = []  # pylint: disable=unreachable

            for _, arr in init_clist:
                if viscid.is_time_like(arr, conservative=True):
                    raise NotImplementedError("Datetime arrays can't be in "
                                              "uniform crds yet")
                arr = np.asarray(arr)
                if len(arr) > 1:
                    diff = arr[1:] - arr[:-1]
                else:
                    diff = np.array([1, 1])
                # This allclose is the problem... when slicing, it doesn't
                # always pass
                rtol = 4 * np.finfo(arr.dtype).eps
                if not np.allclose(diff[0], diff, rtol=rtol):
                    raise NonuniformFullArrayError("Arrays are not uniform, {0}"
                                                   "".format(arr))
                if len(arr) > 0:
                    _nc_linspace_args.append([arr[0], arr[-1], len(arr)])
                else:
                    _nc_linspace_args.append([np.nan, np.nan, 0])
        else:
            for _, arr in init_clist:
                if len(arr) != 3:
                    raise ValueError("is this a full_array?")
                if viscid.is_time_like([arr[0], arr[1]], conservative=True):
                    raise NotImplementedError("Datetime arrays can't be in "
                                              "uniform crds yet")
            _nc_linspace_args = [arr for _, arr in init_clist]

        self._nc_linspace_args = _nc_linspace_args
        self._cc_linspace_args = []
        for args in _nc_linspace_args:
            half_dx = 0.5 * (args[1] - args[0]) / max(args[2], 1)
            cc_args = [args[0] + half_dx, args[1] - half_dx,
                       max(args[2] - 1, 0)]
            self._cc_linspace_args.append(cc_args)

        # node centered things
        self.xl_nc = np.array([args[0] for args in self._nc_linspace_args],
                              dtype=self.dtype)
        self.xh_nc = np.array([args[1] for args in self._nc_linspace_args],
                              dtype=self.dtype)
        self.shape_nc = np.array([args[2] for args in self._nc_linspace_args],
                                 dtype='int')
        self.L_nc = self.xh_nc - self.xl_nc
        self.min_dx_nc = self.L_nc / self.shape_nc
        self.min_dx_nc = np.ma.masked_values(self.min_dx_nc, 0.0)

        # cell centered things
        self.xl_cc = np.array([args[0] for args in self._cc_linspace_args],
                              dtype=self.dtype)
        self.xh_cc = np.array([args[1] for args in self._cc_linspace_args],
                              dtype=self.dtype)
        self.shape_cc = np.array([args[2] for args in self._cc_linspace_args],
                                 dtype='int')
        self.L_cc = self.xh_cc - self.xl_cc
        self.min_dx_cc = self.L_cc / np.clip(self.shape_cc, 1, None)

        init_clist = [(axis, None) for axis, _ in init_clist]
        super(UniformCrds, self).__init__(init_clist, dtype=self.dtype,
                                          **kwargs)

    def _pull_out_axes(self, arrs, axes, center='none'):
        center = center.lower()
        if axes is None:
            axind_list = list(range(len(self._axes)))
        else:
            axind_list = []  # self._axes.index(ax.lower()) for ax in axes]
            for ax in axes:
                try:
                    axind_list.append(self._axes.index(ax.lower()))
                except AttributeError:
                    axind_list.append(ax)
                # except ValueError:
                #     raise

        if center == 'none' or center == 'node':
            return arrs[0][axind_list]
        elif center == 'cell':
            return arrs[1][axind_list]
        else:
            raise ValueError()

    def get_min_dx(self, axes=None, center='node'):
        """Get a minimum cell width for each axis"""
        ret = self._pull_out_axes([self.min_dx_nc, self.min_dx_cc], axes,
                                  center=center)
        return np.ma.masked_values(ret, 0.0)

    def get_xl(self, axes=None, center='node'):
        return self._pull_out_axes([self.xl_nc, self.xl_cc], axes,
                                   center=center)

    def get_xh(self, axes=None, center='node'):
        return self._pull_out_axes([self.xh_nc, self.xh_cc], axes,
                                   center=center)

    def get_L(self, axes=None, center='node'):
        """Get lengths"""
        return self._pull_out_axes([self.L_nc, self.L_cc], axes,
                                   center=center)

    @property
    def nr_points(self):
        return self.get_nr_points()

    def get_nr_points(self, center="none", **kwargs):
        """returns the number of points in a grid defined by these crds"""
        center = center.lower()
        if center == 'none' or center == 'node':
            return self.size_nc
        else:
            return self.size_cc

    def nc2cc(self, default_width=1e-5):
        """Extend coordinates half a grid cell in all directions

        Used for turning node centered fields to cell centered without
        changing the data, just the coordinates.

        Returns:
            New coordinates instance with same type as self
        """
        axes = self.axes
        xl, xh = self.get_xl(), self.get_xh()
        dx = (xh - xl) / self.shape_nc
        # FIXME: I don't think this will extend datetime64/timedelta64 axes
        dx = np.choose(np.abs(dx) > 0, [default_width, dx])
        xl -= 0.5 * dx
        xh += 0.5 * dx
        nx = self.shape_nc + 1
        new_clist = [(ax, [xl[i], xh[i], nx[i]]) for i, ax in enumerate(axes)]
        return type(self)(new_clist)

    def _newax_cval(self, xl, xh, cc):
        if cc:
            return [xl, xh, 2]
        else:
            x0 = 0.5 * (xl + xh)
            return [x0, x0, 1]

    def get_clist(self, axes=None, slc=Ellipsis, full_arrays=False, center="node"):
        if full_arrays:
            return super(UniformCrds, self).get_clist(axes=axes, slc=slc,
                                                      center=center)
        if slc != Ellipsis:
            raise NotImplementedError("use full_arrays=True with slice != Ellipsis"
                                      "for now")
        if axes is None:
            axes = self.axes
        lst = []
        for ax in axes:
            ax = ax.lower()
            if center == "node":
                ls_args = self._nc_linspace_args[self.axes.index(ax)]
            elif center == "cell":
                ls_args = self._cc_linspace_args[self.axes.index(ax)]
            else:
                raise ValueError("node/cell centering only in here")

            if ax in self.reflect_axes:
                lst.append([ax, [-ls_args[1], -ls_args[0], ls_args[2]]])
            else:
                lst.append([ax, list(ls_args)])
        return lst

    def _fill_crds_dict(self):
        assert len(self._nc_linspace_args) == len(self._axes)
        self._src_crds_nc = {}
        for ax, p in zip(self._axes, self._nc_linspace_args):
            self._src_crds_nc[ax] = np.linspace(p[0], p[1], p[2]).astype(self.dtype)
        return super(UniformCrds, self)._fill_crds_dict()


class NonuniformCrds(StructuredCrds):
    _TYPE = "nonuniform"

    def __init__(self, init_clist, full_arrays=True, quiet_init=False, **kwargs):
        if not full_arrays:
            raise ValueError("did you want Uniform crds?")
        super(NonuniformCrds, self).__init__(init_clist, **kwargs)

    def _newax_cval(self, xl, xh, cc):
        if cc:
            return np.array([xl, xh], dtype=self.dtype)
        else:
            return np.array([0.5 * (xl + xh)], dtype=self.dtype)

    def get_clist(self, axes=None, slc=Ellipsis, full_arrays=True, center="node"):
        """??

        Returns:
            a clist of the coordinates sliced if you wish

        Note:
            I recommend using ``numpy.s_`` for making the slice
        """
        if not full_arrays:
            raise ValueError("Unstructured crds only have full arrays")
        return super(NonuniformCrds, self).get_clist(axes=axes, slc=slc,
                                                     center=center)


class UniformCartesianCrds(UniformCrds):
    _TYPE = "uniform_cartesian"
    _axes = ["x", "y", "z"]


class NonuniformCartesianCrds(NonuniformCrds):
    _TYPE = "nonuniform_cartesian"
    _axes = ["x", "y", "z"]


class UniformSphericalCrds(UniformCrds):
    _TYPE = "uniform_spherical"
    _axes = ["phi", "theta", "r"]


class NonuniformSphericalCrds(NonuniformCrds):
    _TYPE = "nonuniform_spherical"
    _axes = ["phi", "theta", "r"]


class UnstructuredCrds(Coordinates):
    _TYPE = "unstructured"

    def __init__(self, **kwargs):
        super(UnstructuredCrds, self).__init__(**kwargs)
        raise NotImplementedError()

def _main():
    print("full array")
    x = np.arange(4)
    print(viscid.extend_arr(x))
    print(viscid.extend_arr(x, n=2))
    print(viscid.extend_arr(x, n=1, cell_fraction=0.5))
    print(viscid.extend_arr(x, n=2, cell_fraction=0.5))
    print(viscid.extend_arr(x, n=1, cell_fraction=0.25))
    print(viscid.extend_arr(x, n=2, cell_fraction=0.25))
    print(viscid.extend_arr(x, n=(2, 3), cell_fraction=0.25))

    print('full array single value')
    x = np.array([0.0])
    print(viscid.extend_arr(x))
    print(viscid.extend_arr(x, n=2))
    print(viscid.extend_arr(x, n=1, cell_fraction=0.5))
    print(viscid.extend_arr(x, n=(2, 3), cell_fraction=0.5))

    print("linspace array")
    x = [0.0, 3.0, 4]
    print(np.linspace(*viscid.extend_arr(x, full_arr=False)))
    print(np.linspace(*viscid.extend_arr(x, full_arr=False, n=2)))
    print(np.linspace(*viscid.extend_arr(x, full_arr=False, n=(2, 3))))

    print("linspace array, single value")
    x = np.array([0.0, 0.0, 1])
    print(np.linspace(*viscid.extend_arr(x, full_arr=False)))
    print(np.linspace(*viscid.extend_arr(x, full_arr=False, n=2)))
    print(np.linspace(*viscid.extend_arr(x, full_arr=False, n=(2, 3))))

if __name__ == "__main__":
    _main()

##
## EOF
##
