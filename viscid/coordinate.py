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
from viscid import vutil


def arrays2crds(crd_arrs, crd_names="xyzuvw"):
    """make either uniform or nonuniform coordnates given full arrays

    Args:
        crd_arrs (array-like): for n-dimensional crds, supply a list
            of n ndarrays
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
        arr = vutil.asarray_dt(arr)

        clist.append((crd_name, arr))
        try:
            atol = 100 * np.finfo(arr.dtype).eps
        except ValueError:
            atol = 0

        if len(arr) > 1:
            diff = arr[1:] - arr[:-1]
        else:
            diff = [1, 1]

        if vutil.isdatetime(diff):
            diff = diff / np.ones((1,), dtype=diff.dtype)

        if np.allclose(diff[0], diff[1:], atol=atol):
            uniform_clist.append((crd_name, [arr[0], arr[-1], len(arr)]))
        else:
            is_uniform = False

    # uniform crds don't play nice with datetime axes
    if any(vutil.isdatetime(arr) for arr in crd_arrs):
        is_uniform = False

    if is_uniform:
        crds = wrap_crds("uniform", uniform_clist)
    else:
        crds = wrap_crds("nonuniform", clist)
    return crds

def lookup_crds_type(type_name):
    for cls in vutil.subclass_spider(Coordinates):
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

def extend_arr_by_half(x, full_arr=True):
    x = vutil.asarray_dt(x)

    if full_arr:
        dxl = x[1] - x[0]
        dxh = x[-1] - x[-2]
        ret = np.concatenate([[x[0] - dxl], x, [x[-1] + dxh]])
    else:
        xl, xh, nx = x
        nx = int(nx)
        dx = (xh - xl) / nx
        xl -= 0.5 * dx
        xh += 0.5 * dx
        nx += 1
        ret = np.array([xl, xh, nx], dtype=x.dtype)
    return ret


class Coordinates(object):
    _TYPE = "none"

    __crds = None

    @property
    def crdtype(self):
        return self._TYPE

    @classmethod
    def istype(cls, type_str):  # noqa
        return cls._TYPE == type_str.lower()

    def is_spherical(self):
        return "spherical" in self._TYPE

    def __init__(self, **kwargs):
        pass

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
    _INIT_FULL_ARRAYS = True
    _CENTER = {"none": "", "node": "nc", "cell": "cc",
               "face": "fc", "edge": "ec"}
    SUFFIXES = list(_CENTER.values())

    _axes = ["x", "y", "z"]

    _src_crds_nc = None
    _src_crds_cc = None

    _reflect_axes = None
    has_cc = None

    def __init__(self, init_clist, has_cc=True, reflect_axes=None,
                 **kwargs):
        """ if caled with an init_clist, then the coordinate names
        are taken from this list """
        super(StructuredCrds, self).__init__(**kwargs)
        self.has_cc = has_cc

        self.reflect_axes = reflect_axes

        if init_clist is not None:
            self._axes = [d[0].lower() for d in init_clist]
        self.clear_crds()
        if init_clist is not None:
            self._set_crds(init_clist)

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
    def axes(self):
        return self._axes

    @property
    def dtype(self):
        return self[self.axes[0]].dtype

    @property
    def shape(self):
        return self.shape_nc

    @property
    def shape_nc(self):
        return [len(self[ax]) for ax in self.axes]

    @property
    def shape_cc(self):
        return [len(self[ax + "cc"]) for ax in self.axes]

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
        if self.__crds is None:
            self._fill_crds_dict()
        return self.__crds

    def clear_crds(self):
        self._src_crds_nc = {}
        self._src_crds_cc = {}
        self.clear_cache()
        # for d in self.axes:
        #     for sfx in self.SUFFIXES:
        #         self.__crds[d + sfx] = None
        #         self.__crds[d.upper() + sfx] = None

    def clear_cache(self):
        self.__crds = None

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
        first, second = vutil.make_fwd_slice(shape, [], rev)
        return arr[tuple(first)][tuple(second)]

    def reflect_slices(self, slices, cc=False, cull_second=True):
        """This is tricky, only use if you know what's going on"""
        # assert len(slices) == len(self._axes)

        rev = [True if ax in self.reflect_axes else False for ax in self._axes]
        if cc:
            shape = self.shape_cc
        else:
            shape = self.shape_nc
        first, second = vutil.make_fwd_slice(shape, slices, rev,
                                             cull_second=cull_second)
        return first, second

    def _fill_crds_dict(self):
        """ do the math to calc node, cell, face, edge crds from
        the src crds """
        self.__crds = {}

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
            self.__crds[axis.lower()] = flatarr
            self.__crds[axis.upper()] = openarr
            # now with suffix
            self.__crds[axis.lower() + sfx] = flatarr
            self.__crds[axis.upper() + sfx] = openarr

        # recalculate all cell centers, and refresh face / edges
        sfx = self._CENTER["cell"]
        for i, a in enumerate(self.axes):
            # a = self.axis_name(a)  # validate input
            if a in self._src_crds_cc:
                ccarr = self._src_crds_cc[a]
            elif self.shape[i] == 1:
                ccarr = self.__crds[a]
            else:
                # doing the cc math this way also works for datetime objects
                ccarr = self.__crds[a][:-1] + 0.5 * (self.__crds[a][1:] -
                                                     self.__crds[a][:-1])
            flatarr, openarr = self._ogrid_single(a, ccarr)
            self.__crds[a + sfx] = flatarr
            self.__crds[a.upper() + sfx] = openarr

        # ok, so this is a little recursive, but it's ok since we set
        # __crds above, note however that now we only have nc and cc
        # crds in __crds
        crds_nc = self.get_crds_nc()
        crds_nc_shaped = self.get_crds_nc(shaped=True)
        crds_cc = self.get_crds_cc()
        crds_cc_shaped = self.get_crds_cc(shaped=True)

        # store references to face and edge centers while we're here
        sfx = self._CENTER["face"]
        for i, a in enumerate(self.axes):
            self.__crds[a + sfx] = [None] * len(self.axes)
            self.__crds[a.upper() + sfx] = [None] * len(self.axes)
            for j, d in enumerate(self.axes):  # pylint: disable=W0612
                if i == j:
                    self.__crds[a + sfx][j] = crds_nc[i]
                    self.__crds[a.upper() + sfx][j] = crds_nc_shaped[i]
                else:
                    self.__crds[a + sfx][j] = crds_cc[i]
                    self.__crds[a.upper() + sfx][j] = crds_cc_shaped[i]

        # same as face, but swap nc with cc
        sfx = self._CENTER["edge"]
        for i, a in enumerate(self.axes):
            self.__crds[a + sfx] = [None] * len(self.axes)
            self.__crds[a.upper() + sfx] = [None] * len(self.axes)
            for j, d in enumerate(self.axes):
                if i == j:
                    self.__crds[a + sfx][j] = crds_cc[i]
                    self.__crds[a.upper() + sfx][j] = crds_cc_shaped[i]
                else:
                    self.__crds[a + sfx][j] = crds_nc[i]
                    self.__crds[a.upper() + sfx][j] = crds_nc_shaped[i]

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

    def ind(self, axis):
        if isinstance(axis, int):
            if axis < len(self.axes):
                return axis
            else:
                raise ValueError()
        else:
            return self.axes.index(axis)

    def axis_name(self, axis):
        if isinstance(axis, string_types):
            if axis in self._crds:
                return axis
            else:
                raise KeyError(axis)
        else:
            return self.axes[axis]

    def get_slice_extent(self, selection):
        """work in progress"""
        # print("get slice extent::", selection)
        float_err_msg = ("Can only get the extent of slices by location, aka "
                         "'x = 0f' or field['0f']")
        axes, selection = self._parse_slice(selection)
        # FIXME: this is a rediculous use of NaN
        extent = np.nan * np.empty((2, self.nr_dims), dtype='f')

        for i, slc in enumerate(selection):
            if slc != slice(None):
                slc = vutil.convert_deprecated_floats(slc, "slc")

                if not isinstance(slc, string_types):
                    raise TypeError(float_err_msg + ":: {0}".format(slc))

                split_slc = slc.split(":")
                for j, s in enumerate(split_slc):
                    if not s:
                        # FIXME: this is a rediculous use of NaN
                        split_slc[j] = np.nan
                    elif s[-1] == 'f':
                        split_slc[j] = float(s[:-1])
                    else:
                        raise ValueError(float_err_msg + ":: {0}".format(s))

                if len(split_slc) > 2:
                    raise NotImplementedError("get_slice_extent doesn't work "
                                              "with stride yet")
                extent[:, i] = split_slc

        # enforce that the values are increasing
        for d in range(extent.shape[1]):
            if extent[1, d] < extent[0, d]:
                extent[:, d] = extent[::-1, d]

        return extent

    def _parse_slice(self, selection):
        """Resolve axes names, Ellipsis, and np.newaxis from selection

        Args:
            selection (str, list): Something to select a subset of a
                field

        Returns:
            tuple (axes, selection)

            * *axes*: list of strings, one for each axis name
            * *selection*: something almost usable by
                :py:func:`numpy.ndarray.__getitem__`. The catch is the
                selection could contain etries that look like "0" or
                "5.0f". These strings are resolved by
                :py:func:`viscid.vutil.to_slices`.
        """
        # SIDE-EFFECT: selection won't have any whitespace
        # parse selection if it's a string into a list of selections
        # that are in the same order as the dimensions of self

        new_axes_names = []
        new_axes_inds = []
        ellipsis_ind = -1

        if selection is np.newaxis:
            pass
        elif not selection:
            selection = slice(None)
        elif isinstance(selection, string_types):
            # giving the whole slice as a string usually means the
            # user is specifying the axis of each slice
            slices = [slice(None)] * self.nr_dims
            other_axes = list(range(len(slices)))
            # kill all whitespace
            selection = "".join(selection.split())
            split_sel = selection.split(',')

            for i, single_sel in enumerate(split_sel):
                if "=" in single_sel:
                    ax, slc = single_sel.split("=")
                    # deal with nexaxis
                    if slc.lower() in ["none", "newaxis"]:
                        assert not (ax in self.axes or ax in new_axes_names)
                        new_axes_names.append(ax)
                        new_axes_inds.append(i)
                    else:
                        slices[self.axes.index(ax)] = slc
                        other_axes.remove(self.axes.index(ax))
                else:
                    if single_sel.lower() in ["none", "newaxis"]:
                        new_axes_names.append(None)
                        new_axes_inds.append(i)
                    elif single_sel.lower() in ["...", "ellipsis"]:
                        # in practice, it's super confusing to properly expand
                        # an ellipsis here given all the other things going on
                        # ie. slices by explicit axis / newaxes / etc.
                        # raise NotImplementedError("Ellipses are not allowed "
                        #                           "in a slice by string")
                        ellipsis_ind = other_axes[0]
                    else:
                        slices[other_axes[0]] = single_sel
                        other_axes.pop(0)
                        # slices[i - nr_skips] = single_sel
            # at this point, selection has length of exactly self.nr_dims
            if ellipsis_ind >= 0:
                for k in range(len(slices) - 1, ellipsis_ind - 1, -1):
                    if slices[k] == slice(None):
                        slices.pop(k)
                    else:
                        break
            selection = slices

        try:
            selection = list(selection)
        except TypeError:
            selection = [selection]

        # make stripped_selection which is selection with all newaxis and
        # Ellspsis elements removed such that
        # len(stripped_selection) == self.nr_dims.
        #
        # Note that new_axes_{names,inds} should be
        # empty b/c if selection was a string that contained newaxes
        # then they were dealt with above and there would be no more
        # left in selection.
        stripped_selection = []
        axes = list(self.axes)
        for i, sel in enumerate(selection):
            try:
                sel = sel.lower()
            except AttributeError:
                pass
            if sel in [np.newaxis, "newaxis", "none"]:
                new_axes_names.append(None)
                new_axes_inds.append(i)
            elif sel in [Ellipsis, '...', "ellipsis"]:
                assert ellipsis_ind == -1
                ellipsis_ind = i - len(new_axes_inds)
            else:
                stripped_selection.append(sel)

        # make names for unnamed new axes
        new_crd_cnt = 0
        for i, name in enumerate(new_axes_names):
            if name is None:
                new_name = "new-x{0}".format(new_crd_cnt)
                while new_name in axes or new_name in new_axes_names:
                    new_crd_cnt += 1
                    new_name = "new-x{0}".format(new_crd_cnt)
                new_axes_names[i] = new_name
                new_crd_cnt += 1

        # now put ellipsis && newaxes back into stripped_selection
        if ellipsis_ind >= 0:
            n_added = self.nr_dims - len(stripped_selection)

            for _ in range(n_added):
                stripped_selection.insert(ellipsis_ind, slice(None))

            for i in range(len(new_axes_inds)):
                if new_axes_inds[i] > ellipsis_ind:
                    new_axes_inds[i] += n_added

        for ind, name in izip(new_axes_inds, new_axes_names):
            stripped_selection.insert(ind, np.newaxis)
            axes.insert(ind, name)
        selection = stripped_selection

        # make sure there's a slice for all dimensions
        nr_new_dims = self.nr_dims + len(new_axes_names) - len(selection)
        selection += [slice(None)] * nr_new_dims

        return axes, selection

    @staticmethod
    def native_slice(crd_arr, slc, center='node', axis=None):
        if axis:
            crd_arr = crd_arr.get_crd(axis, center=center)

        crd_arr = vutil.asarray_dt(crd_arr)

        if slc is not None:
            crd_arr = crd_arr[slc]
        return crd_arr.reshape(-1)

    # def _native_slice(self, axis, slc, center='node'):
    #     """Get a sliced crd array to construct crds of same type

    #     Args:
    #         axis: something __getitem__ will understand to pull
    #             out a single coordinate axis
    #         slc (slice): Should be a valid slice of ints

    #     Returns:
    #         A sliced single coordinate array that __init__ understands
    #         by default
    #     """
    #     if slc is None:
    #         return self.get_crd(axis, center=center).reshape(-1)
    #     else:
    #         return self.get_crd(axis, center=center)[slc].reshape(-1)

    def extend_by_half(self):
        """Extend coordinates half a grid cell in all directions

        Used for turning node centered fields to cell centered without
        changing the data, just the coordinates.

        Returns:
            New coordinates instance with same type as self
        """
        axes = self.axes
        crds_cc = self.get_crds_cc()
        for i, x in enumerate(crds_cc):
            crds_cc[i] = extend_arr_by_half(x, full_arr=True)
        new_clist = [(ax, nc) for ax, nc in zip(axes, crds_cc)]
        return type(self)(new_clist)

    def make_slice(self, selection, cc=False):
        """Turns a slice string into a slice (should be private?)

        Slices should be made using the normal ``field[selection]``
        syntax. This function is more for internal use; however it
        does document the slice string syntax.

        In practice, selection can be a string like
        "y = 3:6:2, z = 0.0f" where integers indicate an index as
        opposed to floats followed by an 'f' which slice by crd value.
        The example slice would be the 3rd and 5th crds in y, and the
        z = 0.0 plane. Selection can also be the usual tuple of slice
        objects / integers like one would give to numpy.

        "y = 3:6:2, z = 0.0" will give (if z[32] is closest to z=0.0)::

            ([slice(None), slice(3, 6, 2), 32],
             [['x', ndarray(all nc x crds)], ['y', array(y[3], y[5])]],
             [['z', 0.0]],
            )

        Parameters:
            selection (int, str, slice): slice string
            cc (bool): cell centered slice

        Returns:
            tuple (slices, slcrds, reduced)

            * **slices**: list of slice objects, one for each axis in
              self
            * **slcrds**: a clist for what the coords will be after the
              slice
            * **reduced**: a list of (axis, location) pairs of which
              axes are sliced out

        Note: cc is necessary for finding the closest plane, otherwise it
            might be off by half a grid cell
        """
        # print("> make slice", selection)
        # parse selection if it's a string into a list of selections where
        # the axes are in the same order as self.axes
        axes, selection = self._parse_slice(selection)
        crd_arrs_nc = []
        crd_arrs_cc = []
        for i, ax, sel in izip(itertools.count(), axes, selection):
            if ax in self.axes:
                crd_arrs_nc.append(self.get_nc(ax))
                crd_arrs_cc.append(self.get_cc(ax))
            else:
                assert sel == np.newaxis
                if cc:
                    crd_arrs_nc.append(np.array([-1e-1, 1e-1], dtype=self.dtype))
                    crd_arrs_cc.append(np.array([0.0], dtype=self.dtype))
                else:
                    crd_arrs_nc.append(np.array([0.0], dtype=self.dtype))
                    crd_arrs_cc.append(np.array([0.0], dtype=self.dtype))
                # selection[i] = slice(None)

        crd_arrs = crd_arrs_cc if cc else crd_arrs_nc
        slices = list(vutil.to_slices(crd_arrs, selection))

        # Figure out what the selection is doing. If the slice reduces
        # out a dimension, put it in reduced. Also apply the slices to
        # the coordinate arrays so they can be used in to create new
        # fields later on.
        sliced_clist = []
        reduced = []

        # some types of slices require the result to be nonuniform crds,
        # so resolve that here
        crds_type_name = self._TYPE
        # for slices with a step, always make the crds nonuniform,
        # it's just easier that way
        for i, slc in enumerate(slices):
            if isinstance(slc, slice):
                if cc and slc.step not in [None, 1, -1]:
                    crds_type_name = "nonuniform"
                    if len(self._TYPE.split("_")) > 1:
                        more = "_".join(self._TYPE.split('_')[1:])
                        crds_type_name += "_" + more
        crds_type = lookup_crds_type(crds_type_name)

        for i, slc in enumerate(slices):
            axis = axes[i]

            if isinstance(slc, slice):
                sss = (slc.start, slc.stop, slc.step)
                if not all(s is None or hasattr(s, "__index__") for s in sss):
                    raise TypeError("bad sss:", sss)

                crd_slc = slice(slc.start, slc.stop, slc.step)

                sliced_clist_item = None

                # if using step on a cell centered field, we need to figure
                # stuff out, since the data && crd slices can't be the same,
                # and it's not as simple as just adding 1 b/c of the stride
                if cc and crd_slc.step not in [None, 1, -1]:
                    crd_cc = crd_arrs_cc[i][slc]
                    crd_nc = 0.5 * (crd_cc[1:] + crd_cc[:-1])

                    allslc = slice(crd_slc.start, crd_slc.stop,
                                   crd_slc.step // np.abs(crd_slc.step))
                    allnc = crd_arrs_nc[i][allslc]
                    crd_nc = np.concatenate([[allnc[0]], crd_nc, [allnc[-1]]])

                    sliced_clist_item = [axis, crd_nc, crd_cc]

                elif cc and crd_slc.stop is not None:
                    if crd_slc.stop >= 0:
                        if crd_slc.step is None or crd_slc.step > 0:
                            newstop = crd_slc.stop + 1
                        else:
                            newstop = crd_slc.stop - 1
                    else:  # slc.stop < 0
                        # this will slice crds nc, so if we're going backward,
                        # the extra crd will be included
                        # print("am i used?", crd_slc.step)
                        newstop = crd_slc.stop
                    crd_slc = slice(crd_slc.start, newstop, crd_slc.step)

                if sliced_clist_item is None:
                    cval = crds_type.native_slice(crd_arrs_nc[i], crd_slc)
                    sliced_clist_item = [axis, cval]

                sliced_clist.append(sliced_clist_item)

            elif hasattr(slc, "__index__"):
                cval = crds_type.native_slice(crd_arrs_nc[i], slc)
                reduced.append([axis, cval])

            elif slc == np.newaxis:
                crd_nc = crds_type.native_slice(crd_arrs_nc[i], slice(None))
                sliced_clist.append([axis, crd_nc])

            else:
                raise TypeError()

        # print("< slice made", slices)
        return slices, sliced_clist, reduced, crds_type

    def make_slice_reduce(self, selection, cc=False):
        """make slice, and reduce dims that were not explicitly sliced"""
        slices, crdlst, reduced, crds_type = self.make_slice(selection, cc=cc)
        # augment slices / reduced
        for i, axis in enumerate(self.axes):
            reduce_axis = False
            if slices[i] == slice(None):
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
        # print("MAKE SLICE REDUCE : slices", slices, "crdlst", crdlst,
        #       "reduced", reduced)
        return slices, crdlst, reduced, crds_type

    def make_slice_keep(self, selection, cc=False):
        """make slice, but put back dims that were explicitly reduced"""
        slices, crdlst, reduced, crds_type = self.make_slice(selection, cc=cc)
        # put reduced dims back, reduced will be in the same order as self.axes
        # since make_slice loops over self.axes to do the slices; this enables
        # us to call insert in the loop
        for axis, loc in reduced:  # pylint: disable=W0612
            axis_ind = self.ind(axis)
            # slices[axis_ind] will be an int not a slice since it was reduced
            loc_ind = slices[axis_ind]
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

                # FIXME: this is a terrible way to hack a uniform
                # xl, xh, nx type clist
                if not self._INIT_FULL_ARRAYS:
                    crd = [crd[0], crd[-1], 2]
            else:
                if loc_ind == -1:
                    crd = crd_nc[-1:]
                    slc = slice(-1, None)
                else:
                    crd = crd_nc[loc_ind:loc_ind + 1]
                    slc = slice(loc_ind, loc_ind + 1)
                # FIXME: this is a terrible way to hack a uniform
                # xl, xh, nx type clist
                if not self._INIT_FULL_ARRAYS:
                    crd = [crd[0], crd[0], 1]

            slices[axis_ind] = slc
            crdlst.insert(axis_ind, [axis, crd])

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
        if slices == [slice(None)] * len(slices):
            return self
        return wrap_crds(crds_type, crdlst, dtype=self.dtype)

    def slice_and_reduce(self, selection, cc=False):
        """Get crds that describe a slice (subset) of this grid. Go
        through, and if the slice didn't touch a dim with only one crd,
        reduce it
        """
        slices, crdlst, reduced, crds_type = self.make_slice_reduce(selection,
                                                                    cc=cc)
        # pass through if nothing happened
        if slices == [slice(None)] * len(slices):
            return self
        return wrap_crds(crds_type, crdlst, dtype=self.dtype)

    def slice_and_keep(self, selection, cc=False):
        slices, crdlst, reduced, crds_type = self.make_slice_keep(selection,
                                                                  cc=cc)
        # pass through if nothing happened
        if slices == [slice(None)] * len(slices):
            return self
        return wrap_crds(crds_type, crdlst, dtype=self.dtype)

    slice_reduce = slice_and_reduce
    slice_keep = slice_and_keep

    def slice_interp(self, selection, cc=False):
        _, crdlst, _, crds_type = self.make_slice_keep(selection, cc=cc)

        axes, selection = self._parse_slice(selection)

        if np.newaxis in selection or Ellipsis in selection:
            raise NotImplementedError("Can't yet do an interpolated slice "
                                      "with newaxis or Ellipsis elements.")

        # for slices that were specified using a float, set that crd
        # to the desired float instead of the nearest crd as does slice_keep
        for i, slc in enumerate(selection):
            slc = vutil.convert_deprecated_floats(slc, "slc")
            if isinstance(slc, string_types):
                assert slc[-1] == 'f'
                val = np.array(float(slc[:-1]), dtype=self.dtype)
                if cc:
                    dxh = 0.5 * (crdlst[i][1][1] - crdlst[i][1][0])
                    crdlst[i][1][0] = val - dxh
                    crdlst[i][1][1] = val + dxh
                else:
                    # FIXME: I don't think this is right
                    crdlst[i][1][0] = val
        return wrap_crds(crds_type, crdlst, dtype=self.dtype)

    def get_crd(self, axis, shaped=False, center="none"):
        """if axis is not specified, return all coords,
        shaped makes axis capitalized and returns ogrid like crds
        shaped is only used if axis == None
        sfx can be none, node, cell, face, edge
        raises KeyError if axis not found
        """
        return self.get_crds([axis], shaped, center)[0]

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
            axes = [a.upper() if shaped else a for a in self.axes]
        if not isinstance(axes, (list, tuple)):
            try:
                axes = [a.upper() if shaped else a for a in axes]
            except TypeError:
                axes = [axes]
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

    def get_clist(self, axes=None, slc=None, full_arrays=True, center='node'):
        """??

        Returns:
            a clist of the coordinates sliced if you wish

        Note:
            I recommend using ``numpy.s_`` for making the slice
        """
        if not full_arrays:
            raise NotImplementedError("you need uniform crds for this")
        if slc is None:
            slc = slice(None)
        if axes is None:
            axes = self.axes

        ret = [[axis, self.get_crd(axis, center=center)[slc]] for axis in axes]

        if slc is None and full_arrays and center == 'node' and self._src_crds_cc:
            for i, axis in enumerate(axes):
                if axis in self._src_crds_cc:
                    ret[i].append(self._src_crds_cc[axis])
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
            arr[i, :] = np.tile(np.repeat(c, np.prod(shape[:i])),
                                np.prod(shape[i + 1:]))
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
        return np.array([x[0] for x in self.get_crds(axes, center=center)])

    def get_xh(self, axes=None, center='node'):
        return np.array([x[-1] for x in self.get_crds(axes, center=center)])

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
    _INIT_FULL_ARRAYS = False

    _nc_linspace_args = None
    _cc_linspace_args = None

    dtype = None

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
                 **kwargs):
        """
        Args:
            init_clist: this should look something like
                [('y', [yl, yh, ny]), ('x', [xl, xh, nx])] unless
                full_arrays is True.
            full_arrays (bool): indicates that clist is given as
                full coordinate arrays, like for non-uniform crds

        Raises:
            ValueError if full_arrays and crds are not uniform
        """
        self.dtype = dtype

        if full_arrays:
            s = ("DEPRECATION...\n"
                 "Full arrays for uniform crds shouldn't be used due to \n"
                 "finite precision errors")
            viscid.logger.warn(s)
            _nc_linspace_args = []  # pylint: disable=unreachable
            for _, arr in init_clist:
                if vutil.isdatetime(arr):
                    raise NotImplementedError("Datetime arrays can't be in "
                                              "uniform crds yet")
                arr = np.asarray(arr)
                if len(arr) > 1:
                    diff = arr[1:] - arr[:-1]
                else:
                    diff = np.array([1, 1])
                # This allclose is the problem... when slicing, it doesn't
                # always pass
                atol = 100 * np.finfo(arr.dtype).eps
                if not np.allclose(diff[0], diff[1:], atol=atol):
                    raise ValueError("Crds are not uniform")
                _nc_linspace_args.append([arr[0], arr[-1], len(arr)])
        else:
            for _, arr in init_clist:
                if len(arr) != 3:
                    raise ValueError("is this a full_array?")
                if vutil.isdatetime([arr[0], arr[1]]):
                    raise NotImplementedError("Datetime arrays can't be in "
                                              "uniform crds yet")
            _nc_linspace_args = [arr for _, arr in init_clist]

        self._nc_linspace_args = _nc_linspace_args
        self._cc_linspace_args = []
        # print("_nc_linspace_args::", _nc_linspace_args)
        for args in _nc_linspace_args:
            half_dx = 0.5 * (args[1] - args[0]) / args[2]
            cc_args = [args[0] + half_dx, args[1] - half_dx, args[2] - 1]
            if cc_args[2] == 0:
                cc_args[2] = 1
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

        # cell centered things
        self.xl_cc = np.array([args[0] for args in self._cc_linspace_args],
                              dtype=self.dtype)
        self.xh_cc = np.array([args[1] for args in self._cc_linspace_args],
                              dtype=self.dtype)
        self.shape_cc = np.array([args[2] for args in self._cc_linspace_args],
                                 dtype='int')
        self.L_cc = self.xh_cc - self.xl_cc
        self.min_dx_cc = self.L_cc / self.shape_cc

        init_clist = [(axis, None) for axis, _ in init_clist]
        super(UniformCrds, self).__init__(init_clist, **kwargs)

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
        return self._pull_out_axes([self.min_dx_nc, self.min_dx_cc], axes,
                                   center=center)

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

    @staticmethod
    def native_slice(crd_arr, slc, center='node', axis=None):
        if axis:
            crd_arr = crd_arr.get_crd(axis, center=center)

        if slc is not None:
            crd_arr = crd_arr[slc]

        try:
            nx = len(crd_arr)
            xl = crd_arr[0]
            xh = crd_arr[-1]
            return [xl, xh, nx]
        except TypeError:
            return crd_arr

    # def _native_slice(self, axis, slc, center='node'):
    #     """Get a sliced crd array to construct crds of same type

    #     Args:
    #         axis: something __getitem__ will understand to pull
    #             out a single coordinate axis
    #         slc (slice): Should be a valid slice of ints

    #     Returns:
    #         A sliced single coordinate array that __init__ understands
    #         by default
    #     """
    #     # inds = list(range(self.shape))
    #     if slc is None:
    #         proxy_crd = self.get_crd(axis, center=center)
    #     else:
    #         proxy_crd = self.get_crd(axis, center=center)[slc]

    #     try:
    #         nx = len(proxy_crd)
    #         xl = proxy_crd[0]
    #         xh = proxy_crd[-1]
    #         return [xl, xh, nx]
    #     except TypeError:
    #         return proxy_crd

    def extend_by_half(self):
        """Extend coordinates half a grid cell in all directions

        Used for turning node centered fields to cell centered without
        changing the data, just the coordinates.

        Returns:
            New coordinates instance with same type as self
        """
        axes = self.axes
        xl, xh = self.get_xl(), self.get_xh()
        dx = (xh - xl) / self.shape_nc
        xl -= 0.5 * dx
        xh += 0.5 * dx
        nx = self.shape_nc + 1
        new_clist = [(ax, [xl[i], xh[i], nx[i]]) for i, ax in enumerate(axes)]
        return type(self)(new_clist)

    def atleast_3d(self, xl=-1, xh=1, cc=False):
        if self.nr_dims >= 3:
            return self
        else:
            axes3 = "xyz"
            new_clist = self.get_clist()
            for ax in axes3[self.nr_dims:]:
                if cc:
                    new_clist.append((ax, [xl, xh, 1]))
                else:
                    x0 = 0.5 * (xl + xh)
                    new_clist.append((ax, [x0, x0, 1]))
            return type(self)(new_clist)

    def get_clist(self, axes=None, slc=None, full_arrays=False, center="node"):
        if full_arrays:
            return super(UniformCrds, self).get_clist(axes=axes, slc=slc,
                                                      center=center)
        if slc is not None:
            raise NotImplementedError("use full_arrays=True with slice != None"
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
                lst.append([ax, ls_args])
        return lst

    def _fill_crds_dict(self):
        assert len(self._nc_linspace_args) == len(self._axes)
        self._src_crds_nc = {}
        for ax, p in zip(self._axes, self._nc_linspace_args):
            self._src_crds_nc[ax] = np.linspace(p[0], p[1], p[2]).astype(self.dtype)
        return super(UniformCrds, self)._fill_crds_dict()


class NonuniformCrds(StructuredCrds):
    _TYPE = "nonuniform"

    def __init__(self, init_clist, full_arrays=True, **kwargs):
        if not full_arrays:
            raise ValueError("did you want Uniform crds?")
        super(NonuniformCrds, self).__init__(init_clist, **kwargs)

    def atleast_3d(self, xl=-1, xh=1, cc=False):
        if self.nr_dims >= 3:
            return self
        else:
            axes3 = ['fill_x', 'fill_y', 'fill_z']
            new_clist = self.get_clist()
            for ax in axes3[self.nr_dims:]:
                if cc:
                    new_clist.append((ax, np.array([xl, xh],
                                                   dtype=self.dtype)))
                else:
                    new_clist.append((ax, np.array([0.5 * (xl + xh)],
                                                   dtype=self.dtype)))
            return type(self)(new_clist)

    def get_clist(self, axes=None, slc=None, full_arrays=True, center="node"):
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

##
## EOF
##
