"""For fields that consist of a list of fields + an AMRSkeleton

Note:
    An AMRField is NOT a subclass of Field, but it is a giant wrapper
    around a lot of Field functionality.
"""

from __future__ import print_function
import numpy as np

import viscid
# from viscid.compat import string_types
from viscid.field import Field


__all__ = ["is_list_of_fields"]


def is_list_of_fields(lst):
    """is a sequence a sequence of Field objects?"""
    for item in lst:
        if not isinstance(item, Field):
            return False
    return True


class _FieldListCallableAttrWrapper(object):
    objs = None
    attrname = None
    post_func = None

    def __init__(self, objs, attrname, post_func=None):
        # print(">>> runtime wrapping:", attrname)
        for o in objs:
            if not hasattr(o, attrname):
                raise AttributeError("{0} has no attribute {1}"
                                     "".format(o, attrname))
        self.objs = objs
        self.attrname = attrname
        self.post_func = post_func

    def __call__(self, *args, **kwargs):
        lst = [getattr(o, self.attrname)(*args, **kwargs) for o in self.objs]
        if self.post_func:
            return self.post_func(lst)
        else:
            return lst


class AMRField(object):
    """Field-like

    Contains an AMRSkeleton and a list of Fields. This mimiks a Field,
    but it is NOT a subclass of Field. Many methods of Field are
    wrapped and return a new AMRField.

    If an attribute of Field is not explicitly wrapped, this class will
    try to runtime-wrap that method and return a new AMRField or a list
    containing the result. This will not work for special methods since
    python will not send those through __getattr__ or __getattribute__.
    """
    _TYPE = "amr"
    skeleton = None
    patches = None
    nr_patches = None

    def __init__(self, fields, skeleton):
        if not is_list_of_fields(fields):
            raise TypeError("AMRField can only contain Fields:", fields)
        self.skeleton = skeleton
        self.patches = fields
        self.nr_patches = len(fields)

    @property
    def xl(self):
        if self.skeleton:
            return np.min(self.skeleton.xl, axis=0)
        else:
            return np.min(np.vstack([p.xl for p in self.patches]), axis=0)

    @property
    def xh(self):
        if self.skeleton:
            return np.max(self.skeleton.xh, axis=0)
        else:
            return np.max(np.vstack([p.xh for p in self.patches]), axis=0)

    def get_slice_extent(self, selection):
        extent = self.patches[0]._src_crds.get_slice_extent(selection)
        for i in range(3):
            if np.isnan(extent[0, i]):
                extent[0, i] = self.xl[i]
            if np.isnan(extent[1, i]):
                extent[1, i] = self.xh[i]
        return extent

    ###########
    ## slicing
    def _prepare_amr_slice(self, selection):
        """ return list of patches that contain selection """
        # FIXME: it's not good to reach in to src_field[0]'s private methods
        # like this, but it's also not good to implement these things twice
        # print("??", len(self.patches))
        if len(self.patches) == 0:
            raise ValueError("AMR field must contain patches to be slicable")
        sel_list, _ = self.patches[0]._prepare_slice(selection)
        try:
            extent = self.patches[0]._src_crds.get_slice_extent(selection)
        except RuntimeError:
            raise RuntimeError("Slicing by global index is poorly defined "
                               "for AMR fields; selection = '{0}'"
                               "".format(selection))

        inds = []
        # these are patches that look like they contain selection
        # but might not due to finite precision errors when
        # calculating xh
        maybe = []

        # detect dimensions that only have one cell (or node) and allow any
        # slice in that direction, this helps cases where a field mab be
        # defined or y = [-0.01], but the user tries to slice by 'y=0j'
        all_xl_nc = np.array([patch.crds.xl_nc for patch in self.patches])
        all_xh_nc = np.array([patch.crds.xh_nc for patch in self.patches])
        dim_is_2d = np.all(all_xl_nc == all_xh_nc, axis=0)
        for idim, is_2d in enumerate(dim_is_2d):
            if is_2d:
                all_xl_nc[:, idim] = -np.inf
                all_xh_nc[:, idim] = np.inf

        for i, fld in enumerate(self.patches):
            # - if xl - atol > the extent of the slice in any direction, then
            #   there's no overlap
            # - if xh <= the lower corner of the slice in any direction, then
            #   there's no overlap
            # the atol and equals are done to match cases where extent overlaps
            # the lower corner, but not the upper corner

            # logic goes this way cause extent has NaNs in
            # dimensions that aren't specified in selection... super-kludge

            # also, temporarily disable warnings on NaNs in numpy
            xl_nc = all_xl_nc[i]
            xh_nc = all_xh_nc[i]
            invalid_err_level = np.geterr()['invalid']
            np.seterr(invalid='ignore')
            atol = 100 * np.finfo(xl_nc.dtype).eps
            if (not np.any(np.logical_or(xl_nc - atol > extent[1],
                                         xh_nc <= extent[0]))):
                if np.any(np.isclose(fld.crds.xh_nc, extent[0], atol=atol)):
                    maybe.append(i)
                else:
                    inds.append(i)
            np.seterr(invalid=invalid_err_level)
        # if we found some maybes, but no real hits, then use the maybes
        if maybe and not inds:
            inds = maybe

        if len(inds) == 0:
            viscid.logger.error("selection {0} not in any patch @ time {1}"
                                "".format(selection, self.patches[0].time))
            if self.skeleton:
                s = "    xl= {0}".format(self.skeleton.global_xl)
                viscid.logger.error(s)
                s = "    xh= {0}".format(self.skeleton.global_xh)
                viscid.logger.error(s)
            inds = None
            flds = None
        elif len(inds) == 1:
            inds = inds[0]
            flds = self.patches[inds]
        else:
            flds = [self.patches[i] for i in inds]

        return flds, inds

    def _finalize_amr_slice(self, fld_lst):  # pylint: disable=no-self-use
        skeleton = None  # FIXME
        for fld in fld_lst:
            if isinstance(fld, (int, float, np.number)):
                m = ("Trying to make an AMRField where 1+ patches "
                     "is just a number... You probably slice_reduced "
                     "a field down to a scalar value")
                viscid.logger.error(m)

        # prune out fields that got sliced to smithereens
        for i in reversed(range(len(fld_lst))):
            if fld_lst[i].size == 0:
                viscid.logger.debug("finalize amr slice, remove size 0 patch")
                fld_lst.pop(i)

        # look for uneven dimensions in the results and fill them back out
        axes = [patch.crds.axes for patch in fld_lst]
        ndims = np.array([len(ax) for ax in axes])
        ref_axes = axes[np.argmax(ndims)]
        ref_ndims = np.max(ndims)

        for i, fld in enumerate(fld_lst):
            if ndims[i] < ref_ndims:
                putback = []
                for _ax in ref_axes:
                    if _ax in axes[i]:
                        putback.append('{0}=:'.format(_ax))
                    else:
                        putback.append('{0}=newaxis'.format(_ax))
                putback_slice = ','.join(putback)
                viscid.logger.debug("putback: {0}".format(putback_slice))
                fld_lst[i] = fld[putback_slice]

        return AMRField(fld_lst, skeleton)

    def patch_indices(self, selection):
        """get the indices of the patches that overlap selection
        Args:
            selection (slice, str): anything that can slice a field

        Returns:
            list of indices
        """
        _, inds = self._prepare_amr_slice(selection)
        return inds

    def slice(self, selection):
        fld_lst, _ = self._prepare_amr_slice(selection)
        if not isinstance(fld_lst, list):
            return fld_lst.slice(selection)
        fld_lst = [fld.slice(selection) for fld in fld_lst]
        return self._finalize_amr_slice(fld_lst)

    def slice_reduce(self, selection):
        fld_lst, _ = self._prepare_amr_slice(selection)
        if not isinstance(fld_lst, list):
            return fld_lst.slice_reduce(selection)
        fld_lst = [fld.slice_reduce(selection) for fld in fld_lst]
        return self._finalize_amr_slice(fld_lst)

    def slice_and_keep(self, selection):
        fld_lst, _ = self._prepare_amr_slice(selection)
        if not isinstance(fld_lst, list):
            return fld_lst.slice_and_keep(selection)
        fld_lst = [fld.slice_and_keep(selection) for fld in fld_lst]
        return self._finalize_amr_slice(fld_lst)

    def interpolated_slice(self, selection):
        fld_lst, _ = self._prepare_amr_slice(selection)
        if not isinstance(fld_lst, list):
            raise RuntimeError("can't interpolate to that slice?")

        ret_lst = [fld.interpolated_slice(selection) for fld in fld_lst]
        return self._finalize_amr_slice(ret_lst)

    ###################
    ## special methods

    def __getitem__(self, item):
        return self.slice(item)

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def __delitem__(self, item):
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        """clear all caches"""
        for blk in self.patches:
            blk.clear_cache()
        return None

    def wrap_field_method(self, attrname, *args, **kwargs):
        """Wrap methods whose args are Fields and return a Field"""
        # make sure all args have same number of patches as self
        is_field = [None] * len(args)
        for i, arg in enumerate(args):
            try:
                if arg.nr_patches != self.nr_patches and arg.nr_patches != 1:
                    raise ValueError("AMR fields in math operations must "
                                     "have the same number of patches")
                is_field[i] = True
            except AttributeError:
                is_field[i] = False

        lst = [None] * self.nr_patches
        other = [None] * len(args)
        # FIXME: There must be a better way
        for i, patch in enumerate(self.patches):
            for j, arg in enumerate(args):
                if is_field[j]:
                    try:
                        other[j] = arg.patches[i]
                    except IndexError:
                        other[j] = arg.patches[0]
                else:
                    other[j] = arg
            lst[i] = getattr(patch, attrname)(*other, **kwargs)

        if np.asarray(lst[0]).size == 1:
            # operation reduced to scalar
            arr = np.array(lst)
            return getattr(arr, attrname)(**kwargs)
        else:
            return AMRField(lst, self.skeleton)

    # TODO: as of numpy 1.10, this will be called on ufuncs... this
    #       will help some of the FIXMEs in __array__
    # def __numpy_ufunc__(self, ufunc, method, i, inputs, **kwargs):
    #     pass

    def __array__(self, *args, **kwargs):
        # FIXME: This is heinously inefficient for large arrays because it
        #        makes an  copy of all the arrays... but I don't see
        #        a way around this because ufuncs expect a single array

        # FIXME: adding a dimension to the arrays will break cases like
        #        np.sum(fld, axis=-1), cause that -1 will now be the patch
        #        dimension

        patches = [patch.__array__(*args, **kwargs) for patch in self.patches]
        for i, patch in enumerate(patches):
            patches[i] = np.expand_dims(patch, 0)
        # the vstack will copy all the arrays, this is what __numpy_ufunc__
        # will be able to avoid
        arr = np.vstack(patches)
        # roll the patch dimension to the last dimension... this is for ufuncs
        # that take an axis argument... this way axis will only be confused
        # if it's negative, this is the main reason to use __numpy_ufunc__
        # in the future
        arr = np.rollaxis(arr, 0, len(arr.shape))
        return arr

    def __array_wrap__(self, arr, context=None):  # pylint: disable=unused-argument
        # print(">> __array_wrap__", arr.shape, context)
        flds = []
        for i in range(arr.shape[-1]):
            patch_arr = arr[..., i]
            fld = self.patches[i].__array_wrap__(patch_arr, context=context)
            flds.append(fld)
        return AMRField(flds, self.skeleton)

    def __add__(self, other):
        return self.wrap_field_method("__add__", other)
    def __sub__(self, other):
        return self.wrap_field_method("__sub__", other)
    def __mul__(self, other):
        return self.wrap_field_method("__mul__", other)
    def __div__(self, other):
        return self.wrap_field_method("__div__", other)
    def __truediv__(self, other):
        return self.wrap_field_method("__truediv__", other)
    def __floordiv__(self, other):
        return self.wrap_field_method("__floordiv__", other)
    def __mod__(self, other):
        return self.wrap_field_method("__mod__", other)
    def __divmod__(self, other):
        return self.wrap_field_method("__divmod__", other)
    def __pow__(self, other):
        return self.wrap_field_method("__pow__", other)
    def __lshift__(self, other):
        return self.wrap_field_method("__lshift__", other)
    def __rshift__(self, other):
        return self.wrap_field_method("__rshift__", other)
    def __and__(self, other):
        return self.wrap_field_method("__and__", other)
    def __xor__(self, other):
        return self.wrap_field_method("__xor__", other)
    def __or__(self, other):
        return self.wrap_field_method("__or__", other)

    def __radd__(self, other):
        return self.wrap_field_method("__radd__", other)
    def __rsub__(self, other):
        return self.wrap_field_method("__rsub__", other)
    def __rmul__(self, other):
        return self.wrap_field_method("__rmul__", other)
    def __rdiv__(self, other):
        return self.wrap_field_method("__rdiv__", other)
    def __rtruediv__(self, other):
        return self.wrap_field_method("__rtruediv__", other)
    def __rfloordiv__(self, other):
        return self.wrap_field_method("__rfloordiv__", other)
    def __rmod__(self, other):
        return self.wrap_field_method("__rmod__", other)
    def __rdivmod__(self, other):
        return self.wrap_field_method("__rdivmod__", other)
    def __rpow__(self, other):
        return self.wrap_field_method("__rpow__", other)

    def __iadd__(self, other):
        return self.wrap_field_method("__iadd__", other)
    def __isub__(self, other):
        return self.wrap_field_method("__isub__", other)
    def __imul__(self, other):
        return self.wrap_field_method("__imul__", other)
    def __idiv__(self, other):
        return self.wrap_field_method("__idiv__", other)
    def __itruediv__(self, other):
        return self.wrap_field_method("__itruediv__", other)
    def __ifloordiv__(self, other):
        return self.wrap_field_method("__ifloordiv__", other)
    def __imod__(self, other):
        return self.wrap_field_method("__imod__", other)
    def __ipow__(self, other):
        return self.wrap_field_method("__ipow__", other)

    def __neg__(self):
        return self.wrap_field_method("__neg__")
    def __pos__(self):
        return self.wrap_field_method("__pos__")
    def __abs__(self):
        return self.wrap_field_method("__abs__")
    def __invert__(self):
        return self.wrap_field_method("__invert__")

    def __lt__(self, other):
        return self.wrap_field_method("__lt__", other)
    def __le__(self, other):
        return self.wrap_field_method("__le__", other)
    def __eq__(self, other):
        return self.wrap_field_method("__eq__", other)
    def __ne__(self, other):
        return self.wrap_field_method("__ne__", other)
    def __gt__(self, other):
        return self.wrap_field_method("__gt__", other)
    def __ge__(self, other):
        return self.wrap_field_method("__ge__", other)

    def any(self, **kwargs):
        return self.wrap_field_method("any", **kwargs)
    def all(self, **kwargs):
        return self.wrap_field_method("all", **kwargs)
    def argmax(self, **kwargs):
        return self.wrap_field_method("argmax", **kwargs)
    def argmin(self, **kwargs):
        return self.wrap_field_method("argmin", **kwargs)
    def argpartition(self, **kwargs):
        return self.wrap_field_method("argpartition", **kwargs)
    def argsort(self, **kwargs):
        return self.wrap_field_method("argsort", **kwargs)
    def clip(self, **kwargs):
        return self.wrap_field_method("clip", **kwargs)
    def conj(self, **kwargs):
        return self.wrap_field_method("conj", **kwargs)
    def conjugate(self, **kwargs):
        return self.wrap_field_method("conjugate", **kwargs)
    def cumprod(self, **kwargs):
        return self.wrap_field_method("cumprod", **kwargs)
    def cumsum(self, **kwargs):
        return self.wrap_field_method("cumsum", **kwargs)
    def max(self, **kwargs):
        return self.wrap_field_method("max", **kwargs)
    def mean(self, **kwargs):
        return self.wrap_field_method("mean", **kwargs)
    def min(self, **kwargs):
        return self.wrap_field_method("min", **kwargs)
    def nonzero(self, **kwargs):
        return self.wrap_field_method("nonzero", **kwargs)
    def partition(self, **kwargs):
        return self.wrap_field_method("partition", **kwargs)
    def prod(self, **kwargs):
        return self.wrap_field_method("prod", **kwargs)
    def ptp(self, **kwargs):
        return self.wrap_field_method("ptp", **kwargs)
    def round(self, **kwargs):
        return self.wrap_field_method("round", **kwargs)
    def std(self, **kwargs):
        return self.wrap_field_method("std", **kwargs)
    def sum(self, **kwargs):
        return self.wrap_field_method("sum", **kwargs)

    def __getattr__(self, name):
        # define a callback to finalize
        # print("!! getting attr::", name)
        if callable(getattr(self.patches[0], name)):
            def _wrap(lst):
                try:
                    return AMRField(lst, self.skeleton)
                except TypeError:
                    return lst
            return _FieldListCallableAttrWrapper(self.patches, name, _wrap)
        else:
            # return [getattr(fld, name) for fld in self.patches]
            ret0 = getattr(self.patches[0], name)
            # Check that all patches have the same value. Maybe this should
            # have a debugging flag attached to it since it will take time.
            try:
                all_same = all(getattr(blk, name) == ret0
                               for blk in self.patches[1:])
            except ValueError:
                all_same = all(np.all(getattr(blk, name) == ret0)
                               for blk in self.patches[1:])
            if not all_same:
                raise ValueError("different patches of the AMRField have "
                                 "different values for attribute: {0}"
                                 "".format(name))
            return ret0

##
## EOF
##
