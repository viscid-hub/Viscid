"""For fields that consist of a list of fields + an AMRSkeleton

Note:
    An AMRField is NOT a subclass of Field, but it is a giant wrapper
    around a lot of Field functionality.
"""

import numpy as np

from viscid.compat import string_types
from viscid.field import Field


def is_list_of_fields(lst):
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
    blocks = None
    nr_blocks = None

    def __init__(self, fields, skeleton):
        if not is_list_of_fields(fields):
            raise TypeError("AMRField can only contain Fields:", fields)
        self.skeleton = skeleton
        self.blocks = fields
        self.nr_blocks = len(fields)

    ###########
    ## slicing
    def _prepare_amr_slice(self, selection):
        """ return list of blocks that contain selection """
        # FIXME: it's not good to reach in to src_field[0]'s private methods
        # like this, but it's also not good to implement these things twice
        # print("??", len(self.blocks))
        if len(self.blocks) == 0:
            raise ValueError("AMR field must contain blocks to be slicable")
        selection, _ = self.blocks[0]._prepare_slice(selection)
        extent = self.blocks[0]._src_crds.get_slice_extent(selection)

        ret = []
        for fld in self.blocks:
            # logic goes this way cause extent has NaNs in
            # dimensions that aren't specified in selection... super-kludge
            # print("comparing::", fld.crds.xl_nc, fld.crds.xh_nc, "extent", extent[0], extent[1])
            if (not np.any(np.logical_or(fld.crds.xl_nc > extent[1],
                                         fld.crds.xh_nc <= extent[0]))):
                # print("appending")
                ret.append(fld)

        if len(ret) == 1:
            return ret[0]
        return ret

    def _finalize_amr_slice(self, fld_lst):  # pylint: disable=no-self-use
        skeleton = None  # FIXME
        for fld in fld_lst:
            if isinstance(fld, (int, float, np.number)):
                m = ("Trying to make an AMRField where 1+ blocks "
                     "is just a number... You probably slice_reduced "
                     "a field down to a scalar value")
                print("Warning:", m)
        return AMRField(fld_lst, skeleton)

    def slice(self, selection):
        fld_lst = self._prepare_amr_slice(selection)
        if not isinstance(fld_lst, list):
            return fld_lst.slice(selection)
        fld_lst = [fld.slice(selection) for fld in fld_lst]
        return self._finalize_amr_slice(fld_lst)

    def slice_reduce(self, selection):
        fld_lst = self._prepare_amr_slice(selection)
        if not isinstance(fld_lst, list):
            return fld_lst.slice_reduce(selection)
        fld_lst = [fld.slice_reduce(selection) for fld in fld_lst]
        return self._finalize_amr_slice(fld_lst)

    def slice_and_keep(self, selection):
        fld_lst = self._prepare_amr_slice(selection)
        if not isinstance(fld_lst, list):
            return fld_lst.slice_and_keep(selection)
        fld_lst = [fld.slice_and_keep(selection) for fld in fld_lst]
        return self._finalize_amr_slice(fld_lst)

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

    def __exit__(self, typ, value, traceback):
        """ unload the data """
        for blk in self.blocks:
            blk.unload()
        return None

    def wrap_special_method(self, attrname, *args, **kwargs):
        # print(">>> wrap_special_method:", attrname)
        lst = []
        for fld in self.blocks:
            lst.append(getattr(fld, attrname)(*args, **kwargs))
        return AMRField(lst, self.skeleton)

    def __array__(self, *args, **kwargs):  # pylint: disable=unused-argument,no-self-use
        raise NotImplementedError("AMRFields can not make a single ndarray")

    def __add__(self, other):
        return self.wrap_special_method("__add__", other)
    def __sub__(self, other):
        return self.wrap_special_method("__sub__", other)
    def __mul__(self, other):
        return self.wrap_special_method("__mul__", other)
    def __div__(self, other):
        return self.wrap_special_method("__div__", other)
    def __truediv__(self, other):
        return self.wrap_special_method("__truediv__", other)
    def __floordiv__(self, other):
        return self.wrap_special_method("__floordiv__", other)
    def __mod__(self, other):
        return self.wrap_special_method("__mod__", other)
    def __divmod__(self, other):
        return self.wrap_special_method("__divmod__", other)
    def __pow__(self, other):
        return self.wrap_special_method("__pow__", other)
    def __lshift__(self, other):
        return self.wrap_special_method("__lshift__", other)
    def __rshift__(self, other):
        return self.wrap_special_method("__rshift__", other)
    def __and__(self, other):
        return self.wrap_special_method("__and__", other)
    def __xor__(self, other):
        return self.wrap_special_method("__xor__", other)
    def __or__(self, other):
        return self.wrap_special_method("__or__", other)

    def __radd__(self, other):
        return self.wrap_special_method("__radd__", other)
    def __rsub__(self, other):
        return self.wrap_special_method("__rsub__", other)
    def __rmul__(self, other):
        return self.wrap_special_method("__rmul__", other)
    def __rdiv__(self, other):
        return self.wrap_special_method("__rdiv__", other)
    def __rtruediv__(self, other):
        return self.wrap_special_method("__rtruediv__", other)
    def __rfloordiv__(self, other):
        return self.wrap_special_method("__rfloordiv__", other)
    def __rmod__(self, other):
        return self.wrap_special_method("__rmod__", other)
    def __rdivmod__(self, other):
        return self.wrap_special_method("__rdivmod__", other)
    def __rpow__(self, other):
        return self.wrap_special_method("__rpow__", other)

    def __iadd__(self, other):
        return self.wrap_special_method("__iadd__", other)
    def __isub__(self, other):
        return self.wrap_special_method("__isub__", other)
    def __imul__(self, other):
        return self.wrap_special_method("__imul__", other)
    def __idiv__(self, other):
        return self.wrap_special_method("__idiv__", other)
    def __itruediv__(self, other):
        return self.wrap_special_method("__itruediv__", other)
    def __ifloordiv__(self, other):
        return self.wrap_special_method("__ifloordiv__", other)
    def __imod__(self, other):
        return self.wrap_special_method("__imod__", other)
    def __ipow__(self, other):
        return self.wrap_special_method("__ipow__", other)

    def __neg__(self):
        return self.wrap_special_method("__neg__", )
    def __pos__(self):
        return self.wrap_special_method("__pos__", )
    def __abs__(self):
        return self.wrap_special_method("__abs__", )
    def __invert__(self):
        return self.wrap_special_method("__invert__", )

    def any(self):
        it = (fld.any() for fld in self.blocks)
        return any(it)
    def all(self):
        it = (fld.all() for fld in self.blocks)
        return all(it)

    def __lt__(self, other):
        return self.wrap_special_method("__lt__", other)
    def __le__(self, other):
        return self.wrap_special_method("__le__", other)
    def __eq__(self, other):
        return self.wrap_special_method("__eq__", other)
    def __ne__(self, other):
        return self.wrap_special_method("__ne__", other)
    def __gt__(self, other):
        return self.wrap_special_method("__gt__", other)
    def __ge__(self, other):
        return self.wrap_special_method("__ge__", other)

    def __getattr__(self, name):
        # define a callback to finalize
        # print("!! getting attr::", name)
        if callable(getattr(self.blocks[0], name)):
            def _wrap(lst):
                try:
                    return AMRField(lst, self.skeleton)
                except TypeError:
                    return lst
            return _FieldListCallableAttrWrapper(self.blocks, name, _wrap)
        else:
            return [getattr(fld, name) for fld in self.blocks]


##
## EOF
##
