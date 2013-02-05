#!/usr/bin/env python
# Coordinates get put into fields
# The order of coords in the clist should mirror the data layout, as in
# if data[iz, iy, ix] (C-order, ix is fastest varying index) then list
# should go z, y, x... this is the default order
#
# types:
#      "Structured":
#          "Rectilinear"
#          "Cylindrical"
#          "Spherical"
#      "Unstructured":
#          -> Not Implemented <-

from __future__ import print_function

import numpy as np

from . import vutil


def wrap_crds(typ, clist):
    """  """
    # print(len(clist), clist[0][0], len(clist[0][1]), typ)
    for cls in vutil.subclass_spider(Coordinates):
        if cls.TYPE == typ:
            # return an instance
            return cls(clist)
    raise NotImplementedError("can not decipher crds")


class Coordinates(object):
    TYPE = "None"

    _crds = None

    def __init__(self):
        pass


class StructuredCrds(Coordinates):
    TYPE = "Structured"

    _axes = ["z", "y", "x"]
    _dim = 3

    has_cc = None

    def __init__(self, init_clist=None, has_cc=True, **kwargs):
        """ if caled with an init_clist, then the coordinate names
        are taken from this list """
        super(StructuredCrds, self).__init__(**kwargs)
        self.has_cc = has_cc
        if init_clist:
            self._axes = [d[0].lower() for d in init_clist]
            self._dim = len(self._axes)
        self.clear_crds()
        if init_clist:
            self.set_crds(init_clist)

    @property
    def dim(self):
        return self._dim

    @property
    def axes(self):
        return self._axes

    @property
    def shape(self):
        return [len(self[ax]) for ax in self.axes]

    @property
    def shape_cc(self):
        return [len(self[ax + "cc"]) for ax in self.axes]

    def clear_crds(self):
        self._crds = {}
        for d in self.axes:
            self._crds[d] = None
            self._crds[d.upper()] = None
            if self.has_cc:
                self._crds[d + 'cc'] = None
                self._crds[d.upper() + 'cc'] = None

    def set_crds(self, clist):
        """ called with a list of lists:
        (('x', ndarray), ('y',ndarray), ('z',ndarray))
        Note: input crds are assumed to be node centered
        """
        # print(clist)
        for axis, arr in clist:
            # make axis into string representation
            axis = self.axis_name(axis)
            ind = self.ind(axis)
            arr = np.array(arr, dtype=arr.dtype.name)
            flatarr, openarr = self.ogrid_single(ind, arr)
            self._crds[axis.lower()] = flatarr
            self._crds[axis.upper()] = openarr
            if self.has_cc:
                self._calc_cc_crds([axis])

    def _calc_cc_crds(self, axes):
        for a in axes:
            a = self.axis_name(a)  # validate input
            ccarr = 0.5 * (self._crds[a][1:] + self._crds[a][:-1])
            flatarr, openarr = self.ogrid_single(a, ccarr)
            self._crds[a + "cc"] = flatarr
            self._crds[a.upper() + "cc"] = openarr

    def ogrid_single(self, axis, arr):
        """ returns (flat array, open array) """
        i = self.ind(axis)
        shape = [1] * self.dim
        shape[i] = -1

        # print(arr.shape)

        if len(arr.shape) == 1:
            return arr, np.reshape(arr, shape)
        elif len(arr.shape) == self.dim:
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

    def axis_name(self, axis, shaped=False):
        if isinstance(axis, str):
            if axis in self._crds:
                nom = axis
            else:
                raise KeyError(axis)
        else:
            nom = self.axes[axis]

        if shaped:
            return nom.upper()
        else:
            return nom

    @staticmethod
    def parse_slice_str(selection):
        """ parse a selection string or dict. a trailing i means the number is
        an index, else numbers are interpreted as coordinates... ex:
        selection = 'y=12.32,z-1.0' is line of data
        closest to (x, y=12.32, z=-1.0)
        selection = {'y=12i:14i,z=1i is the slice [:,12:14,1]
        """
        # parse string to dict if necessary
        if isinstance(selection, str):
            sel = {}
            for i, s in enumerate(selection.replace('_', ',').split(',')):
                s = s.split('=')

                # this extra : split is for asking for a subset in one
                # dimension
                dim = s[0].strip()
                slclst = s[1].strip().split(':')

                # if number ends in i or j, cast to int as index, else leave
                # as float to denote a crd
                for i, val in enumerate(slclst):
                    if val[-1] in ['i', 'j']:
                        slclst[i] = int(val[:-1])
                    else:
                        slclst[i] = float(val)
                sel[dim] = slclst
            return sel
        elif isinstance(selection, dict):
            return selection
        else:
            raise TypeError()

    def make_slice(self, selection, use_cc=False):
        """ use a selection dict that looks like:
        {'x': (0,12), 'y': 0.0, 'z': (-13.0, 13.0)}
        to get a list of slices and crds that one
        could use to wrap a new field
        """
        selection = self.parse_slice_str(selection)
        slices = [None] * self.dim
        slcrds = [None] * self.dim

        # if use_cc:
        #     assert(self.has_cc)

        # go through all axes and see if they are selected
        for dind, axis in enumerate(self.axes):
            if not axis in selection:
                slices[dind] = slice(None)
                slcrds[dind] = [axis, self[axis]]
                continue

            val = selection[axis]
            if isinstance(val, slice):
                sl = val
                slices[dind] = sl
                slcrds[dind] = [axis, self[axis][sl]]
                continue

            # expect val to be a list to describe start, stop, stride
            if isinstance(val, (int, float)):
                val = [val]

            # do plane finding if vals are floats for the first two
            # elements only, the third would be a stride, no lookup necessary
            for i, v in enumerate(val[:2]):
                if isinstance(v, float):
                    crdarr = self[axis + "cc"] if use_cc else self[axis]
                    val[i] = np.argmin(np.abs(crdarr - v))
            #val[0] = min(val[0] - 1, 0)

            if len(val) == 1:
                #sl = slice(val[0], val[0] + 2)
                sl = val[0]
            elif len(val) == 2:
                #sl = slice(val[0], val[1] + 2)
                sl = slice(val[0], val[1])
            elif len(val) == 3:
                #sl = slice(val[0], val[1] + 2, val[2])
                sl = slice(val[0], val[1], val[2])
            else:
                raise ValueError()

            slices[dind] = sl
            slcrds[dind] = [axis, self._crds[axis][sl].reshape(-1)]

        # auto trim last dimension
        for i in range(self.dim - 1, -1, -1):
            if slcrds[i][1].size <= 1:
                # print(slcrds)
                slcrds.pop(i)
        # print(slices)
        return tuple(slices), slcrds

    def get_shape(self, axis):
        """ supports cc as well as normal crds """
        if isinstance(axis, (list, tuple)):
            return [len(self[ax]) for ax in axis]
        else:
            return len(self[axis])

    def get_nc(self, axis=None, shaped=False):
        if axis == None:
            axis = self.axes
        if isinstance(axis, (list, tuple)):
            return [self._crds[self.axis_name(a, shaped)] for a in axis]
        else:
            return self._crds[self.axis_name(axis, shaped)]


    def get_cc(self, axis=None, shaped=False):
        if axis == None:
            axis = self.axes
        if isinstance(axis, (list, tuple)):
            return [self._crds[self.axis_name(a, shaped) + "cc"] for a in axis]
        else:
            return self._crds[self.axis_name(axis, shaped) + "cc"]

    def get_clist(self, slce=slice(None)):
        """ return a clist of the coordinates sliced if you wish
        I recommend using numpy.s_ for making the slice """
        return [[axis, self.get_nc(axis)[slce]] for axis in self.axes]

    def __getitem__(self, axis):
        """ returns coord  """
        return self.get_nc(axis)

    def __setitem__(self, axis, arr):
        return self.set_crds((axis, arr))

    def __contains__(self, item):
        return item in self._crds


class RectilinearCrds(StructuredCrds):
    TYPE = "Rectilinear"
    _axes = ["z", "y", "x"]

    def __init__(self, init_clist=None, **kwargs):
        super(RectilinearCrds, self).__init__(init_clist=init_clist, **kwargs)


class CylindricalCrds(StructuredCrds):
    TYPE = "Cylindrical"
    _axes = ["z", "theta", "r"]

    def __init__(self, init_clist=None, **kwargs):
        super(CylindricalCrds, self).__init__(init_clist=init_clist, **kwargs)


class SphericalCrds(StructuredCrds):
    TYPE = "Spherical"
    _axes = ["z", "theta", "r"]

    def __init__(self, init_clist=None, **kwargs):
        super(SphericalCrds, self).__init__(init_clist=init_clist, **kwargs)


class UnstructuredCrds(Coordinates):
    TYPE = "Unstructured"

    def __init__(self, **kwargs):
        super(UnstructuredCrds, self).__init__(**kwargs)
        raise NotImplementedError()

##
## EOF
##
