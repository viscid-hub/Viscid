#!/usr/bin/env python
""" Coordinates get put into fields
The order of coords in the clist should mirror the data layout, as in
if data[iz, iy, ix] (C-order, ix is fastest varying index) then list
should go z, y, x... this is the default order

types:
     "Structured":
         "Rectilinear"
         "Cylindrical"
         "Spherical"
     "Unstructured":
         -> Not Implemented <- """

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
    CENTER = {None: "", "Node": "", "Cell": "cc", "Face": "fc", "Edge": "ec"}
    sfxIXES = list(CENTER.values())

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
        self.clear_crds()
        if init_clist:
            self.set_crds(init_clist)

    @property
    def dim(self):
        return len(self._axes)

    @property
    def axes(self):
        return self._axes

    @property
    def shape(self):
        return self.shape_nc

    @property
    def shape_nc(self):
        return [len(self[ax]) for ax in self.axes]

    @property
    def shape_cc(self):
        return [len(self[ax + "cc"]) for ax in self.axes]

    def clear_crds(self):
        self._crds = {}
        for d in self.axes:
            for sfx in self.sfxIXES:
                self._crds[d + sfx] = None
                self._crds[d.upper() + sfx] = None

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

        # recalculate all cell centers, and refresh face / edges
        sfx = self.CENTER["Cell"]
        for a in self.axes:
            a = self.axis_name(a)  # validate input
            ccarr = 0.5 * (self._crds[a][1:] + self._crds[a][:-1])
            flatarr, openarr = self.ogrid_single(a, ccarr)
            self._crds[a + sfx] = flatarr
            self._crds[a.upper() + sfx] = openarr

        crds_nc = self.get_crd()
        crds_nc_shaped = self.get_crd(shaped=True)
        crds_cc = self.get_crd(center="Cell")
        crds_cc_shaped = self.get_crd(shaped=True, center="Cell")

        # store references to face and edge centers while we're here
        sfx = self.CENTER["Face"]
        for i, a in enumerate(self.axes):
            self._crds[a + sfx] = [None] * len(self.axes)
            self._crds[a.upper() + sfx] = [None] * len(self.axes)
            for j, d in enumerate(self.axes): #pylint: disable=W0612
                if i == j:
                    self._crds[a + sfx][j] = crds_nc[i]
                    self._crds[a.upper() + sfx][j] = crds_nc_shaped[i]
                else:
                    self._crds[a + sfx][j] = crds_cc[i]
                    self._crds[a.upper() + sfx][j] = crds_cc_shaped[i]
        # same as face, but swap nc with cc
        sfx = self.CENTER["Face"]
        for i, a in enumerate(self.axes):
            self._crds[a + sfx] = [None] * len(self.axes)
            self._crds[a.upper() + sfx] = [None] * len(self.axes)
            for j, d in enumerate(self.axes):
                if i == j:
                    self._crds[a + sfx][j] = crds_cc[i]
                    self._crds[a.upper() + sfx][j] = crds_cc_shaped[i]
                else:
                    self._crds[a + sfx][j] = crds_nc[i]
                    self._crds[a.upper() + sfx][j] = crds_nc_shaped[i]

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

    def axis_name(self, axis):
        if isinstance(axis, str):
            if axis in self._crds:
                return axis
            else:
                raise KeyError(axis)
        else:
            return self.axes[axis]

    @staticmethod
    def parse_slice_str(selection):
        """ parse a selection string or dict. a trailing i means the number is
        an index, else numbers are interpreted as coordinates... ex:
        selection = 'y=12.32,z-1.0' is line of data
        closest to (x, y=12.32, z=-1.0)
        selection = {'y=12i:14i,z=1i is the slice [:,12:14,1]
        """
        # parse string to dict if necessary
        if isinstance(selection, dict):
            return selection
        elif selection is None:
            return {}
        elif isinstance(selection, str):
            sel = {}
            for i, s in enumerate(selection.replace('_', ',').split(',')):
                s = s.split('=')

                # this extra : split is for asking for a subset in one
                # dimension
                dim = s[0].strip()
                slclst = s[1].strip().split(':')

                # if number ends in i or j, cast to int as index, else leave
                # as float to denote a crd... also, if it's the 3rd value then
                # it's a stride... cant really have a float stride can we
                for i, val in enumerate(slclst):
                    if len(val) == 0:
                        slclst[i] = None
                    elif val[-1] in ['i', 'j'] or i == 2:
                        slclst[i] = int(val[:-1])
                    else:
                        slclst[i] = float(val)
                sel[dim] = slclst
            return sel

        else:
            raise TypeError()

    def make_slice(self, selection, use_cc=False, rm_len1_dims=False):
        """ use a selection dict that looks like:
        {'x': (0,12), 'y': 0.0, 'z': (-13.0, 13.0)}
        to get a list of slices and crds that one
        could use to wrap a new field...
        return example for "y=3i:6i:2,z=0":
        ([slice(None), slice(3, 5, 2), 32],  # list of data slices
         [['x', array(1, 2, 3)], ['y', array(0, 1)]],  # new clist
         [['z', 0.0]],  # list of coords that are taken out by the slices
        )
        rm_len1_dims will automatically add directions with only
        one coord value to the selection dict. the idea is to auto reduce
        3d to 2d if the data is naturally 2d (one dicection has no depth)
        """
        # TODO: see if np.s_ could clean this up a bit
        selection = self.parse_slice_str(selection)
        slices = [None] * self.dim
        # colapse = [None] * self.dim
        slcrds = [None] * self.dim
        reduced = []

        # go through all axes and see if they are selected
        for dind, axis in enumerate(self.axes):
            if not axis in selection:
                n = len(self[axis + "cc"]) if use_cc else len(self[axis])
                if rm_len1_dims and n == 1:
                    slices[dind] = np.s_[0]
                    slcrds[dind] = None
                    crdval = self[axis + "cc"][0] if use_cc else self[axis][0]
                    reduced.append([axis, crdval])
                else:
                    slices[dind] = slice(None)
                    slcrds[dind] = [axis, self[axis]]
                continue

            sel = selection[axis]

            if isinstance(sel, slice):
                slc = sel

            # expect val to be a list to describe start, stop, stride
            if isinstance(sel, (int, float)):
                sel = [sel]

            if isinstance(sel, (list, tuple)):
                # do plane finding if vals are floats for the first two
                # elements only, the third would be a stride, no lookup
                # necessary
                for i, v in enumerate(sel[:2]):
                    if isinstance(v, float):
                        # in truth, i have no idea if this would work with
                        # decreasing coords
                        # if self[axis][-1] < self[axis][0]:
                        #     raise ValueError("I will not slice decreasing "
                        #                      "coords.")

                        # find index of closest node
                        if use_cc:
                            diff = v - self[axis + "cc"]
                        else:
                            diff = v - self[axis]
                        closest_ind = np.argmin(np.abs(diff))

                        if i == 0:
                            sel[i] = closest_ind  # always a node
                        else:  # i == 1 due to slice sel[:2]
                            sel[i] = closest_ind + 1

                if len(sel) == 1:
                    ind = sel[0]
                    slices[dind] = np.s_[ind]
                    slcrds[dind] = None
                    loc = self[axis + "cc"][ind] if use_cc else self[axis][ind]
                    reduced.append([axis, loc])
                    continue
                elif len(sel) == 2:
                    slc = np.s_[sel[0]:sel[1]]
                elif len(sel) == 3:
                    slc = np.s_[sel[0]:sel[1]:sel[2]]
                else:
                    raise ValueError()

            if use_cc and slc.stop is not None:
                crd_slc = slice(slc.start, slc.stop + 1, slc.step)
            else:
                crd_slc = slc
            slices[dind] = slc
            slcrds[dind] = [axis, self[axis][crd_slc].reshape(-1)]

        # print(slices, slcrds)

        # remove len1_dims
        slcrds = [crd for crd in slcrds if crd is not None]

        # print(slices)
        # print("*** slices: ", slices)
        # print("*** crds: ", slcrds)
        return tuple(slices), slcrds, reduced

    def get_crd(self, axis=None, shaped=False, center=None):
        """ if axis is not specified, return all coords,
        shaped makes axis capitalized and returns ogrid like crds
        shaped is only used if axis == None
        sfx can be None, Node, Cell, Face, Edge """
        
        if axis == None:
            axis = [a.upper() if shaped else a for a in self.axes]

        sfx = self.CENTER[center]
        if isinstance(axis, (list, tuple)):
            return [self._crds[self.axis_name(a) + sfx] for a in axis]
        else:
            return self._crds[self.axis_name(axis) + sfx]

    def get_culled_axes(self, ignore=2):
        """ return list of axes names, but discard axes whose coords have
        length <= ignore... useful for 2d fields that only have 1 cell
        center in the 3rd direction """
        return [name for name in self.axes if len(self[name]) > ignore]

    def get_clist(self, slce=slice(None)):
        """ return a clist of the coordinates sliced if you wish
        I recommend using numpy.s_ for making the slice """
        return [[axis, self.get_crd(axis)[slce]] for axis in self.axes]

    def __getitem__(self, axis):
        """ returns coord identified by axis """
        return self.get_crd(axis)

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
    _axes = ["phi", "theta", "r"]

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
