""" Container for grid coordinates

Coordinates primarily go into Field objects. The order of coords in
the clist should mirror the data layout, as in if data[iz, iy, ix]
(C-order, ix is fastest varying index) then list should go z, y, x...
this is the default order

types:
     "Structured":
         "nonuniform_cartesian"
         "Cylindrical"
         "Spherical"
     "Unstructured":
         -> Not Implemented <-
"""

from __future__ import print_function
# from timeit import default_timer as time
import itertools

import numpy as np

from viscid import vutil


def wrap_crds(typ, clist):
    """  """
    # print(len(clist), clist[0][0], len(clist[0][1]), typ)
    for cls in vutil.subclass_spider(Coordinates):
        if cls.istype(typ):
            # return an instance
            return cls(clist)
    raise NotImplementedError("can not decipher crds")


class Coordinates(object):
    _TYPE = "none"

    __crds = None

    @property
    def type(self):
        return self._TYPE

    @classmethod
    def istype(cls, type_str):
        return cls._TYPE == type_str.lower()

    def is_spherical(self):
        return "spherical" in self._TYPE

    def __init__(self):
        pass

    def as_coordinates(self):
        return self


class StructuredCrds(Coordinates):
    _TYPE = "structured"
    _CENTER = {"none": "", "node": "nc", "cell": "cc",
              "face": "fc", "edge": "ec"}
    SUFFIXES = list(_CENTER.values())

    _axes = ["z", "y", "x"]

    _src_crds_nc = None

    transform_funcs = None
    has_cc = None

    def __init__(self, init_clist, has_cc=True, transform_funcs=None,
                 **kwargs):
        """ if caled with an init_clist, then the coordinate names
        are taken from this list """
        super(StructuredCrds, self).__init__(**kwargs)
        self.has_cc = has_cc

        if transform_funcs is not None:
            self.transform_funcs = transform_funcs

        if init_clist:
            self._axes = [d[0].lower() for d in init_clist]
        self.clear_crds()
        if init_clist:
            self.set_crds(init_clist)

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
    def _crds(self):
        "!!! BIG NOTE: THIS PROPERTY IS STILL PRIVATE !!!"
        if self.__crds is None:
            self._fill_crds_dict()
        return self.__crds

    def clear_crds(self):
        self._src_crds_nc = {}
        self._purge_cache()
        # for d in self.axes:
        #     for sfx in self.SUFFIXES:
        #         self.__crds[d + sfx] = None
        #         self.__crds[d.upper() + sfx] = None

    def _purge_cache(self):
        self.__crds = None

    def unload(self):
        """ does not guarentee that the memory will be freed """
        self._purge_cache()

    def set_crds(self, clist):
        """ called with a list of lists:
        (('x', ndarray), ('y',ndarray), ('z',ndarray))
        Note: input crds are assumed to be node centered
        """
        for axis, data in clist:
            # axis = self.axis_name(axis)
            if not axis in self._axes:
                raise KeyError()
            # ind = self.ind(axis)
            self._src_crds_nc[axis.lower()] = data

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

            if self.transform_funcs is not None:
                if axis in self.transform_funcs:
                    arr = self.transform_funcs[axis](self, arr)
                elif ind in self.transform_funcs:
                    arr = self.transform_funcs[ind](self, arr)

            flatarr, openarr = self._ogrid_single(ind, arr)
            self.__crds[axis.lower()] = flatarr
            self.__crds[axis.upper()] = openarr
            # now with suffix
            self.__crds[axis.lower() + sfx] = flatarr
            self.__crds[axis.upper() + sfx] = openarr

        # recalculate all cell centers, and refresh face / edges
        sfx = self._CENTER["cell"]
        for a in self.axes:
            # a = self.axis_name(a)  # validate input
            ccarr = 0.5 * (self.__crds[a][1:] + self.__crds[a][:-1])
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
            for j, d in enumerate(self.axes): #pylint: disable=W0612
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
        if isinstance(axis, str):
            if axis in self._crds:
                return axis
            else:
                raise KeyError(axis)
        else:
            return self.axes[axis]

    def _parse_slice(self, selection):
        """ parse a selection string or dict. a trailing i means the number is
        an index, else numbers are interpreted as coordinates... ex:
        selection = 'y=12.32,z-1.0' is line of data
        closest to (x, y=12.32, z=-1.0)
        selection = {'y=12i:14i,z=1i is the slice [:,12:14,1]
        """
        # parse string to dict if necessary
        if isinstance(selection, dict):
            return selection
        elif isinstance(selection, (slice, int, float)):
            return {self._axes[0]: selection}
        elif selection is None or len(selection) == 0:
            return {}
        elif isinstance(selection, str):
            sel = {}
            for i, s in enumerate(selection.replace('_', ',').split(',')):
                swhole = s
                s = s.split('=')
                if not len(s) == 2:
                    raise IndexError("There must be exactly 1 '=' per dim, "
                                     "I see '{0}'".format(swhole))
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
                        slclst[i] = int(val.rstrip('ij'))
                    else:
                        slclst[i] = float(val)
                sel[dim] = slclst
            return sel
        elif isinstance(selection, (tuple, list)):
            ret = {}
            try:
                i = selection.index(Ellipsis)
                sln = [slice(None)] * (self.nr_dims - (len(selection) - 1))
                selection = selection[:i] + sln + selection[i + 1:]
                for i in range(selection):
                    if selection[i] == Ellipsis:
                        selection[i] = slice(None)
            except ValueError:
                pass

            for sel, axis in zip(selection, self._axes):
                ret[axis] = sel
            return ret
        else:
            raise TypeError()

    def make_slice(self, selection, cc=False):
        """Turns a slice string into a slice (should be private?)

        Slices should be made using the normal ``field[selecton]``
        syntax. This function is more for internal use; however it
        does document the slice string syntax.

        In practice, selection can be a string like "y=3i:6i:2,z=0"
        where i indicates slice by index as opposed to bare numbers
        which slice by crd value. The example slice would be the 3rd
        and 5th crds in y, and the z = 0.0 plane. Selection can also
        be the usual tuple of slice objects / integers like one would
        give to numpy.

        "y=3i:6i:2,z=0" will give (if z[32] is closest to z=0.0)

            ([slice(None), slice(3, 6, 2), 32],
            [['x', ndarray(all nc x crds)], ['y', array(y[3], y[5])]],
            [['z', 0.0]],
            )

        Parameters:
            selecton (str): slice string
            cc (bool): cell centered slice

        Returns:
            tuple (slices, slcrds, reduced)
            slices: list of slice objects, one for each axis in self
            slcrds: a clist for what the coords will be after the
            slice
            reduced: a list of (axis, location) pairs of which axes
            are sliced out

        Note: cc is necessary for finding the closest plane, otherwise it
            might be off by half a grid cell
        """
        # this turns all types of selection input styles into a selection dict
        # which looks like {'axis': (start,stop?,step?), ...}
        # where start /step are optional and start / stop can be floats which
        # indicate that we need to lookup the index
        selection = self._parse_slice(selection)
        slices = [None] * self.nr_dims
        slcrds = [None] * self.nr_dims
        reduced = []

        # go through all axes and see if they are selected
        for dind, axis in enumerate(self.axes):
            if not axis in selection:
                slices[dind] = slice(None)
                slcrds[dind] = [axis, self.get_nc(axis)]
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
                        # find index of closest node
                        if cc:
                            diff = v - self.get_cc(axis)
                        else:
                            diff = v - self.get_nc(axis)
                        closest_ind = np.argmin(np.abs(diff))

                        if i == 0:
                            sel[i] = closest_ind  # always a node
                        else:  # i == 1 due to slice sel[:2]
                            sel[i] = closest_ind + 1  # + 1 to be inclusive

                # ok, if one value specified, numpy semantics say reduce that
                # dimension out
                if len(sel) == 1:
                    ind = sel[0]

                    slices[dind] = ind
                    slcrds[dind] = None
                    if cc:
                        loc = self.get_cc(axis)[ind]
                    else:
                        loc = self.get_nc(axis)[ind]
                    reduced.append([axis, loc])

                    # we set slices, slcrds, and reduced and there needs be no
                    # extra cc logic to extend the slice stop since there are
                    # no more coords in this dimension
                    continue

                elif len(sel) == 2:
                    slc = np.s_[sel[0]:sel[1]]
                elif len(sel) == 3:
                    slc = np.s_[sel[0]:sel[1]:sel[2]]
                else:
                    raise ValueError()

            # if we're doing a cc slice, there needs to be one more node
            # than data element, so add an extra node to slc.stop
            if cc and slc.stop is not None:
                if slc.stop >= 0:
                    newstop = slc.stop + 1
                else:  # slc.stop < 0
                    # this will slice crds nc, so if we're going backward, the
                    # extra crd will be included
                    newstop = slc.stop
                crd_slc = slice(slc.start, newstop, slc.step)
            else:
                crd_slc = slc
            slices[dind] = slc
            slcrds[dind] = [axis, self[axis][crd_slc].reshape(-1)]

        # remove dimensions that were sliced out
        slcrds = [crd for crd in slcrds if crd is not None]

        # print("MAKE SLICE : slices", slices, "crdlst", slcrds,
        #       "reduced", reduced)
        return slices, slcrds, reduced

    def make_slice_reduce(self, selection, cc=False):
        """make slice, and reduce dims that were not explicitly sliced"""
        slices, crdlst, reduced = self.make_slice(selection, cc=cc)
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
        return slices, crdlst, reduced

    def make_slice_keep(self, selection, cc=False):
        """make slice, but put back dims that were explicitly reduced"""
        slices, crdlst, reduced = self.make_slice(selection, cc=cc)
        # put reduced dims back, reduced will be in the same order as self.axes
        # since make_slice loops over self.axes to do the slices; this enables
        # us to call insert in the loop
        for axis, loc in reduced: #pylint: disable=W0612
            axis_ind = self.ind(axis)
            # slices[axis_ind] will be an int not a slice since it was reduced
            loc_ind = slices[axis_ind]
            crd_nc = self.get_nc(axis)

            if cc:
                if loc_ind == -1:
                    crd = crd_nc[-2:]
                    slc = slice(-1, None)
                if loc_ind == -2:
                    crd = crd_nc[-2:]
                    slc = slice(axis_ind, axis_ind + 1)
                else:
                    crd = crd_nc[loc_ind:loc_ind + 2]
                    slc = slice(axis_ind, axis_ind + 1)
            else:
                if loc_ind == -1:
                    crd = crd_nc[-1:]
                    slc = slice(-1, None)
                else:
                    crd = crd_nc[loc_ind:loc_ind + 1]
                    slc = slice(loc_ind, loc_ind + 1)

            slices[axis_ind] = slc
            crdlst.insert(axis_ind, [axis, crd])

        # should be no more reduced crds
        reduced = []
        # print("MAKE SLICE KEEP : slices", slices, "crdlst", crdlst,
        #       "reduced", reduced)
        return slices, crdlst, reduced

    def slice(self, selection, cc=False):
        """Get crds that describe a slice (subset) of this grid.
        Reduces dims the same way numpy / fields do. Chances are
        you want either slice_reduce or slice_keep
        """
        slices, crdlst, reduced = self.make_slice(selection, cc=cc)
        # pass through if nothing happened
        if slices == [slice(None)] * len(slices):
            return self
        return wrap_crds(self._TYPE, crdlst)

    def slice_reduce(self, selection, cc=False):
        """Get crds that describe a slice (subset) of this grid. Go
        through, and if the slice didn't touch a dim with only one crd,
        reduce it
        """
        slices, crdlst, reduced = self.make_slice_reduce(selection,
                                                         cc=cc)
        # pass through if nothing happened
        if slices == [slice(None)] * len(slices):
            return self
        return wrap_crds(self._TYPE, crdlst)

    def slice_keep(self, selection, cc=False):
        slices, crdlst, reduced = self.make_slice_keep(selection,
                                                       cc=cc)
        # pass through if nothing happened
        if slices == [slice(None)] * len(slices):
            return self
        return wrap_crds(self._TYPE, crdlst)

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
        if axes == None:
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

    def get_clist(self, axes=None, slc=None):
        """??

        Returns:
            a clist of the coordinates sliced if you wish

        Note:
            I recommend using ``numpy.s_`` for making the slice
        """
        if slc is None:
            slc = slice(None)
        if axes is None:
            axes = self.axes
        return [[axis, self.get_crd(axis)[slc]] for axis in axes]

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

    def points(self, center="none"):
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

    def nr_points(self, center="none"):
        """returns the number of points in a grid defined by these crds"""
        return np.prod([len(crd) for crd in self.get_crds(center=center)])

    def iter_points(self, center="none", **kwargs): #pylint: disable=W0613
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
        return self.set_crds((axis, arr))

    def __delitem__(self, item):
        raise ValueError("can not delete crd this way")

    def __contains__(self, item):
        if item[-2:] in list(self._CENTER.values()):
            item = item[:-2].lower()
        return item in self._crds


#FIXME: really, __init__ should be rewritten for uniform crds, since you
#       don't need a full clist, just an origin + d[xyz], but this hack
#       will kinda work for now
class UniformCrds(StructuredCrds):
    _TYPE = "uniform"

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
    #     return [(self._crds[self.axis_name(a) + sfx][1] -
    #              self._crds[self.axis_name(a) + sfx][0]) for a in axes]


class NonuniformCrds(StructuredCrds):
    _TYPE = "nonuniform"


class UniformCartesianCrds(UniformCrds):
    _TYPE = "uniform_cartesian"
    _axes = ["z", "y", "x"]

    def __init__(self, init_clist, **kwargs):
        super(UniformCartesianCrds, self).__init__(init_clist, **kwargs)


class NonuniformCartesianCrds(NonuniformCrds):
    _TYPE = "nonuniform_cartesian"
    _axes = ["z", "y", "x"]

    def __init__(self, init_clist, **kwargs):
        super(NonuniformCartesianCrds, self).__init__(init_clist, **kwargs)


# class UniformSphericalCrds(UniformCrds):
#     _TYPE = "uniform_spherical"
#     _axes = ["phi", "theta", "r"]

#     def __init__(self, init_clist=None, **kwargs):
#         super(UniformSphericalCrds, self).__init__(init_clist, **kwargs)


class NonuniformSphericalCrds(NonuniformCrds):
    _TYPE = "nonuniform_spherical"
    _axes = ["phi", "theta", "r"]

    _target_nc_params = None

    def __init__(self, init_clist, **kwargs):
        """This subclass is hackilly different from other crd types...
        init_clist MUST be list of the form:
        [
            ['phi', [0.0, 360.0, 181]],
            ['theta', [0.0, 180.0, 61]],
            ['r', [0.95, 1.05, 2]],
        ]
        where the list is the first, last, num given to
        numpy.linspace
        """
        self._axes = [d[0].lower() for d in init_clist]
        self._target_nc_params = [d[1] for d in init_clist]
        super(NonuniformSphericalCrds, self).__init__(None, **kwargs)

    def _fill_crds_dict(self):
        """do the math to calc node, cell, face, edge crds from
        the src crds
        """
        # SUPERHACKY! ok, so we're ignoring the cordinates in the file, and
        # just assuming they span the sphere... that way all we actually need
        # is nphi, ntheta, so when we need the crds, fill _src_crds_nc as
        # though we had them
        self._src_crds_nc = {}
        for ax, p in zip(self._axes, self._target_nc_params):
            self._src_crds_nc[ax] = np.linspace(p[0], p[1], p[2])
        return super(NonuniformSphericalCrds, self)._fill_crds_dict()


class UnstructuredCrds(Coordinates):
    _TYPE = "unstructured"

    def __init__(self, **kwargs):
        super(UnstructuredCrds, self).__init__(**kwargs)
        raise NotImplementedError()

##
## EOF
##
