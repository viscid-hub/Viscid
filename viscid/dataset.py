#!/usr/bin/env python
""" test docstring """

from __future__ import print_function
import logging
# import bisect

import numpy as np

from .bucket import Bucket
from .vutil import spill_prefix

class Dataset(object):
    """ datasets contain grids or other datasets
    (GridCollection derrives from Dataset)
    It is the programmer's responsibility to ensure objects added to a AOEUIDH
    dataset have __getitem__ and get_fields methods, this is not
    enforced
    """
    name = None
    time = None
    children = None  # Bucket or (time, grid)
    active_child = None

    topology_info = None
    geometry_info = None
    crds = None

    def __init__(self, name, time=None):
        self.name = name
        self.children = Bucket()

        self.active_child = None
        self.time = time

    def add(self, child, set_active=True):
        self.children[child.name] = child
        if set_active:
            self.active_child = child

    def unload(self):
        """ unload is meant to give children a chance to free caches, the idea
        being that an unload will free memory, but all the functionality is
        preserved, so data is accessable without an explicit reload
        """
        for child in self.children:
            child.unload()
        # TODO: does anything else need to be unloaded in here?

    # def remove_all_items(self):
    #     raise NotImplementedError()

    def activate(self, child_handle):
        """ it may not look like it, but this will recursively look
        in my active child for the handle because it uses getitem """
        self.active_child = self.children[child_handle]

    def activate_time(self, time):
        """ this is basically 'activate' except it specifically picks out
        temporal datasets, and does all children, not just the active child """
        for child in self.children:
            try:
                child.activate_time(time)
            except AttributeError:
                pass

    def n_times(self, slice_str=":"):
        for child in self.children:
            try:
                return child.n_times(slice_str)
            except AttributeError:
                pass
        raise RuntimeError("I find no temporal datasets")

    def iter_times(self, slice_str=":"):
        for child in self.children:
            try:
                return child.iter_times(slice_str)
            except AttributeError:
                pass
        raise RuntimeError("I find no temporal datasets")

    def spill(self, recursive=False, prefix=""):
        for child in self.children:
            suffix = ""
            if child is self.active_child:
                suffix = " <-- active"
            print("{0}{1}{2}".format(prefix, child, suffix))
            if recursive:
                child.spill(recursive=True, prefix=prefix + spill_prefix)

    # def get_non_dataset(self):
    #     """ recurse down datasets until active_grid is not a subclass
    #         of Dataset """
    #     if isinstance(self.activate_grid, Dataset):
    #         return self.active_grid.get_non_dataset()
    #     else:
    #         return self.active_grid

    def get_field(self, fldname, time=None):
        """ recurse down active children to get a field """
        child = self.active_child

        if child is None:
            logging.warn("Could not get appropriate child...")
            return None
        else:
            return child.get_field(fldname, time=time)

    def get_child(self, item):
        """ get a child from this Dataset,  """
        return self.children[item]

    def __getitem__(self, item):
        """ if a child exists with handle, return it, else ask
        the active child if it knows what you want """
        if item in self.children:
            return self.get_child(item)
        elif self.active_child is not None:
            return self.active_child[item]
        else:
            raise KeyError()

    def __delitem__(self, item):
        child = self.get_child(item)
        child.unload()
        self.children.remove_item(child)

    def __len__(self):
        return self.children.__len__()

    def __setitem__(self, name, child):
        # um... is this kosher??
        child.name = name
        self.add(child)

    def __contains__(self, item):
        if item in self.children:
            return True
        # FIXME: this might cause a bug somewhere someday
        if item in self.active_child:
            return True
        return False

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.unload()
        return None

    # def __iter__(self):
    #     for child in self.children:
    #         yield child

    # def __next__(self):
    #     raise NotImplementedError()


class DatasetTemporal(Dataset):
    _last_ind = 0
    # _all_times = None

    def __init__(self, name, time=None):
        super(DatasetTemporal, self).__init__(name, time=time)
        # ok, i want more control over my childen than a bucket can give
        # TODO: it's kind of a kludge to create a bucket then destroy it
        # so soon, but it's not a big deal
        self.children = []
        # self._all_times = []

    def add(self, child, set_active=True):
        if child is None:
            raise RuntimeError()
        if child.time is None:
            child.time = 0.0
            logging.warn("A child with no time? Something is strange...")
        # this keeps the children in time order
        self.children.append((child.time, child))
        self.children.sort()
        # binary in sorting... maybe more efficient?
        #bisect.insort(self.children, (child.time, child))
        if set_active:
            self.active_child = child

    def unload(self):
        """ unload is meant to give children a chance to free caches, the idea
        being that an unload will free memory, but all the functionality is
        preserved, so data is accessable without an explicit reload
        """
        for child in self.children:
            child[1].unload()
        # TODO: does anything else need to be unloaded in here?

    def activate(self, time):
        self.active_child = self.get_child(time)

    def activate_time(self, time):
        """ this is basically 'activate' except it specifically picks out
        temporal datasets """
        self.activate(time)

    def _slice_time(self, slice_str=":"):
        times = np.array([child[0] for child in self.children])
        slc_lst = [s.strip() for s in slice_str.split(":")]
        # slc_lst[:2] = [float(t) if t != "" else None for t in slc_lst[:2]]
        slc_lst[:2] = [np.argmin(np.abs(float(t) - times)) if t != "" else None
                       for t in slc_lst[:2]]
        slc_lst[2:] = [int(s) if s != "" else None for s in slc_lst[2:]]
        if len(slc_lst) == 1:
            slc_lst = [slc_lst[0], slc_lst[0] + 1]
        slc = slice(*slc_lst) #pylint: disable=W0142
        return slc

    def n_times(self, slice_str=":"):
        slc = self._slice_time(slice_str=slice_str)
        return len(self.children[slc])

    def iter_times(self, slice_str=":"):
        slc = self._slice_time(slice_str=slice_str)
        return (child[1] for child in self.children[slc])

    def spill(self, recursive=False, prefix=""):
        for child in self.children:
            suffix = ""
            if child[1] is self.active_child:
                suffix = " <-- active"
            # print("{0}{1}{2}".format(prefix, child, suffix))
            if recursive:
                child[1].spill(recursive=True, prefix=prefix + spill_prefix)

    def get_field(self, fldname, time=None):
        """ recurse down active children to get a field """
        if time is not None:
            child = self.get_child(time)
        else:
            child = self.active_child

        if child is None:
            logging.warn("Could not get appropriate child...")
            return None
        else:
            return child.get_field(fldname, time=time)

    def get_child(self, item):
        """ if item is an int and < len(children), it is an index in a list,
        else I will find the cloest time to float(item) """
        if len(self.children) <= 1:
            # it's appropriate to get an index error if len == 0
            self._last_ind = 0
            return self.children[0][1]
        elif isinstance(item, int) and item < len(self.children):
            #print('integering')
            self._last_ind = item
            return self.children[item][1]
        else:
            # NOTE: this has gone too far
            time = float(item)
            last_ind = self._last_ind
            closest_ind = -1
            # print(time, last_ind)
            if time >= self.children[last_ind][0]:
                # print("forward")
                i = last_ind + 1
                while i < len(self.children):
                    this_time = self.children[i][0]
                    # print(i, this_time)
                    if time <= this_time:
                        avg = 0.5 * (self.children[i - 1][0] + this_time)
                        if time >= avg:
                            # print(">= ", avg)
                            closest_ind = i
                        else:
                            # print("< ", avg)
                            closest_ind = i - 1
                        break
                    i += 1
                if closest_ind < 0:
                    closest_ind = len(self.children) - 1
            else:
                # print("backward")
                i = last_ind - 1
                while i >= 0:
                    this_time = self.children[i][0]
                    # print(i, this_time)
                    if time >= self.children[i][0]:
                        avg = 0.5 * (self.children[i + 1][0] + this_time)
                        if time >= avg:
                            # print(">= ", avg)
                            closest_ind = i + 1
                        else:
                            # print("< ", avg)
                            closest_ind = i
                        break
                    i -= 1
                if closest_ind < 0:
                    closest_ind = 0
            # print("closest_ind: ", closest_ind)
            self._last_ind = closest_ind
            return self.children[closest_ind][1]

    def __contains__(self, item):
        if isinstance(item, int) and item < len(self.children):
            return True
        try:
            float(item)
            return True
        except ValueError:
            return item in self.active_child

    # def __getitem__(self, item):
    #     """ Get a dataitem or list of dataitems based on time, grid, and
    #         varname. the 'active' components are given by default, but varname
    #         is manditory, else how am i supposed to know what to serve up for
    #         you. Examples:
    #         dataset[time, 'gridhandle', 'varname'] == DataItem
    #         dataset['time', 'gridhandle', 'varname'] == DataItem
    #         dataset[timeslice, 'gridhandle', 'varname'] == list of DataItems
    #         dataset[time, 'varname'] == DataItem using active grid
    #         dataset['varname'] == DataItem using active time / active grid
    #         """
    #     req_grid = None

    #     if not isinstance(item, tuple):
    #         item = (item,)

    #     varname = item[-1]
    #     ntimes = len(item) - 1 # -1 for varname
    #     try:
    #         if len(item) > 1:
    #             req_grid = self.grids[item[-2]]
    #     except KeyError:
    #         pass
    #     if not req_grid:
    #         req_grid = self.active_grid
    #         ntimes -= 1

    #     if ntimes == 0:
    #         grids = [self.grid_by_time(self.active_time)]
    #     else:
    #         grids = [self.grid_by_time(t) for t in item[:ntimes]]

    #     if len(grids) == 1:
    #         return grids[0][varname]
    #     else:
    #         return [g[varname] for g in grids]

    # def grid_by_time(self, time):
    #     """ returns grid for this specific time, time can also be a slice """
    #     if isinstance(time, slice):
    #         pass
    #     else:
    #         pass
