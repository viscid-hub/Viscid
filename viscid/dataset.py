#!/usr/bin/env python
""" test docstring """

from __future__ import print_function
import re
from string import ascii_letters
from datetime import datetime
from itertools import islice, chain

import numpy as np

from viscid import logger
from viscid.compat import string_types
from viscid.bucket import Bucket
from viscid import tree
from viscid import vutil
from viscid.vutil import tree_prefix, convert_floating_slice

class Dataset(tree.Node):
    """Datasets contain grids or other datasets

    Note:
        Datasets should probably be created using a vfile's
        `_make_dataset` to make sure the info dict is propogated
        appropriately

    It is the programmer's responsibility to ensure objects added to a AOEUIDH
    dataset have __getitem__ and get_fields methods, this is not
    enforced
    """
    children = None  # Bucket or (time, grid)
    active_child = None

    topology_info = None
    geometry_info = None
    crds = None


    def __init__(self, *args, **kwargs):
        """info is for information that is shared for a whole
        tree, from vfile all the way down to fields
        """
        super(Dataset, self).__init__(*args, **kwargs)
        self.children = Bucket(ordered=True)
        self.active_child = None

    def add(self, child, set_active=True):
        self.prepare_child(child)
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

    def nr_times(self, slice_str=":"):
        for child in self.children:
            try:
                return child.nr_times(slice_str)
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

    def get_times(self, slice_str=":"):
        return list(self.iter_times(slice_str=slice_str))

    def get_time(self, slice_str=":"):
        try:
            return next(self.iter_times(slice_str))
        except StopIteration:
            raise RuntimeError("Dataset has no time slices")

    def iter_fields(self, time=None, named=None):
        """ generator for fields in the active dataset,
        this will recurse down to a grid """
        child = self.active_child

        if child is None:
            logger.warn("Could not get appropriate child...")
            return None
        else:
            return child.iter_fields(time=time, named=named)

    def print_tree(self, depth=-1, prefix=""):
        if prefix == "":
            print(self)
            prefix += tree_prefix

        for child in self.children:
            suffix = ""
            if child is self.active_child:
                suffix = " <-- active"
            print("{0}{1}{2}".format(prefix, child, suffix))
            if depth != 0:
                child.print_tree(depth=depth - 1, prefix=prefix + tree_prefix)

    # def get_non_dataset(self):
    #     """ recurse down datasets until active_grid is not a subclass
    #         of Dataset """
    #     if isinstance(self.activate_grid, Dataset):
    #         return self.active_grid.get_non_dataset()
    #     else:
    #         return self.active_grid

    def get_field(self, fldname, time=None, slc=None):
        """ recurse down active children to get a field """
        child = self.active_child

        if child is None:
            logger.warn("Could not get appropriate child...")
            return None
        else:
            return child.get_field(fldname, time=time, slc=slc)

    def get_grid(self, time=None):
        """ recurse down active children to get a field """
        child = self.active_child

        if child is None:
            logger.warn("Could not get appropriate child...")
            return None
        else:
            return child.get_grid(time=time)

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

    def __exit__(self, exc_type, value, traceback):
        self.unload()
        return None

    def __iter__(self):
        return self.children.__iter__()

    # def __next__(self):
    #     raise NotImplementedError()


class DatasetTemporal(Dataset):
    """
    Note:
        Datasets should probably be created using a vfile's
        `_make_dataset` to make sure the info dict is propogated
        appropriately
    """
    _last_ind = 0
    # _all_times = None

    def __init__(self, *args, **kwargs):
        super(DatasetTemporal, self).__init__(*args, **kwargs)
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
            logger.warn("A child with no time? Something is strange...")
        # this keeps the children in time order
        self.prepare_child(child)
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

    #############################################################################
    ## here begins a slew of functions that make specifying a time / time slice
    ## super general
    @staticmethod
    def _parse_time_slice_str(slc_str):
        r"""
        Args:
            slc_str (str): must be a single string containing a single
                time slice

        Returns:
            one of {int, float, string, or slice (can contain ints,
            floats, or strings)}

        Note:
            Individual elements of the slice can look like an int,
            float, or they can have the form [A-Z]+[\d:]+\.\d*. This
            last one is a datetime-like representation with some
            preceding letters. The preceding letters are
        """
        # regex parse the sting into a list of datetime-like strings,
        # integers, floats, and bare colons that mark the slices
        # Note: for datetime-like strings, the letters preceeding a datetime
        # are necessary, otherwise 02:20:30.01 would have more than one meaning
        rstr = r"\s*(?:(?!:)[A-Z]+[-\d:]+\.\d*|:|(?=.)-?\d*(?:\.\d*)?)\s*"
        r = re.compile(rstr, re.I)

        all_times = r.findall(slc_str)
        if len(all_times) == 1 and all_times[0] != ":":
            return vutil.str_to_value(all_times[0])

        # fill in implied slice colons, then replace them with something
        # unique... like !!
        all_times += [':'] * (2 - all_times.count(':'))
        all_times = [s if s != ":" else "!!" for s in all_times]
        # this is kinda silly, but turn all times back into a string,
        # then split it again, this is the easiest way to parse something
        # like '1::2'
        ret = "".join(all_times).split("!!")
        # convert empty -> None, ints -> ints and floats->floats
        for i, val in enumerate(ret):
            ret[i] = vutil.str_to_value(val)
        if len(ret) > 3:
            raise ValueError("Could not decipher slice: {0}. Perhaps you're "
                             "missing some letters in front of a time "
                             "string?".format(slc_str))
        return slice(*ret)

    def _slice_time(self, slc=":"):
        """
        Args:
            slc (str, slice, list): can be a single string containing
                slices, slices, ints, floats, datetime objects, or a
                list of any of the above.

        Returns:
            list of slices (containing integers only) or ints
        """
        # print("SLC::", slc)
        if not isinstance(slc, (list, tuple)):
            slc = [slc]

        # expand strings that are comma separated lists of strings
        _slc = []
        for s in slc:
            if isinstance(s, string_types):
                for _ in s.split(','):
                    _slc.append(_)
            else:
                _slc.append(s)
        slc = _slc

        ret = []
        times = np.array([child[0] for child in self.children])
        for s in slc:
            if isinstance(s, string_types):
                s = self._parse_time_slice_str(s)

            if isinstance(s, slice):
                slc_lst = [s.start, s.stop, s.step]
            else:
                slc_lst = [s]

            # do translation from string/datetime/etc -> floats
            for i in range(min(len(slc_lst), 2)):
                translation = self._translate_time(slc_lst[i])
                if translation != NotImplemented:
                    slc_lst[i] = translation

            # convert floats in the slice to integers
            if len(slc_lst) == 1:
                if isinstance(slc_lst[0], (float, np.floating)):
                    slc_lst = [np.argmin(np.abs(times - slc_lst[0]))]
            else:
                slc_lst = convert_floating_slice(times, *slc_lst)

            inttypes = (int, np.integer)
            isint = [isinstance(v, inttypes) for v in slc_lst if v is not None]
            if not all(isint):
                raise ValueError("Could not decipher time slice: "
                                 "{0}".format(s))

            if len(slc_lst) == 1:
                ret += slc_lst
            else:
                # make the slice inclusive, no matter what
                slc_lst = [None if _s is None else int(_s) for _s in slc_lst]
                ret.append(slice(*slc_lst))
        return ret

    def _time_slice_to_iterator(self, slc):
        """
        Args:
            slc: a slice (containing ints only) or an int, or a list
                of any of the above

        Returns:
            a flat iterator of self.children of all the slices chained
        """
        if not isinstance(slc, (list, tuple)):
            slc = [slc]

        child_iter_lst = []
        for s in slc:
            if isinstance(s, slice):
                inds = range(len(self.children))[s]
                it = (self.children[i] for i in inds)
                child_iter_lst.append(it)
            else:
                child_iter_lst.append([self.children[s]])
        return chain(*child_iter_lst)

    def nr_times(self, slice_str=":"):
        slc = self._slice_time(slice_str)
        child_iterator = self._time_slice_to_iterator(slc)
        return len(list(child_iterator))

    def iter_times(self, slice_str=":"):
        slc = self._slice_time(slice_str)
        child_iterator = self._time_slice_to_iterator(slc)

        for child in child_iterator:
            with child[1] as target:
                yield target

    def get_times(self, slice_str=":"):
        return list(self.iter_times(slice_str=slice_str))

    def get_time(self, slice_str=":"):
        return self.get_times(slice_str)[0]

    ## ok, that's enough for the time stuff
    ########################################

    def iter_fields(self, time=None, named=None):
        """ generator for fields in the active dataset,
        this will recurse down to a grid """
        if time is not None:
            child = self.get_child(time)
        else:
            child = self.active_child

        if child is None:
            logger.warn("Could not get appropriate child...")
            return None
        else:
            return child.iter_fields(time=time, named=named)

    def print_tree(self, depth=-1, prefix=""):
        if prefix == "":
            print(self)
            prefix += tree_prefix

        for child in self.children:
            suffix = ""
            if child[1] is self.active_child:
                suffix = " <-- active"
            print("{0}{1} (t={2}){3}".format(prefix, child, child[0], suffix))
            if depth != 0:
                child[1].print_tree(depth=depth - 1, prefix=prefix + tree_prefix)

    def get_field(self, fldname, time=None, slc=None):
        """ recurse down active children to get a field """
        if time is not None:
            child = self.get_child(time)
        else:
            child = self.active_child

        if child is None:
            logger.warn("Could not get appropriate child...")
            return None
        else:
            return child.get_field(fldname, time=time, slc=None)

    def get_grid(self, time=None):
        """ recurse down active children to get a field """
        if time is not None:
            child = self.get_child(time)
        else:
            child = self.active_child

        if child is None:
            logger.warn("Could not get appropriate child...")
            return None
        else:
            return child.get_grid(time=time)

    def get_child(self, item):
        """ if item is an int and < len(children), it is an index in a list,
        else I will find the cloest time to float(item) """
        # print(">> get_child:", item)
        # print(">> slice is:", self._slice_time(item))
        # always just return the first slice's child... is this wrong?
        child = self.children[self._slice_time(item)[0]][1]
        return child

    def __contains__(self, item):
        if isinstance(item, int) and item < len(self.children):
            return True
        try:
            float(item)
            return True
        except ValueError:
            return item in self.active_child

    def __iter__(self):
        for child in self.children:
            yield child[1]

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
    #     nr_times = len(item) - 1 # -1 for varname
    #     try:
    #         if len(item) > 1:
    #             req_grid = self.grids[item[-2]]
    #     except KeyError:
    #         pass
    #     if not req_grid:
    #         req_grid = self.active_grid
    #         nr_times -= 1

    #     if nr_times == 0:
    #         grids = [self.grid_by_time(self.active_time)]
    #     else:
    #         grids = [self.grid_by_time(t) for t in item[:nr_times]]

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
