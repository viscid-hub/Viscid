#!/usr/bin/env python
""" test docstring """

from __future__ import print_function
import re
from itertools import chain
from operator import itemgetter

import numpy as np

import viscid
from viscid import logger
from viscid.compat import string_types
from viscid.bucket import Bucket
from viscid.grid import Grid
from viscid import tree
from viscid import vutil
from viscid.vutil import tree_prefix
from viscid.sliceutil import standardize_sel, std_sel2index, selection2values


__all__ = ['to_dataframe', 'from_dataframe']


def to_dataframe(collection, fld_names=None, selection=Ellipsis,
                 time_sel=slice(None), time_col='time', datetime_col='datetime'):
    """Consolidate field collection into pandas dataframe

    Args:
        collection (sequence): Can be one of (Field, List[Field],
            Dataset, Grid)
        fld_names (sequence, None): grab specific fields by name,
            or None to grab all fields
        selection (selection): optional spatial selection
        time (selection): optional time selection

    Returns:
        pandas.DataFrame
    """
    if not hasattr(collection, 'to_dataframe'):
        if not isinstance(collection, (list, tuple)):
            collection = [collection]

        collection_dict = {}
        for fld in collection:
            if fld.time in collection_dict:
                collection_dict[fld.time].append(fld)
            else:
                collection_dict[fld.time] = [fld]

        dset = DatasetTemporal()
        for t in sorted(list(collection_dict.keys())):
            fld_list = collection_dict[t]
            grid = Grid()
            grid.crds = fld_list[0].crds
            grid.time = t
            grid.basetime = fld_list[0].basetime
            grid.add_field(*fld_list)
            dset.add(grid)

        if len(collection_dict) == 1:
            collection = dset.get_grid()
        else:
            collection = dset

    frame = collection.to_dataframe(fld_names=fld_names, selection=selection,
                                    time_sel=time_sel, time_col=time_col,
                                    datetime_col=datetime_col)
    return frame

def from_dataframe(frame, crd_cols=None, time_col='time', datetime_col='datetime'):
    """Make either a DatasetTemporal or Grid from pandas dataframe

    Args:
        frame (pandas.DataFrame): frame to parse
        crd_cols (List[Str], None): list of column names for coordinates
        time_col (str): column name of times
        datetime_col (str): column name of datetimes

    Returns:
        DatasetTemporal or Grid

    Raises:
        ValueError: if only 1 row given and crd_cols is None
    """
    import pandas

    # discover times and possible basetime
    try:
        unique_times = frame[time_col].drop_duplicates()
        if 'datetime' in frame:
            unique_datetimes = frame[datetime_col].drop_duplicates()
            if len(unique_times) > 1:
                dt_datetime = unique_datetimes.iloc[1] - unique_datetimes.iloc[0]
                dt_time = unique_times.iloc[1] - unique_times.iloc[0]
                t0_timedelta = unique_times.iloc[0] * (dt_datetime / dt_time)
            else:
                t0_timedelta = viscid.as_timedelta64(1e6 * unique_times.iloc[0], 'us')
            basetime = unique_datetimes.iloc[0] - t0_timedelta
        else:
            basetime = None
        frame0 = frame[frame[time_col] == unique_times[0]]
    except KeyError:
        unique_times = np.array([0.0])
        basetime = None
        frame0 = frame

    # discover crd_cols if not given
    if crd_cols is None:
        frame1 = frame0.drop([time_col, datetime_col], axis=1, errors='ignore')
        if len(frame1) <= 1:
            raise ValueError("With only 1 row, crd_cols must be specified.")
        for icol in range(frame1.shape[1]):
            diff = frame1.iloc[1, icol] - frame1.iloc[0, icol]
            if diff != np.zeros((1,), dtype=diff.dtype):
                break
        crd_cols = frame1.columns[:icol + 1]

    # discover field shape and make coordinates
    crd_arrs = [frame[col].drop_duplicates() for col in crd_cols]
    shape = [len(arr) for arr in crd_arrs]
    crds = viscid.arrays2crds(crd_arrs, crd_names=crd_cols)

    fld_names = list(frame.columns)
    for _col in [time_col, datetime_col] + list(crd_cols):
        if _col in fld_names:
            fld_names.remove(_col)

    # wrap everything up into grids
    grids = []
    for time in unique_times:
        grid = Grid()
        grid.time = time
        grid.basetime = basetime
        try:
            frame1 = frame[frame[time_col] == time]
        except KeyError:
            frame1 = frame
        for name in fld_names:
            arr = frame1[name].values.reshape(shape)
            fld = viscid.wrap_field(arr, crds, name=name, center='node')
            grid.add_field(fld)
        grids.append(grid)

    if len(grids) > 1:
        ret = DatasetTemporal()
        for grid in grids:
            ret.add(grid)
        ret.basetime = basetime
    else:
        ret = grids[0]

    return ret


class DeferredChild(object):
    def __init__(self, callback, callback_args, callback_kwargs, parent=None,
                 name='NoName', time=0.0):
        self.callback = callback
        self.callback_args = callback_args if callback_args else ()
        self.callback_kwargs = callback_kwargs if callback_kwargs else {}
        self.parents = []
        if parent is not None:
            self.parents.append(parent)
        self.name = name
        self.time = time

    def resolve(self):
        ret = self.callback(*self.callback_args, **self.callback_kwargs)
        if self.parents:
            # this is a little kludgy, but at the moment, prepare_child
            # is only used to add the parent to the list of a child's parents
            self.parents[0].prepare_child(ret)
        return ret

    def clear_cache(self):
        pass

    def remove_all_items(self):
        pass

    def print_tree(self, depth=-1, prefix=""):
        print('{0}{1}'.format(prefix, self))


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
        super(Dataset, self).__init__(**kwargs)
        if self.children is None:
            self.children = Bucket(ordered=True)
        self.active_child = None
        for arg in args:
            self.add(arg)

    def add(self, child, set_active=True):
        self.prepare_child(child)
        self.children[child.name] = child
        if set_active:
            self.active_child = child

    def add_deferred(self, key, callback, callback_args=None,
                     callback_kwargs=None, set_active=True):
        child = DeferredChild(callback, callback_args=callback_args,
                              callback_kwargs=callback_kwargs,
                              parent=self, name=key)
        self.add(child, set_active=set_active)

    def _clear_cache(self):
        for child in self.children:
            child.clear_cache()

    def clear_cache(self):
        """Clear all childrens' caches"""
        self._clear_cache()

    def remove_all_items(self):
        for child in self.children:
            self.tear_down_child(child)
            child.remove_all_items()
        self.children = Bucket(ordered=True)

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

    def nr_times(self, sel=slice(None), val_endpoint=True, interior=False,
                 tdunit='s', tol=100):
        for child in self.children:
            try:
                return child.nr_times(sel=sel, val_endpoint=val_endpoint,
                                      interior=interior, tdunit=tdunit, tol=tol)
            except AttributeError:
                pass
        raise RuntimeError("I find no temporal datasets")

    def iter_times(self, sel=slice(None), val_endpoint=True, interior=False,
                   tdunit='s', tol=100, resolved=True):
        for child in self.iter_resolved_children():
            try:
                return child.iter_times(sel=sel, val_endpoint=val_endpoint,
                                        interior=interior, tdunit=tdunit, tol=tol,
                                        resolved=resolved)
            except AttributeError:
                pass
        raise RuntimeError("I find no temporal datasets")

    def tslc_range(self, sel=slice(None), tdunit='s'):
        """Find endpoints for a time slice selection

        Note:
            If the selection is slice-by-location, the values are not
            adjusted to the nearest frame. For this functionality,
            you will want to use :py:func:`get_times` and pull out the
            first and last values.
        """
        for child in self.children:
            try:
                return child.tslc_range(sel=sel, tdunit=tdunit)
            except AttributeError:
                pass
        raise RuntimeError("I find no temporal datasets")

    def get_times(self, sel=slice(None), val_endpoint=True, interior=False,
                  tdunit='s', tol=100):
        return list(self.iter_times(sel=sel, val_endpoint=val_endpoint,
                                    interior=interior, tdunit=tdunit, tol=tol,
                                    resolved=False))

    def get_time(self, sel=slice(None), val_endpoint=True, interior=False,
                 tdunit='s', tol=100):
        try:
            return next(self.iter_times(sel=sel, val_endpoint=val_endpoint,
                                        interior=interior, tdunit=tdunit, tol=tol))
        except StopIteration:
            raise RuntimeError("Dataset has no time slices")

    def to_dataframe(self, fld_names=None, selection=Ellipsis,
                     time_sel=slice(None), time_col='time',
                     datetime_col='datetime'):
        """Consolidate grid's field data into pandas dataframe

        Args:
            fld_names (sequence, None): grab specific fields by name,
                or None to grab all fields
            selection (selection): optional spatial selection
            time (selection): optional time selection

        Returns:
            pandas.DataFrame
        """
        # deferred import so that viscid does not depend on pandas
        import pandas
        frames = [child.to_dataframe(fld_names=fld_names, selection=selection,
                                     time_sel=time_sel, time_col=time_col,
                                    datetime_col=datetime_col)
                  for child in self.children]
        frame = pandas.concat(frames, ignore_index=True, sort=False)
        # make sure crds are all at the beginning, since concat can reorder them
        col0 = list(frames[0].columns)
        frame = frame[col0 + list(set(frame.columns) - set(col0))]
        return frame

    def iter_fields(self, time=None, fld_names=None):
        """ generator for fields in the active dataset,
        this will recurse down to a grid """
        child = self.active_child

        if child is None:
            logger.error("Could not get appropriate child...")
            return None
        else:
            return child.iter_fields(time=time, fld_names=fld_names)

    def iter_field_items(self, time=None, fld_names=None):
        """ generator for (name, field) in the active dataset,
        this will recurse down to a grid """
        child = self.active_child

        if child is None:
            logger.error("Could not get appropriate child...")
            return None
        else:
            return child.iter_field_items(time=time, fld_names=fld_names)

    def field_dict(self, time=None, fld_names=None, **kwargs):
        """ fields as dict of {name: field} """
        child = self.active_child

        if child is None:
            logger.error("Could not get appropriate child...")
            return None
        else:
            return child.field_dict(time=time, fld_names=fld_names)

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

    def get_field(self, fldname, time=None, slc=Ellipsis):
        """ recurse down active children to get a field """
        child = self.active_child

        if child is None:
            logger.error("Could not get appropriate child...")
            return None
        else:
            return child.get_field(fldname, time=time, slc=slc)

    def get_grid(self, time=None):
        """ recurse down active children to get a field """
        child = self.active_child.resolve()

        if child is None:
            logger.error("Could not get appropriate child...")
            return None
        else:
            return child.get_grid(time=time)

    def get_child(self, item):
        """ get a child from this Dataset,  """
        return self.children[item].resolve()

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
        # FIXME, is it possable to de-resolve item to a DeferredChild?
        child = self.get_child(item)
        child.clear_cache()
        self.children.remove_item(child)

    def __len__(self):
        return self.children.__len__()

    def __setitem__(self, name, child):
        # um... is this kosher??
        child.name = name
        self.add(child)

    def __contains__(self, item):
        # FIXME, is it possable to de-resolve item to a DeferredChild?
        if item in self.children:
            return True
        # FIXME: this might cause a bug somewhere someday
        if item in self.active_child:
            return True
        return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        self.clear_cache()
        return None

    def __iter__(self):
        return self.iter_resolved_children()

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
        # ok, i want more control over my childen than a bucket can give
        # FIXME: this is kind of not cool since children is a public
        # attribute yet it's a different type here
        self.children = []
        super(DatasetTemporal, self).__init__(*args, **kwargs)

    def add(self, child, set_active=True):
        if child is None:
            raise RuntimeError()
        if child.time is None:
            child.time = 0.0
            logger.error("A child with no time? Something is strange...")
        # this keeps the children in time order
        self.prepare_child(child)
        self.children.append((child.time, child))
        self.children.sort(key=itemgetter(0))
        # binary in sorting... maybe more efficient?
        # bisect.insort(self.children, (child.time, child))
        if set_active:
            self.active_child = child

    def add_deferred(self, time, callback, callback_args=None,
                     callback_kwargs=None, set_active=True):
        child = DeferredChild(callback, callback_args=callback_args,
                              callback_kwargs=callback_kwargs,
                              parent=self, time=time)
        self.add(child, set_active=set_active)

    def remove_all_items(self):
        for child in self.children:
            self.tear_down_child(child[1])
            child[1].remove_all_items()
        self.children = []

    def clear_cache(self):
        """Clear all childrens' caches"""
        for child in self.children:
            child[1].clear_cache()

    def activate(self, time):
        self.active_child = self.get_child(time)

    def activate_time(self, time):
        """ this is basically 'activate' except it specifically picks out
        temporal datasets """
        self.activate(time)

    #############################################################################
    ## here begins a slew of functions that make specifying a time / time slice
    ## super general
    def _slice_time(self, sel=slice(None), val_endpoint=True, interior=False,
                    tdunit='s', tol=100):
        """
        Args:
            slc (str, slice, list): can be a single string containing
                slices, slices, ints, floats, datetime objects, or a
                list of any of the above.

        Returns:
            list of slices (containing integers only) or ints
        """
        times = np.array([child[0] for child in self.children])

        try:
            basetime = self.basetime
        except viscid.NoBasetimeError:
            basetime = None

        std_sel = standardize_sel(sel)
        idx_sel = std_sel2index(std_sel, times, val_endpoint=val_endpoint,
                                interior=interior, tdunit=tdunit, epoch=basetime)
        return idx_sel

    def _time_slice_to_iterator(self, slc):
        """
        Args:
            slc: a slice (containing ints only) or an int, or a list
                of any of the above

        Returns:
            a flat iterator of self.children of all the slices chained
        """
        inds = np.arange(len(self.children))[slc]
        if not isinstance(inds, np.ndarray):
            inds = np.asarray(inds).reshape(-1)
        return (self.children[i] for i in inds)

    def nr_times(self, sel=slice(None), val_endpoint=True, interior=False,
                 tdunit='s', tol=100):
        slc = self._slice_time(sel=sel, val_endpoint=val_endpoint,
                               interior=interior, tdunit=tdunit, tol=tol)
        child_iterator = self._time_slice_to_iterator(slc)
        return len(list(child_iterator))

    def iter_times(self, sel=slice(None), val_endpoint=True, interior=False,
                   tdunit='s', tol=100, resolved=True):
        slc = self._slice_time(sel=sel, val_endpoint=val_endpoint,
                               interior=interior, tdunit=tdunit, tol=tol)
        child_iterator = self._time_slice_to_iterator(slc)

        for child in child_iterator:
            # FIXME: this isn't general, but so far the only files we're
            # read have only contained one Grid / AMRGrid. Without get_grid()
            # here, the context manager will unload the file when done, but
            # that's not what we wanted here, we wanted to just clear caches
            if resolved:
                what = child[1].resolve().get_grid()
            else:
                what = child[1]

            with what as target:
                yield target

    def get_times(self, sel=slice(None), val_endpoint=True, interior=False,
                  tdunit='s', tol=100):
        return list(self.iter_times(sel=sel, val_endpoint=val_endpoint,
                                    interior=interior, tdunit=tdunit, tol=tol,
                                    resolved=False))

    def get_time(self, sel=slice(None), val_endpoint=True, interior=False,
                 tdunit='s', tol=100):
        return self.get_times(sel=sel, val_endpoint=val_endpoint,
                              interior=interior, tdunit=tdunit, tol=tol)[0]

    def tslc_range(self, sel=slice(None), tdunit='s'):
        """Find endpoints for a time slice selection

        Note:
            If the selection is slice-by-location, the values are not
            adjusted to the nearest frame. For this functionality,
            you will want to use :py:func:`get_times` and pull out the
            first and last values.
        """
        times = np.array([child[0] for child in self.children])

        try:
            basetime = self.basetime
        except viscid.NoBasetimeError:
            basetime = None

        return selection2values(times, sel, epoch=basetime, tdunit=tdunit)

    ## ok, that's enough for the time stuff
    ########################################

    def to_dataframe(self, fld_names=None, selection=Ellipsis,
                     time_sel=slice(None), time_col='time',
                     datetime_col='datetime'):
        """Consolidate grid's field data into pandas dataframe

        Args:
            fld_names (sequence, None): grab specific fields by name,
                or None to grab all fields
            selection (selection): optional spatial selection
            time (selection): optional time selection

        Returns:
            pandas.DataFrame
        """
        # deferred import so that viscid does not depend on pandas
        import pandas
        frames = [child.to_dataframe(fld_names=fld_names, selection=selection,
                                     time_sel=time_sel, time_col=time_col,
                                     datetime_col=datetime_col)
                  for child in self.iter_times(sel=time_sel)]
        frame = pandas.concat(frames, ignore_index=True, sort=False)
        # make sure crds are all at the beginning, since concat can reorder them
        col0 = list(frames[0].columns)
        frame = frame[col0 + list(set(frame.columns) - set(col0))]
        return frame

    def iter_fields(self, time=None, fld_names=None):
        """ generator for fields in the active dataset,
        this will recurse down to a grid """
        if time is not None:
            child = self.get_child(time)
        else:
            child = self.active_child

        if child is None:
            logger.error("Could not get appropriate child...")
            return None
        else:
            return child.iter_fields(time=time, fld_names=fld_names)

    def iter_field_items(self, time=None, fld_names=None):
        """ generator for (name, field) in the active dataset,
        this will recurse down to a grid """
        if time is not None:
            child = self.get_child(time)
        else:
            child = self.active_child

        if child is None:
            logger.error("Could not get appropriate child...")
            return None
        else:
            return child.iter_field_items(time=time, fld_names=fld_names)

    def field_dict(self, time=None, fld_names=None):
        """ fields as dict of {name: field} """
        if time is not None:
            child = self.get_child(time)
        else:
            child = self.active_child

        if child is None:
            logger.error("Could not get appropriate child...")
            return None
        else:
            return child.field_dict(fld_names=fld_names)

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

    def get_field(self, fldname, time=None, slc=Ellipsis):
        """ recurse down active children to get a field """
        if time is not None:
            child = self.get_child(time)
        else:
            child = self.active_child

        if child is None:
            logger.error("Could not get appropriate child...")
            return None
        else:
            return child.get_field(fldname, time=time, slc=slc)

    def get_grid(self, time=None):
        """ recurse down active children to get a field """
        if time is not None:
            child = self.get_child(time)
        else:
            child = self.active_child.resolve()

        if child is None:
            logger.error("Could not get appropriate child...")
            return None
        else:
            return child.get_grid(time=time)

    def get_child(self, item):
        """ if item is an int and < len(children), it is an index in a list,
        else I will find the cloest time to float(item) """
        # print(">> get_child:", item)
        # print(">> slice is:", self._slice_time(item))
        # always just return the first slice's child... is this wrong?
        child = self.children[self._slice_time(item)][1].resolve()
        return child

    def __contains__(self, item):
        if isinstance(item, int) and item > 0 and item < len(self.children):
            return True
        if isinstance(item, string_types) and item[-1] == 'f':
            try:
                val = float(item[:-1])
                if val >= self.children[0][0] and val <= self.children[-1][0]:
                    return True
                else:
                    return False
            except ValueError:
                pass
        return item in self.active_child

    def iter_resolved_children(self):
        return (child[1].resolve() for child in self.children)
