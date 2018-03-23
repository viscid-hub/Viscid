#!/usr/bin/env python
"""Convenience functions for slicing by value"""

from __future__ import print_function
from itertools import count
import re

import numpy as np

from viscid import logger
from viscid.compat import izip, string_types
from viscid.npdatetime import (is_datetime_like, is_timedelta_like,
                               as_datetime64, as_timedelta64, time_diff)


__all__ = ["str2value", "parse_time_slice_str", "make_fwd_slice",
           "to_slices", "to_slice", "make_slice_inclusive",
           "selections2values", "selection2values", "slice2values"]


def str2value(s):
    """try to parse things like none, true, false, ints, floats, etc."""
    ret = s
    s_clean = s.strip().lower()

    if len(s_clean) == 0 or s_clean == "none":
        ret = None
    elif s_clean == "true":
        ret = True
    elif s_clean == "false":
        ret = True
    elif s_clean == "True":
        ret = True
    else:
        try:
            ret = int(s_clean)
        except ValueError:
            try:
                ret = float(s_clean)
            except ValueError:
                pass
    return ret

def _warn_deprecated_float(val, varname='value'):
    s = ("DEPRECATION...\n"
         "Slicing by float is deprecated. The slice by value syntax is \n"
         "now a string that has a trailing 'f', as in 'x=0f' [{0} = {1}]"
         "".format(varname, val))
    logger.warning(s)

def _standardize_slcval(val, epoch=None, tdunit='s'):
    """Standardize things that can appear in a slice

    Returns:
        One of the following,
         - None
         - np.newaxis
         - int
         - '{flt}f'.format(flt=val)

        Datetime-like and timedelta-like values are converted
        to floats using epoch and tdunit.

        Deprecation warnings arise when trying to convert bare floats
        or floats in strings that don't end if 'f'.
    """
    if is_timedelta_like(val, conservative=True):
        ret = "{0}f".format(as_timedelta64(val) / np.timedelta64(1, tdunit))
    elif is_datetime_like(val, conservative=True):
        if epoch is None:
            epoch = np.datetime64(0, 'us')
        tflt = time_diff(val, epoch, most_precise=True) / np.timedelta64(1, tdunit)
        ret = "{0}f".format(tflt)
    elif isinstance(val, (int, np.integer)):
        ret = val
    elif val in [None, "None", "none"]:
        ret = None
    elif val in [np.newaxis, "newaxis"]:
        # note, np.newaxis is None, so this probably dosen't happen, but
        # in case it changes in the future...
        ret = np.newaxis
    elif isinstance(val, (float, np.floating)):
        _warn_deprecated_float(val)
        ret = "{0}f".format(val)
    elif isinstance(val, string_types):
        if val[-1] == 'f':
            # gymnastics to validate the contents
            ret = "{0}f".format(float(val[:-1]))
        else:
            try:
                ret = int(val)
            except ValueError:
                try:
                    ret = '{0}f'.format(float(val))
                    _warn_deprecated_float(val)
                except ValueError:
                    raise
    else:
        raise TypeError("I'm not sure what to do with {0} (type = {1})"
                        "".format(val, type(val)))
    return ret

def _arr2float(arr, epoch=None, tdunit='s', dtype='f8'):
    # If arr is a datetime64, then turn it into a float with
    # units of tdunit since the given epoch. if no epoch is given
    # then epoch = arr[0]. From this point forward, arr is treated
    # like a float array, and slices should be converted to floats
    # using the same epoch and tdunit.

    # arr = np.asarray(arr)
    if is_timedelta_like(arr, conservative=True):
        arr = as_timedelta64(arr) / np.timedelta64(1, tdunit)
    elif is_datetime_like(arr, conservative=True):
        arr = as_datetime64(arr)
        if epoch is None:
            epoch = arr[0]
        arr = time_diff(arr, epoch, most_precise=True) / np.timedelta64(1, tdunit)
    else:
        arr = np.asarray(arr)
    return arr.astype(dtype), epoch

def parse_time_slice_str(slc_str):
    r"""
    Args:
        slc_str (str): must be a single string containing a single
            time slice

    Returns:
        one of {int, string, or slice (can contain ints,
        floats, or strings)}

    Note:
        Individual elements of the slice can look like an int,
        float with trailing 'f', or they can have the form
        [A-Z]+[\d:]+\.\d*. This last one is a datetime-like
        representation with some preceding letters. The preceding
        letters are
    """
    # regex parse the sting into a list of datetime-like strings,
    # integers, floats, and bare colons that mark the slices
    # Note: for datetime-like strings, the letters preceeding a datetime
    # are necessary, otherwise 02:20:30.01 would have more than one meaning
    rstr = (r"\s*(?:(?!:)[A-Z]+[-\d:T]+\.\d*|:|[-+]?[0-9]*\.?[0-9]+f?)\s*|"
            r"[-+]?[0-9]+")
    r = re.compile(rstr, re.I)

    all_times = r.findall(slc_str)
    if len(all_times) == 1 and all_times[0] != ":":
        return str2value(all_times[0])

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
        ret[i] = str2value(val)
    if len(ret) > 3:
        raise ValueError("Could not decipher slice: '{0}'. Perhaps you're "
                         "missing some letters in front of a time "
                         "string?".format(slc_str))
    # trim trailing dots
    ret = [r.rstrip('.') if hasattr(r, 'rstrip') else r for r in ret]
    return slice(*ret)

def make_fwd_slice(shape, slices, reverse=None, cull_second=True):
    """Make sure slices go forward

    This function returns two slices equivalent to `slices` such
    that the first slice always goes forward. This is necessary because
    h5py can't deal with reverse slices such as [::-1].

    The optional `reverse` can be used to interpret a dimension as
    flipped. This is used if the indices in a slice are based on a
    coordinate array that has already been flipped. For instance, the
    result is equivalent to `arr[::-1][slices]`, but in a way that can
    be  handled by h5py. This lets us efficiently load small subsets
    of large arrays on disk, which is most useful when the large array
    is coming through sshfs.

    Note:
        The only restriction on slices is that neither start nor stop
        can be outide the range [-L, L].

    Args:
        shape: shape of the array that is to be sliced
        slices: a tuple of slices to work with
        reverse (optional): list of bools that indicate if the
            corresponding value in slices should be ineterpreted as
            flipped
        cull_second (bool, optional): iff True, remove elements of
            the second slice for dimensions that don't exist after
            the first slice has completed. This is only here for
            a super-hacky case when slicing fields.
    Returns:
        (first_slice, second_slice)

        * first_slice: a forward-only slice that retrieves the
          desired elements of an array
        * second_slice: a slice that does [::1] or [::-1] as needed
          to make the result equivalent to slices. If keep_all,
          then this may contain None indicating that this
          dimension no longer exists after the first slice.

    Examples:
        >> a = np.arange(8)
        >> first, second = make_fwd_slice(len(a),slice(None, None, -1))
        >> (a[::-1] == a[first][second]).all()
        True

        >> a = np.arange(4*5*6).reshape((4, 5, 6))
        >> first, second = make_fwd_slice(a.shape,
        >>                                [slice(None, -1, 1),
        >>                                 slice(-1, None, 1),
        >>                                 slice(-4, -1, 2)],
        >>                                [True, True, True])
        >> a1 = a[::-1, ::-1, ::-1][:-1, -1:, -4:-1:2]
        >> a2 = a[first][second]
        >> a1 == a2
        True
    """
    if reverse is None:
        reverse = []
    if not isinstance(shape, (list, tuple, np.ndarray)):
        shape = [shape]
    if not isinstance(slices, (list, tuple)):
        slices = [slices]
    if not isinstance(reverse, (list, tuple)):
        reverse = [reverse]

    newax_inds = [i for i, x in enumerate(slices) if x == np.newaxis]
    shape = list(shape)
    for i in newax_inds:
        shape.insert(i, 1)

    # ya know, lets just go through all the dimensions in shape
    # just to be safe and default to an empty slice / no reverse
    slices = slices + [slice(None)] * (len(shape) - len(slices))
    reverse = reverse + [False] * (len(slices) - len(reverse))

    first_slc = [slice(None)] * len(slices)
    second_slc = [slice(None, None, 1)] * len(first_slc)

    for i, slc, L, rev in izip(count(), slices, shape, reverse):
        if isinstance(slc, slice):
            step = slc.step if slc.step is not None else 1
            start = slc.start if slc.start is not None else 0
            stop = slc.stop if slc.stop is not None else L
            if start < 0:
                start += L
            if stop < 0:
                stop += L

            # sanity check the start/stop since we're gunna be playing
            # fast and loose with them
            if start < 0 or stop < 0:
                raise IndexError("((start = {0}) or (stop = {1})) < 0"
                                 "".format(start, stop))
            if start > L or stop > L:
                raise IndexError("((start={0}) or (stop={1})) > (L={2})"
                                 "".format(start, stop, L))

            # now do the math of flipping the slice if needed, these branches
            # change start, stop, and step so they can be used to create a new
            # slice below
            if rev:
                if step < 0:
                    step = -step
                    if slc.start is None:
                        start = L - 1
                    if slc.stop is None:
                        start = L - 1 - start
                        stop = None
                    else:
                        start, stop = L - 1 - start, L - 1 - stop
                else:
                    start, stop = L - stop, L - start
                    start += ((stop - 1 - start) % step)
                    second_slc[i] = slice(None, None, -1)
            elif step < 0:
                step = -step
                if slc.start is None:
                    start = L - 1

                if slc.stop is None:
                    start, stop = 0, start + 1
                    start = ((stop - 1 - start) % step)
                else:
                    start, stop = stop + 1, start + 1
                    start += ((stop - 1 - start) % step)

                second_slc[i] = slice(None, None, -1)

            # check that our slice is valid
            assert start is None or (start >= 0 and start <= L), \
                "start (={0}) is outside range".format(start)
            assert start is None or (start >= 0 and start <= L), \
                "start (={0}) is outside range".format(start)
            assert start is None or stop is None or start == stop == 0 or \
                start < stop, "bad slice ordering: {0} !< {1}".format(start, stop)
            assert step > 0
            slc = slice(start, stop, step)

        elif isinstance(slc, (int, np.integer)):
            second_slc[i] = None
            if rev:
                slc = (L - 1) - slc

        elif slc == np.newaxis:
            second_slc[i] = "NEWAXIS"

        first_slc[i] = slc

    first_slc = [s for s in first_slc if s is not np.newaxis]
    if cull_second:
        second_slc = [s for s in second_slc if s is not None]
    second_slc = [np.newaxis if s == "NEWAXIS" else s for s in second_slc]
    return first_slc, second_slc

def _resolve_float(value, epoch=None, tdunit='s'):
    value = _standardize_slcval(value, epoch=epoch, tdunit=tdunit)

    try:
        value = value.rstrip()
        if len(value) == 0:
            raise ValueError("Can't slice with nothing")
        elif value[-1] == 'f':
            ret = float(value[:-1])
        else:
            ret = int(value)  # pylint: disable=redefined-variable-type

    except AttributeError:
        ret = value.__index__()

    return ret

def _closest_index(arr, value, epoch=None, tdunit='s'):
    value = _standardize_slcval(value, epoch=epoch, tdunit=tdunit)

    try:
        value = value.rstrip()
        if len(value) == 0:
            raise ValueError("Can't slice with nothing")
        elif value[-1] == 'f':
            arr, epoch = _arr2float(arr, epoch=epoch, tdunit=tdunit)
            index = int(np.argmin(np.abs(arr - float(value[:-1]))))
        else:
            index = int(value)

    except AttributeError:
        index = value.__index__()

    return index

def extract_index(arr, start=None, stop=None, step=None, val_endpoint=True,
                  interior=False, epoch=None, tdunit='s', tol=100):
    """Get integer indices for slice parts

    If start, stop, or step are strings, they are either cast to
    integers or used for a float lookup if they have a trailing 'f'.

    An example float lookup is::
        >>> [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]['0.2f:0.6f:2']
        [0.2, 0.4, 0.6]

    Normally (val_endpoint=True, interior=False), the rules for float
    lookup val_endpoints are,
        - The slice will never include an element whose value in arr
          is < start (or > if the slice is backward)
        - The slice will never include an element whose value in arr
          is > stop (or < if the slice is backward)
        - !! The slice WILL INCLUDE stop if you don't change
          val_endpoint. This is different from normal slicing, but
          it's more natural when specifying a slice as a float.
          To this end, an epsilon tolerance can be given to
          determine what's close enough.
        - TODO: implement floating point steps, this is tricky
          since arr need not be uniformly spaced, so step is
          ambiguous in this case

    If interior=True, then the slice is expanded such that start and
    stop are interior to the sliced array.

    Args:
        arr (ndarray): filled with floats to do the lookup
        start (None, int, str): like slice().start
        stop (None, int, str): like slice().stop
        step (None, int): like slice().step
        val_endpoint (bool): iff True then include stop in the slice when
            slicing-by-value (DOES NOT EFFECT SLICE-BY-INDEX).
            Set to False to get python slicing symantics when it
            comes to excluding stop, but fair warning, python
            symantics feel awkward here. Consider the case
            [0.1, 0.2, 0.3][:0.25]. If you think this should include
            0.2, then leave keep val_endpoint=True.
        interior (bool): if True, then extend both ends of the slice
            such that slice-by-value endpoints are interior to the
            slice
        epoch (datetime64-like): Epoch for to go datetime64 <-> float
        tdunit (str): Presumed time unit for floats
        tol (int): number of machine epsilons to consider
            "close enough"

    Returns:
        start, stop, step after floating point vals have been
        converted to integers
    """
    arr, epoch = _arr2float(arr, epoch=epoch, tdunit=tdunit)

    start = _standardize_slcval(start, epoch=epoch, tdunit=tdunit)
    stop = _standardize_slcval(stop, epoch=epoch, tdunit=tdunit)

    if interior and not val_endpoint:
        logger.warning("For interior slices, val_endpoint must be True, I'll "
                       "change that for you.")
        val_endpoint = True

    try:
        epsilon = tol * np.finfo(arr.dtype).eps
    except ValueError:
        # array is probably of type numpy.int*
        epsilon = 0.01

    _step = 1 if step is None else int(step)
    epsilon_step = epsilon if _step > 0 else -epsilon

    # print("?!? |{0}|  |{1}|".format(start, stop))

    startstop = [start, stop]
    eps_sign = [1, -1]

    # if start or stop is not an int, try to make it one
    byval = [None] * 2
    for i in range(2):
        s = startstop[i]
        _epsilon_step = epsilon_step

        # s is a string if and only if it's slice by float value
        if isinstance(s, string_types):
            assert len(s) != 0  # startstop[0] = None ???
            byval[i] = float(s[:-1])
            # print("byval[", i, "]", s, byval[i])

            if _epsilon_step:
                diff = arr - byval[i] + (eps_sign[i] * _epsilon_step)
            else:
                diff = arr - byval[i]
            zero = np.array([0]).astype(diff.dtype)[0]

            # FIXME: there is far too much decision making here
            if i == 0:
                # start
                if _step > 0:
                    diff = np.ma.masked_less(diff, zero)
                else:
                    diff = np.ma.masked_greater(diff, zero)

                if np.ma.count(diff) == 0:
                    # start value is past the wrong end of the array
                    if _step > 0:
                        startstop[i] = len(arr)
                    else:
                        # start = -len(arr) - 1
                        # having a value < -len(arr) won't play
                        # nice with make_fwd_slice, but in this
                        # case, the slice will have no data, so...
                        return 0, 0, step
                else:
                    startstop[i] = int(np.argmin(np.abs(diff)))
            else:
                # stop
                if _step > 0:
                    diff = np.ma.masked_greater(diff, zero)
                    if np.ma.count(diff) == 0:
                        # stop value is past the wong end of the array
                        startstop[i] = 0
                    else:
                        startstop[i] = int(np.argmin(np.abs(diff)))
                        if val_endpoint:
                            startstop[i] += 1
                else:
                    diff = np.ma.masked_less(diff, zero)
                    if np.ma.count(diff) == 0:
                        # stop value is past the wrong end of the array
                        startstop[i] = len(arr)
                    else:
                        startstop[i] = int(np.argmin(np.abs(diff)))
                        if val_endpoint:
                            if startstop[i] > 0:
                                startstop[i] -= 1
                            else:
                                # 0 - 1 == -1 which would wrap to the end of
                                # of the array... instead, just make it None
                                startstop[i] = None
            # print("startstop[", i, "]", startstop[i])

    # turn start, stop, step into indices
    start, stop = startstop
    sss = [start, stop, step]
    for i, s in enumerate(sss):
        if s is None:
            pass
        elif isinstance(s, string_types):
            sss[i] = int(s)
        else:
            sss[i] = s.__index__()

    # start stop and step are now all integers... yay
    start, stop, step = sss
    if interior:
        start, stop, step = _interiorize_slice(arr, byval[0], byval[1],
                                               start, stop, step)
    return (start, stop, step)

def _expand_newaxis(arrs, slices):
    for i, sl in enumerate(slices):
        if sl in [None, np.newaxis, "None", "newaxis"]:
            if len(arrs) < len(slices):
                arrs.insert(i, None)
            slices[i] = np.newaxis
    return arrs, slices

def _prepare_selections(arrs, selections):
    """make arrs and selections something that can be zipped together"""
    if isinstance(selections, string_types):
        selections = "".join(selections.split())
        selections = selections.split(",")

    if not isinstance(selections, (list, tuple)):
        raise TypeError("To wrap a single slice use vutil.to_slice(...)")

    if arrs is None:
        arrs = [None] * len(selections)

    arrs, selections = _expand_newaxis(arrs, selections)

    if len(arrs) != len(selections):
        raise RuntimeError("len(arrs) must == len(slices):: {0} {1}"
                           "".format(len(arrs), len(selections)))

    return arrs, selections

def _standardize_selection(selection):
    std_selection = None
    is_slice_list = None

    if isinstance(selection, string_types):
        selection = "".join(selection.split())
        selection = parse_time_slice_str(selection)

    if hasattr(selection, "__index__"):
        std_selection = selection
        is_slice_list = False
    elif selection in [None, np.newaxis, "newaxis", "None", "none"]:
        std_selection = None
        is_slice_list = False
    else:
        is_slice_list = True

        if isinstance(selection, slice):
            std_selection = [selection.start, selection.stop, selection.step]
        elif isinstance(selection, string_types):
            # kill whitespace
            std_selection = [v.strip() for v in selection.split(":")]
        else:
            if not isinstance(selection, list):
                try:
                    selection = list(selection)
                except TypeError:
                    selection = [selection]
            std_selection = selection

        if len(std_selection) > 3:
            raise ValueError("slices can have at most start, stop, step:"
                             "{0}".format(selection))

    return std_selection, is_slice_list

def to_slices(arrs, selections, val_endpoint=True, interior=False, epoch=None,
              tdunit='s', tol=100):
    """Wraps :py:func:`to_slice` for multiple arrays / slices

    Args:
        arrs (list, None): list of arrays for float lookups, must be
            the same length as s.split(','). If all slices are by
            index, then `arrs` can be `None`.
        selections (list, str): list of things that :py:func:`to_slice`
            understands, or a comma separated string of slices
        val_endpoint (bool): passed to :py:func:`extract_index` if needed
        interior (bool): passed to :py:func:`extract_index` if needed
        epoch (datetime64-like): Epoch for to go datetime64 <-> float
        tdunit (str): Presumed time unit for floats
        tol (int): passed to :py:func:`extract_index` if needed

    Returns:
        tuple of slice objects

    See Also:
        * :py:func:`to_slice`
        * :py:func:`extract_index`
    """
    arrs, selections = _prepare_selections(arrs, selections)
    ret = []
    for arr, slc in izip(arrs, selections):
        ret.append(to_slice(arr, slc, val_endpoint=val_endpoint, interior=interior,
                            epoch=epoch, tdunit=tdunit, tol=tol))
    return tuple(ret)

def to_slice(arr, selection, val_endpoint=True, interior=False, epoch=None,
             tdunit='s', tol=100):
    """Convert anything describing a slice to a slice object

    Args:
        arr (array): Array that you're going to slice if you specify any
            parts of the slice by value. If all slices are by index,
            `arr` can be `None`.
        selection (int, str, slice): Something that can be turned into
            a slice. Ints are returned as-is. Slices and strings are
            parsed to see if they contain indices, or values. Values
            are strings that contain a number followed by an 'f'. Refer
            to :py:func:`extract_index` for the slice-by-value
            semantics.
        val_endpoint (bool): passed to :py:func:`extract_index` if needed
        interior (bool): passed to :py:func:`extract_index` if needed
        epoch (datetime64-like): Epoch for to go datetime64 <-> float
        tdunit (str): Presumed time unit for floats
        tol (int): passed to :py:func:`extract_index` if needed

    Returns:
        slice object or int

    Raises:
        ValueError: Description

    See Also:
        * :py:func:`extract_index`
    """
    std_selection, is_slice_list = _standardize_selection(selection)

    if is_slice_list:
        slclst = std_selection
        if arr is None:
            if len(slclst) == 1:
                # FIXME: i'm not sure if this is guarded for cases where
                #        slclst[0] is something like '0.0'
                if isinstance(slclst[0], string_types) and '.' in slclst[0]:
                    raise RuntimeError("Fell through a crack.")
                ret = int(slclst[0])
            else:
                ret = slice(*[None if a is None else int(a) for a in slclst])
        elif len(slclst) == 1:
            ret = _closest_index(arr, slclst[0], epoch=epoch, tdunit=tdunit)
        else:
            # sss -> start step step
            sss = extract_index(arr, *slclst, val_endpoint=val_endpoint,
                                interior=interior, epoch=epoch, tdunit=tdunit,
                                tol=tol)
            ret = slice(*sss)
    else:
        ret = std_selection

    return ret

def _interiorize_slice(arr, start_val, stop_val, start, stop, step,
                       verify=True):
    """Ensure start_val and stop_val interior to the sliced array

    Args:
        arr (sequence): sequence being sliced
        start_val (None, float): Value used to find start index, or
            None to not adjust start
        stop_val (None, float): Value used to find stop index, or
            None to net adjust stop
        start (int): start of slice
        stop (int): stop of slice
        step (int): step of slice
        verify (bool): verify that start_val and stop_val are interior
            to the resulting sequence

    Returns:
        (start, stop, step), adusted sliced start:stop:step indices
    """
    step = 1 if step is None else step

    if start_val is not None and start is not None:
        start_slcval = arr[start]

        if step > 0 and start_slcval > start_val:
            # print("ADJUSTING start fwd")
            if start == 0:
                start = None
            else:
                start -= 1
        elif step < 0 and start_slcval < start_val:
            # print("ADJUSTING start rev")
            if start == -1:
                start = None
            else:
                start += 1

    if stop_val is not None and stop is not None:
        if step > 0:
            stop_slcval = arr[0] if stop == 0 else arr[stop - 1]
            if stop_slcval < stop_val:
                # print("ADJUSTING stop fwd")
                if stop == -1:
                    stop = None
                else:
                    stop += 1
        elif step < 0:
            stop_slcval = arr[-1] if stop in (-1, len(arr)) else arr[stop + 1]
            if stop_slcval > stop_val:
                # print("ADJUSTING stop rev")
                if stop == 0:
                    stop = None
                else:
                    stop -= 1

    # for debug, check that interior does what it says
    if verify:
        subarr = arr[start:stop:step]
        ends = [v for v in sorted([subarr[0], subarr[-1]])]
        for v in [start_val, stop_val]:
            if v is not None:
                if v < ends[0] or v > ends[-1]:
                    raise RuntimeError("Logic issue in interiorize, "
                                       "v: {0} ends: {1}".format(v, ends))

    return start, stop, step

def make_slice_inclusive(start, stop=None, step=None):
    """Extend the end of a slice by 1 element

    Chances are you don't want to use this function.

    Args:
        start (int, None): same as a slice.start
        stop (int, None): same as a slice.stop
        step (int, None): same as a slice.step

    Returns:
        (start, stop, step) To be given to slice()
    """
    if stop is None:
        return start, stop, step

    if step is None or step > 0:
        if stop == -1:
            stop = None
        else:
            stop += 1
    else:
        if stop == 0:
            stop = None
        else:
            stop -= 1
    return start, stop, step

def selections2values(arrs, selections, nr_dims, epoch=None, tdunit='s'):
    """find the extrema values for a given selection

    Args:
        arrs (None, sequence): array that is being selected, if this is
            None, then the output will contain np.nan where it can not
            infer values.
        selections (slice, str): selections that could be given to
            :py:func:`to_slices`
        nr_dims (int): arrs and selections are expanded to this length
        epoch (datetime64-like): Epoch for to go datetime64 <-> float
        tdunit (str): Presumed time unit for floats

    Returns:
        ndarray with shape (nr_dims, 2) of extents as floats

        - If arr is None and start/stop are None, then they will become
          +/- inf depending on the sign of step.
        - If arr is None and start/stop are slice-by-index, they will
          become NaN.
    """
    arrs, selections = _prepare_selections(arrs, selections)
    n_new = max(nr_dims - len(arrs), 0)
    arrs += [None] * n_new
    selections += [slice(None)] * n_new

    ret = []
    for arr, sel in izip(arrs, selections):
        ret.append(selection2values(arr, sel, epoch=epoch, tdunit=tdunit))
    return np.vstack(ret)

def selection2values(arr, selection, epoch=None, tdunit='s'):
    """find the extrema values for a given selection

    Args:
        arr (None, sequence): array that is being selected, if this is
            None, then the output will contain np.nan where it can not
            infer values.
        selection (slice, str): a single selection that could be given
            to :py:func:`to_slice`
        epoch (datetime64-like): Epoch for to go datetime64 <-> float
        tdunit (str): Presumed time unit for floats

    Returns:
        (start_val, stop_val) as floats

        - If arr is None and start/stop are None, then they will become
          +/- inf depending on the sign of step.
        - If arr is None and start/stop are slice-by-index, they will
          become NaN.
    """
    std_selection, is_slice_list = _standardize_selection(selection)

    if is_slice_list:
        slclst = std_selection
        if len(slclst) == 1:
            # FIXME: i'm not sure if this is guarded for cases where
            #        slclst[0] is something like '0.0', but i think it's ok
            #        since this goes through slice2values, which does guard this
            slclst += [slclst[0], 1]
        ret = slice2values(arr, slclst[0], slclst[1], slclst[2], epoch=epoch,
                           tdunit=tdunit)
    else:
        if hasattr(std_selection, "__index__"):
            if arr is None:
                ret = (np.nan, np.nan)
            else:
                ret = tuple([arr[std_selection]] * 2)
        elif std_selection is None:
            ret = (-np.inf, np.inf)
        else:
            raise RuntimeError("Not sure how we got here")

    return ret

def slice2values(arr, start, stop, step=None, epoch=None, tdunit='s'):
    """Find the values that correspond with start and stop

    Returns:
        (start_val, stop_val) as floats

        - If arr is None and start/stop are None, then they will become
          +/- inf depending on the sign of step.
        - If arr is None and start/stop are slice-by-index, they will
          become NaN.
    """
    step = 1 if step is None else step
    if arr is None:
        arr_flt = None
        arr_extents = [-np.inf, np.inf]
    else:
        arr_flt, epoch = _arr2float(arr, epoch=epoch, tdunit=tdunit)
        arr_extents = [arr_flt[0], arr_flt[-1]]

    if step < 0:
        arr_extents = arr_extents[::-1]

    start = _standardize_slcval(start, epoch=epoch, tdunit=tdunit)
    stop = _standardize_slcval(stop, epoch=epoch, tdunit=tdunit)

    # start/stop are strings if and only if they are slice by float value
    ss = [start, stop]
    for i, s in enumerate(ss):
        if isinstance(s, string_types):
            ss[i] = float(s[:-1])
        elif s is None:
            ss[i] = arr_extents[i]
        elif hasattr(s, '__index__'):
            if arr_flt is None:
                ss[i] = np.nan
            else:
                ss[i] = arr_flt[ss[i]]
        else:
            raise TypeError("Not sure what to do with {0} [type = {1}]"
                            "".format(s, type(s)))
    start, stop = ss
    return start, stop

##
## EOF
##
