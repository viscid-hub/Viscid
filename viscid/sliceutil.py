#!/usr/bin/env python
"""Convenience functions for slicing by value"""

from __future__ import print_function
from itertools import count

import numpy as np

from viscid import logger
from viscid.compat import izip, string_types
from viscid.npdatetime import (is_datetime_like, is_timedelta_like,
                               as_datetime64, as_timedelta64)


__all__ = ["make_fwd_slice", "to_slices", "to_slice", "make_slice_inclusive"]


def value_is_float_not_int(value):
    """Return if value is a float and not an int"""
    # this is klugy and only needed to display deprecation warnings
    try:
        int(value)
        return False
    except ValueError:
        try:
            float(value)
            return True
        except ValueError:
            return False
    except TypeError:
        return False

def convert_timelike(value):
    if value in (None, ''):
        return None
    elif is_datetime_like(value, conservative=True):
        return as_datetime64(value)
    elif is_timedelta_like(value, conservative=True):
        return as_timedelta64(value)
    else:
        return value

def convert_deprecated_floats(value, varname="value"):
    if value_is_float_not_int(value):
        # TODO: eventually, a ValueError should be raised here
        s = ("DEPRECATION...\n"
             "Slicing by float is deprecated. The slice by value syntax is \n"
             "now a string that has a trailing 'f', as in 'x=0f' [{0} = {1}]"
             "".format(varname, value))
        logger.warning(s)
        value = "{0}f".format(value)
    return value

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
            assert start is None or (0 <= start and start <= L), \
                "start (={0}) is outside range".format(start)
            assert start is None or (0 <= start and start <= L), \
                "start (={0}) is outside range".format(start)
            assert start is None or stop is None or start < stop, \
                "bad slice ordering: {0} !< {1}".format(start, stop)
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

def _closest_index(arr, value):
    float_err_msg = ("Slicing by floats is no longer supported. If you "
                     "want to slice by location, suffix the value with "
                     "'f', as in 'x = 0f'.")
    value = convert_deprecated_floats(value, "value")

    try:
        value = value.rstrip()
        if len(value) == 0:
            raise ValueError("Can't slice with nothing")
        elif value[-1] == 'f':
            index = int(np.argmin(np.abs(np.asarray(arr) - float(value[:-1]))))
        else:
            index = int(value)

    except AttributeError:
        index = value.__index__()

    return index

def extract_index(arr, start=None, stop=None, step=None, endpoint=True,
                  tol=100):
    """Get integer indices for slice parts

    If start, stop, or step are strings, they are either cast to
    integers or used for a float lookup if they have a trailing 'f'.

    An example float lookup is::
        >>> [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]['0.2f:0.6f:2']
        [0.2, 0.4, 0.6]

    The rules for float lookup endpoints are:
        - The slice will never include an element whose value in arr
          is < start (or > if the slice is backward)
        - The slice will never include an element whose value in arr
          is > stop (or < if the slice is backward)
        - !! The slice WILL INCLUDE stop if you don't change endpoint.
          This is different from normal slicing, but
          it's more natural when specifying a slice as a float.
          To this end, an epsilon tolerance can be given to
          determine what's close enough.
        - TODO: implement floating point steps, this is tricky
          since arr need not be uniformly spaced, so step is
          ambiguous in this case

    Args:
        arr (ndarray): filled with floats to do the lookup
        start (None, int, str): like slice().start
        stop (None, int, str): like slice().stop
        step (None, int): like slice().step
        endpoint (bool): iff True then include stop in the slice.
            Set to False to get python slicing symantics when it
            comes to excluding stop, but fair warning, python
            symantics feel awkward here. Consider the case
            [0.1, 0.2, 0.3][:0.25]. If you think this should include
            0.2, then leave keep endpoint=True.
        tol (int): number of machine epsilons to consider
            "close enough"

    Returns:
        start, stop, step after floating point vals have been
        converted to integers
    """
    float_err_msg = ("Slicing by floats is no longer supported. If you "
                     "want to slice by location, suffix the value with "
                     "'f', as in 'x = 0f'.")
    arr = np.asarray(arr)
    try:
        epsilon = tol * np.finfo(arr.dtype).eps
    except ValueError:
        # array is probably of type numpy.int*
        epsilon = 0.01

    _step = 1 if step is None else int(step)
    epsilon_step = epsilon if _step > 0 else -epsilon

    start = convert_timelike(start)
    stop = convert_timelike(stop)

    start = convert_deprecated_floats(start, "start")
    stop = convert_deprecated_floats(stop, "stop")

    # print("?!? |{0}|  |{1}|".format(start, stop))

    startstop = [start, stop]
    eps_sign = [1, -1]

    # if start or stop is not an int, try to make it one
    for i in range(2):
        byval = None
        s = startstop[i]
        _epsilon_step = epsilon_step

        if is_datetime_like(s, conservative=True):
            byval = s.astype(arr.dtype)
            _epsilon_step = 0
        elif is_timedelta_like(s, conservative=True):
            byval = s.astype(arr.dtype)
            _epsilon_step = 0
        else:
            try:
                s = s.strip()
                if len(s) == 0:
                    startstop[i] = None
                elif s[-1] == 'f':
                    byval = float(s[:-1])
            except AttributeError:
                pass

        if byval is not None:
            if _epsilon_step:
                diff = arr - byval + (eps_sign[i] * _epsilon_step)
            else:
                diff = arr - byval
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
                        if endpoint:
                            startstop[i] += 1
                else:
                    diff = np.ma.masked_less(diff, zero)
                    if np.ma.count(diff) == 0:
                        # stop value is past the wrong end of the array
                        startstop[i] = len(arr)
                    else:
                        startstop[i] = int(np.argmin(np.abs(diff)))
                        if endpoint:
                            if startstop[i] > 0:
                                startstop[i] -= 1
                            else:
                                # 0 - 1 == -1 which would wrap to the end of
                                # of the array... instead, just make it None
                                startstop[i] = None
    start, stop = startstop

    # turn start, stop, step into indices
    sss = [start, stop, step]
    for i, s in enumerate(sss):
        if s is None:
            pass
        elif isinstance(s, string_types):
            sss[i] = int(s)
        else:
            sss[i] = s.__index__()
    return sss

def _expand_newaxis(arrs, slices):
    for i, sl in enumerate(slices):
        if sl in [None, np.newaxis, "None", "newaxis"]:
            if len(arrs) < len(slices):
                arrs.insert(i, None)
            slices[i] = np.newaxis
    return arrs, slices

def to_slices(arrs, slices, endpoint=True, tol=100):
    """Wraps :py:func:`to_slice` for multiple arrays / slices

    Args:
        arrs (list, None): list of arrays for float lookups, must be
            the same length as s.split(','). If all slices are by
            index, then `arrs` can be `None`.
        slices (list, str): list of things that
            :py:func:`to_slice` understands, or a comma
            separated string of slices
        endpoint (bool): passed to :py:func:`extract_index` if needed
        tol (int): passed to :py:func:`extract_index` if needed

    Returns:
        tuple of slice objects

    See Also:
        * :py:func:`to_slice`
        * :py:func:`extract_index`
    """
    try:
        slices = "".join(slices.split())
        slices = slices.split(",")
    except AttributeError:
        pass

    if not isinstance(slices, (list, tuple)):
        raise TypeError("To wrap a single slice use vutil.to_slice(...)")

    if arrs is None:
        arrs = [None] * len(slices)

    arrs, slices = _expand_newaxis(arrs, slices)

    if len(arrs) != len(slices):
        raise ValueError("len(arrs) must == len(slices):: {0} {1}"
                         "".format(len(arrs), len(slices)))

    ret = []
    for arr, slcstr in izip(arrs, slices):
        ret.append(to_slice(arr, slcstr, endpoint=endpoint, tol=tol))
    return tuple(ret)

def to_slice(arr, s, endpoint=True, tol=100):
    """Convert anything describing a slice to a slice object

    Args:
        arr (array): Array that you're going to slice if you specify any
            parts of the slice by value. If all slices are by index,
            `arr` can be `None`.
        s (int, str, slice): Something that can be turned into a slice.
            Ints are returned as-is. Slices and strings are parsed to
            see if they contain indices, or values. Values are strings
            that contain a number followed by an 'f'. Refer to
            :py:func:`extract_index` for the slice-by-value
            semantics.
        endpoint (bool): passed to :py:func:`extract_index` if
            needed
        tol (int): passed to :py:func:`extract_index` if
            needed

    Returns:
        slice object or int

    See Also:
        * :py:func:`extract_index`
    """
    ret = None

    try:
        # kill whitespace
        s = "".join(s.split())
    except AttributeError:
        pass

    if hasattr(s, "__index__"):
        ret = s
    elif s in [np.newaxis, None, "newaxis", "None"]:
        ret = np.newaxis
    else:
        if isinstance(s, slice):
            slclst = [s.start, s.stop, s.step]
        elif isinstance(s, string_types):
            # kill whitespace
            slclst = [v.strip() for v in s.split(":")]
        else:
            if not isinstance(s, list):
                try:
                    s = list(s)
                except TypeError:
                    s = [s]
            slclst = s

        if len(slclst) > 3:
            raise ValueError("slices can have at most start, stop, step:"
                             "{0}".format(s))
        # sss -> start step step

        if arr is None:
            if len(slclst) == 1:
                ret = int(slclst[0])
            else:
                ret = slice(*[None if a is None else int(a) for a in slclst])
        elif len(slclst) == 1:
            ret = _closest_index(arr, slclst[0])
        else:
            sss = extract_index(arr, *slclst, endpoint=endpoint,
                                tol=tol)
            ret = slice(*sss)
    return ret

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

##
## EOF
##
