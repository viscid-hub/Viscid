#!/usr/bin/env python
"""Handle slice by index / location (value)"""

from __future__ import print_function
from datetime import datetime, timedelta
from itertools import count
import re

import numpy as np

from viscid import logger
from viscid.compat import izip, string_types
from viscid.npdatetime import (is_datetime_like, is_timedelta_like,
                               as_datetime64, as_timedelta64, time_diff)


_R_MULTILEVEL_BRACKETS = r"\[[^\]]*\[[^\]]*\]"
_R_IN_BRACKS = r"\[[^\[\]]+\]"
_R_DATE = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
_R_TIME02 = r"[0-9]{2}(?::[0-9]{2}){0,2}(?:\.[0-9]+)?"
_R_TIME12 = r"[0-9]{2}(?::[0-9]{2}){1,2}(?:\.[0-9]+)?"
RE_DTIME_GROUP = (r"(?:[utUT]*{date}(?:[tT]{time02})?|[utUT]+{time02}|"
                  r"[utUT]*{time12})".format(date=_R_DATE, time02=_R_TIME02,
                                             time12=_R_TIME12))
_R_DTIME = re.compile(r"(\s*{0}\s*)".format(RE_DTIME_GROUP))
# _R_DTIME_SLC is _R_DTIME that must begin with r'[ut]+'
RE_DTIME_SLC_GROUP = (r"[utUT]+(?:{date}(?:[tT]{time})?|{time})"
                      r"".format(date=_R_DATE, time=_R_TIME02))
_R_DTIME_SLC = re.compile(r"(\s*{0}\s*)".format(RE_DTIME_SLC_GROUP))

_emit_deprecated_float_warning = True


__all__ = ["prune_comp_sel", "raw_sel2sel_list", "fill_nd_sel_list",
           "standardize_sel_list", "standardize_sel", "standardize_value",
           "std_sel_list2index", "std_sel2index",
           "sel_list2values", "sel2values", "selection2values",
           "make_fwd_slice", "all_slices_none"
           ]


def prune_comp_sel(sel_list, comp_names):
    """Discover and remove vector component from selection list"""
    comp_slc = slice(None)
    comp_idx = None

    # discover sel_names and rip out any vector-component slices if given
    for i, s in enumerate(sel_list):
        try:
            s = s.strip().lower()
            if s in comp_names:
                if comp_idx is not None:
                    raise IndexError("Multiple vector component slices given, {0}"
                                     "".format(tuple(sel_list)))
                comp_slc = comp_names.index(s)
                comp_idx = i
        except (TypeError, AttributeError):
            pass

    if comp_idx is not None:
        sel_list.pop(comp_idx)
        comp_idx = None

    return sel_list, comp_slc

def raw_sel2sel_list(sel):
    """Turn generic selection into something standard we can work with

    Args:
        sel (object): some slice selection

    Returns:
        (sel_list, input_type): items in sel_list are guarenteed to be
            of type {slice, int, complex, string, numpy.ndarray}

    Raises:
        TypeError: slice-by-array invalid dtype
        ValueError: slice-by-array not 1D
    """
    # type(None) includes np.newaxis
    valid_types = (slice, int, np.integer, float, np.floating, complex, np.complex,
                   np.complexfloating, datetime, np.datetime64, timedelta,
                   np.timedelta64, type(Ellipsis), type(None))

    if isinstance(sel, tuple):
        sel_list = list(sel)
        # input_type = 'tuple'
    elif isinstance(sel, list):
        # DANGER, this is different behavior from tuple since sel
        #         will become an ndarray before we return... maybe
        #         supply a warning like numpy 1.15 does?
        sel_list = [sel]
        # input_type = 'list'
    elif isinstance(sel, valid_types):
        sel_list = [sel]
        # input_type = 'single-value'
    elif isinstance(sel, string_types):
        sel = sel.replace('_', ',')
        # mutate commas between brackets (they indicate strings)
        if re.search(_R_MULTILEVEL_BRACKETS, sel) is not None:
            raise IndexError("Slice-by-array in Viscid is limited to 1D "
                             "arrays since coordinate arrays become "
                             "ambiguous otherwise.")
        sel = re.sub(_R_IN_BRACKS, lambda x: x.group(0).replace(",", "@"), sel)
        sel_list = [s for s in sel.split(",")]
        # put the commas back into the arrays
        sel_list = [s.replace('@', ',') for s in sel_list]
        # input_type = 'string'
    else:
        sel_list = [sel]
        # input_type = 'other'

    for i, s in enumerate(sel_list):
        if not isinstance(s, valid_types + string_types):
            sel_list[i] = np.asarray(s)

    for _, s in enumerate(sel_list):
        if isinstance(s, np.ndarray):
            is_valid_dtype = (np.issubdtype(s.dtype, np.integer)
                              or np.issubdtype(s.dtype, np.complexfloating)
                              or np.issubdtype(s.dtype, np.bool_))
            if not is_valid_dtype:
                if s.dtype == np.object:
                    raise IndexError("Slice interpreted as slice-by-array of "
                                     "object dtype. If you did\nnot intend to "
                                     "slice-by-array, then you probably gave "
                                     "an n-dimensional slice\nas a list instead "
                                     "of a tuple... oops.")
                raise IndexError("Slice-by-array with invalid dtype '{0}' (must "
                                 "be (int, bool, complex, timedeta64, datetime64)."
                                 "".format(s.dtype))

            is_valid_shape = len(s.shape) == 1
            if not is_valid_shape:
                # print("what is s??", type(s), s.shape, s)
                raise IndexError("Slice-by-array in Viscid is limited to 1D "
                                 "arrays since coordinate arrays become "
                                 "ambiguous otherwise.")
    return sel_list

def fill_nd_sel_list(sel_list, ax_names):
    """fully determine a sparsely selected sel_list"""

    if (len(sel_list) == 1 and not isinstance(sel_list[0], np.ndarray)
        and sel_list in ([Ellipsis], ['...'], ['Ellipsis'], ['ellipsis'])):
        # short circuit all logic if sel_list == [Ellipsis]
        full_sel_list = [slice(None) for _ in ax_names]
        full_ax_names = [name for name in ax_names]
        full_newdim_flags = [False for _ in ax_names]
        return full_sel_list, full_ax_names, full_newdim_flags

    sel_list0 = tuple(sel_list)

    sel_names = [None] * len(sel_list)

    # discover sel_names
    for i, s in enumerate(sel_list):
        if isinstance(s, string_types) and '=' in s:
            sel_names[i], sel_list[i] = [ss.strip() for ss in s.split('=', 1)]

    # discover which items in sel_list are ellipsis or newaxis
    pre_elip_newax_idxs = []
    post_elip_newax_idxs = []
    ellipsis_idx = -1

    for i, s in enumerate(sel_list):
        if isinstance(s, (list, np.ndarray)):
            pass
        elif s in (Ellipsis, ) or (isinstance(s, string_types)
                                 and s.strip().lower() in ('ellipsis', '...')):
            sel_list[i] = Ellipsis
            if ellipsis_idx < 0:
                ellipsis_idx = i
            else:
                raise IndexError("Only one ellipsis per slice please, {0}"
                                 "".format(sel_list0))
        elif s in (None, np.newaxis) or (isinstance(s, string_types)
                                         and s.strip().lower() in ('none',
                                                                   'newaxis')):
            sel_list[i] = np.newaxis
            if ellipsis_idx < 0:
                pre_elip_newax_idxs.append(i)
            else:
                post_elip_newax_idxs.append(i)
    newax_idxs = pre_elip_newax_idxs + post_elip_newax_idxs
    n_newax = len(newax_idxs)

    n_named_axes = len([None for name in sel_names if name is not None])
    n_named_newaxes = len([None for sel, name in zip(sel_list, sel_names)
                           if name is not None and sel == np.newaxis])

    # replace ellipsis with some number of slice(None)
    if ellipsis_idx >= 0:
        if n_named_axes > n_named_newaxes:
            raise IndexError("Field indexing with Ellipsis can only be used "
                             "with named axes if those named axes are "
                             "numpy.newaxis, sel = {0}".format(sel_list0))
        n_fill = len(ax_names) + n_newax - (len(sel_list) - 1)
        idx = ellipsis_idx
        sel_list = sel_list[:idx] + [slice(None)] * n_fill + sel_list[idx + 1:]
        sel_names = sel_names[:idx] + [None] * n_fill + sel_names[idx + 1:]
        newax_idxs = [i if i < idx else i + n_fill - 1 for i in newax_idxs]

    # now let's assemble the final full slice list / ax names...
    full_ax_names = []
    full_sel_list = []
    full_newdim_flags = []

    remaining_ax_names = list(ax_names)
    n_newax_seen = 0
    n_unnamed_newax_seen = 0
    for i, name, sel in zip(count(), sel_names, sel_list):
        if name is None:
            if i in newax_idxs:
                name = 'new-x{0:d}'.format(n_unnamed_newax_seen)
                n_unnamed_newax_seen += 1
                n_newax_seen += 1
            else:
                name = remaining_ax_names.pop(0)
        else:  # named selection
            if i in newax_idxs:
                n_newax_seen += 1
                if name in full_ax_names or name in ax_names:
                    raise IndexError("New axis duplicates name, {0}"
                                     "".format(sel_list0))
            else:  # named selection that is not newaxis
                if name in full_ax_names:
                    j = full_ax_names.index(name)
                    if full_sel_list[j] != slice(None):
                        raise IndexError("Axis '{0}' is repeated in slice {1}"
                                         "".format(name, sel_list0))
                    full_ax_names[j] = name
                    full_sel_list[j] = sel
                    full_newdim_flags[j] = False
                    continue  # <- poor style, sorry
                else:
                    n_skipped = remaining_ax_names.index(name)
                    full_ax_names += remaining_ax_names[:n_skipped]
                    full_sel_list += [slice(None)] * n_skipped
                    full_newdim_flags += [False] * n_skipped
                    remaining_ax_names = remaining_ax_names[n_skipped + 1:]

        full_ax_names.append(name)
        full_sel_list.append(sel)
        full_newdim_flags.append(i in newax_idxs)

    if remaining_ax_names:
        full_ax_names += remaining_ax_names
        full_sel_list += [slice(None)] * len(remaining_ax_names)
        full_newdim_flags += [False] * len(remaining_ax_names)

    # print("sel0: ", sel_list0, '\n',
    #       "sel_list: ", sel_list, '\n',
    #       "full_ax_names: ", full_ax_names, '\n',
    #       "full_sel_list: ", full_sel_list, '\n',
    #       "full_newdim_flags: ", full_newdim_flags, '\n',
    #       "----\n",
    #       "len(full_ax_names): ", len(full_ax_names), '\n',
    #       "len(ax_names): ", len(ax_names), '\n',
    #       "n_newax: ", n_newax, '\n',
    #       sep='')

    assert len(full_ax_names) == len(ax_names) + n_newax
    assert len(full_sel_list) == len(ax_names) + n_newax

    return full_sel_list, full_ax_names, full_newdim_flags

#######################################################################

def _warn_deprecated_float(val, varname='value'):
    global _emit_deprecated_float_warning  # pylint: disable=global-statement
    if _emit_deprecated_float_warning:
        frame = _user_written_stack_frame()
        s = ("DEPRECATION...\n"
             "Slicing by float is deprecated. Slicing by location is now \n"
             "performed with an imaginary number, or a string with a trailing \n"
             "'f', as in 0j, 'x=0j', or 'x=0f'. This warning comes from:\n"
             "    {0}:{1}\n"
             "    >>> {2}"
             "".format(frame[1], frame[2], frame[4][0].strip()))
        logger.warning(s)
        _emit_deprecated_float_warning = False

def _is_time_str(s):
    m = re.match(_R_DTIME, s)
    return m is not None and m.end() - m.start() == len(s)

def _split_slice_str(sel):
    all_times = re.findall(_R_DTIME_SLC, sel)
    at_sel = re.sub(_R_DTIME_SLC, '@', sel)
    split_sel = [s.strip() for s in at_sel.split(':')]
    for s in all_times:
        split_sel[split_sel.index('@')] = s
    return split_sel

#######################################################################

def standardize_sel_list(sel_list):
    """turn all selection list elements into fundamental python types"""
    for i, sel in enumerate(sel_list):
        sel_list[i] = standardize_sel(sel)
    return sel_list

def standardize_sel(sel):
    """turn selection list element into fundamental python types"""
    if isinstance(sel, string_types):
        sel = sel.strip().lower()

        if sel[0] == '[' and sel[-1] == ']':
            sel = standardize_value(sel, bool_argwhere=True)
        elif ':' in sel:
            sel = slice(*[standardize_value(s, bool_argwhere=True)
                          for s in _split_slice_str(sel)])
            assert isinstance(sel.step, (type(None), int, np.integer))
            assert not isinstance(sel.step, (np.datetime64, np.timedelta64))
        else:
            sel = standardize_value(sel, bool_argwhere=True)
    elif isinstance(sel, slice):
        sel = slice(*[standardize_value(s, bool_argwhere=True)
                      for s in [sel.start, sel.stop, sel.step]])
        if sel.step is None:
            sel = slice(sel.start, sel.stop, 1)
    else:
        sel = standardize_value(sel, bool_argwhere=True)
    return sel

def standardize_value(sel, bool_argwhere=False):
    """Turn a value element to fundamental type or array

    Returns:
        One of the following::
            - None
            - np.newaxis
            - Ellipsis
            - bool
            - int
            - complex (slice by value)
            - numpy.datetime64
            - numpy.timedelta64
            - ndarray
                - numpy.integer
                - numpy.bool\_
                - numpy.timedelta64
                - numpy.datetime64
                - numpy.complex
    """
    if isinstance(sel, (np.datetime64, np.timedelta64)):
        pass
    elif isinstance(sel, (int, np.integer)):
        pass
    elif isinstance(sel, (complex, np.complex, np.complexfloating)):
        assert sel.real == 0.0
    elif isinstance(sel, (float, np.floating)):
        _warn_deprecated_float(sel)
        sel = 1j * sel
    elif isinstance(sel, (list, np.ndarray)):
        assert len(sel.shape) == 1
        assert isinstance(sel[0], (int, bool, np.integer, np.complex,
                                   np.complexfloating, np.bool_))
        if bool_argwhere and isinstance(sel[0], (bool, np.bool_)):
            sel = np.argwhere(sel).reshape(-1)
    elif sel in (np.newaxis, None, True, False, Ellipsis):
        pass
    elif isinstance(sel, string_types):
        sel = sel.strip().lower()
        if sel in ('newaxis', 'numpy.newaxis', 'np.newaxis'):
            sel = np.newaxis
        elif sel in ('none', ''):
            sel = None
        elif sel == 'true':
            sel = True
        elif sel == 'false':
            sel = False
        elif sel in ('...', 'ellipsis'):
            sel = Ellipsis
        elif sel[0] == '[' and sel[-1] == ']':
            sel = sel[1:-1].replace(',', ' ')
            if 'true' in sel or 'false' in sel:
                sel = sel.lower()
                sel = sel.replace('true', '1')
                sel = sel.replace('false', '0')
                _orig_sel = sel
                sel = np.fromstring(sel, dtype='i', sep=' ')
                if bool_argwhere:
                    sel = np.argwhere(sel).reshape(-1)
                else:
                    sel = sel.astype(np.bool_)
                if sel.shape == ():
                    raise ValueError("bool array as string did not parse: [{0}]"
                                     "".format(_orig_sel))
            else:
                sel = sel.replace('f', 'j')
                n_js = sel.count('j')
                if n_js > 0:
                    _orig_sel = sel
                    sel = np.array(sel.split()).astype(np.complex)
                    assert np.allclose(sel.real, 0.0)
                    sel = sel.imag
                    if sel.shape == ():
                        raise ValueError("float array as string did not parse: "
                                         "[{0}]".format(_orig_sel))
                    if n_js != len(sel):
                        _warn_deprecated_float(_orig_sel)
                    sel = 1j * sel
                elif all(_is_time_str(s) for s in sel.split()):
                    try:
                        sel = as_timedelta64([s.lstrip('ut') for s in sel.split()])
                    except ValueError:
                        sel = as_datetime64([s.lstrip('ut') for s in sel.split()])
                elif 'e' in sel or 'E' in sel:
                    _warn_deprecated_float(sel)
                    sel = np.fromstring(sel, dtype=np.float, sep=' ')
                else:
                    sel = np.fromstring(sel, dtype=np.integer, sep=' ')
        elif _is_time_str(sel):
            try:
                sel = as_timedelta64(sel.lstrip('ut'))
            except ValueError:
                sel = as_datetime64(sel.lstrip('ut'))
        else:
            try:
                sel = int(sel)
            except ValueError:
                try:
                    if 'j' in sel or 'f' in sel:
                        sel = sel.replace('f', 'j')
                        sel = complex(sel)
                    else:
                        sel = float(sel)
                        _warn_deprecated_float(sel)
                        sel = 1j * sel
                except ValueError:
                    raise ValueError("Unexpected std type '{0}'".format(sel))
    elif is_timedelta_like(sel, conservative=True):
        sel = as_timedelta64(sel)
    elif is_datetime_like(sel, conservative=True):
        sel = as_datetime64(sel)
    else:
        raise ValueError("Unexpected std type '{0}' ({1})"
                         "".format(sel, type(sel)))
    return sel

#######################################################################

def std_sel_list2index(std_sel_list, crd_arrs, val_endpoint=True, interior=False,
                       tdunit='s', epoch=None, tol=100):
    """turn standardized selection list into index slice"""
    return [std_sel2index(std_sel, crd_arr, val_endpoint=val_endpoint,
                          interior=interior, tdunit=tdunit, epoch=epoch)
            for std_sel, crd_arr in zip(std_sel_list, crd_arrs)
            ]

def std_sel2index(std_sel, crd_arr, val_endpoint=True, interior=False,
                  tdunit='s', epoch=None):
    """Turn single standardized selection into slice (by int or None)

    Normally (val_endpoint=True, interior=False), the rules for float
    lookup val_endpoints are::

        - The slice will never include an element whose value in arr
          is < start (or > if the slice is backward)
        - The slice will never include an element whose value in arr
          is > stop (or < if the slice is backward)
        - !! The slice WILL INCLUDE stop if you don't change
          val_endpoint. This is different from normal slicing, but
          it's more natural when specifying a slice as a float.

    If interior=True, then the slice is expanded such that start and
    stop are interior to the sliced array.

    Args:
        std_sel: single standardized selection
        arr (ndarray): filled with floats to do the lookup
        val_endpoint (bool): iff True then include stop in the slice when
            slicing-by-value (DOES NOT EFFECT SLICE-BY-INDEX).
            Set to False to get python slicing symantics when it
            comes to excluding stop, but fair warning, python
            symantics feel awkward here. Consider the case
            [0.1, 0.2, 0.3][:0.25]. If you think this should include
            0.2, then leave keep val_endpoint=True.
        interior (bool): if True, then extend both ends of the slice
            such that slice-by-location endpoints are interior to the
            slice
        epoch (datetime64-like): Epoch for to go datetime64 <-> float
        tdunit (str): Presumed time unit for floats
        tol (int): number of machine epsilons to consider
            "close enough"
    """
    idx = None

    if interior and not val_endpoint:
        logger.warning("For interior slices, val_endpoint must be True, I'll "
                       "change that for you.")
        val_endpoint = True

    if isinstance(std_sel, slice):
        assert isinstance(std_sel.step, (int, np.integer, type(None)))
        start_val = None
        stop_val = None

        orig_step = std_sel.step
        ustep = 1 if std_sel.step is None else int(std_sel.step)
        sgn = np.sign(ustep)

        if (isinstance(std_sel.start, (int, np.integer, type(None)))
            and not isinstance(std_sel.start, (np.datetime64, np.timedelta64))):
            ustart = std_sel.start
        else:
            ustart, tol = _unify_sbv_types(std_sel.start, crd_arr, tdunit='s',
                                           epoch=epoch)
            start_val = ustart
            diff = crd_arr - ustart + (tol * sgn)
            zero = np.array([0]).astype(diff.dtype)[0]

            if ustep > 0:
                diff = np.ma.masked_less(diff, zero)
            else:
                diff = np.ma.masked_greater(diff, zero)

            if np.ma.count(diff) == 0:
                # start value is past the wrong end of the array
                if ustep > 0:
                    ustart = len(crd_arr)
                else:
                    # start = -len(arr) - 1
                    # having a value < -len(arr) won't play
                    # nice with make_fwd_slice, but in this
                    # case, the slice will have no data, so...
                    return slice(0, 0, ustep)
            else:
                ustart = np.argmin(np.abs(diff))

        if (isinstance(std_sel.stop, (int, np.integer, type(None)))
            and not isinstance(std_sel.stop, (np.datetime64, np.timedelta64))):
            ustop = std_sel.stop
        else:
            ustop, tol = _unify_sbv_types(std_sel.stop, crd_arr, tdunit='s',
                                          epoch=epoch)
            stop_val = ustop
            diff = crd_arr - ustop - (tol * sgn)
            zero = np.array([0]).astype(diff.dtype)[0]

            if ustep > 0:
                diff = np.ma.masked_greater(diff, zero)
            else:
                diff = np.ma.masked_less(diff, zero)

            if ustep > 0:
                if ustop < crd_arr[0]:
                    # stop value is past the wong end of the array
                    ustop = 0
                else:
                    ustop = int(np.argmin(np.abs(diff)))
                    if val_endpoint:
                        ustop += 1
            else:
                if ustop > crd_arr[-1]:
                    # stop value is past the wrong end of the array
                    ustop = len(crd_arr)
                else:
                    ustop = int(np.argmin(np.abs(diff)))
                    if val_endpoint:
                        if ustop > 0:
                            ustop -= 1
                        else:
                            # 0 - 1 == -1 which would wrap to the end of
                            # of the array... instead, just make it None
                            ustop = None
        idx = slice(ustart, ustop, orig_step)

        if interior:
            _a, _b, _c = _interiorize_slice(crd_arr, start_val, stop_val,
                                            idx.start, idx.stop, idx.step)
            idx = slice(_a, _b, _c)

    else:
        # slice by single value or ndarray of single values (int, float, times)
        usel, _ = _unify_sbv_types(std_sel, crd_arr, tdunit='s', epoch=epoch)
        if (isinstance(usel, (int, np.integer, type(None)))
            and not isinstance(usel, (np.datetime64, np.timedelta64))):
            idx = usel
        elif isinstance(usel, np.ndarray):
            if isinstance(usel[0, 0], np.integer):
                idx = usel.reshape(-1)
            else:
                idx = np.argmin(np.abs(crd_arr.reshape(-1, 1) - usel), axis=0)
        else:
            idx = np.argmin(np.abs(crd_arr - usel))

    return idx

def _unify_sbv_types(std_val, crd_arr, tdunit='s', epoch=None):
    uval = None
    tol = None

    if std_val is None:
        pass
    elif (isinstance(std_val, (int, np.integer))
          and not isinstance(std_val, (np.datetime64, np.timedelta64))):
        uval, tol = std_val, 0
    elif (isinstance(std_val, np.ndarray) and isinstance(std_val[0], np.integer)
          and not isinstance(std_val[0], (np.datetime64, np.timedelta64))):
        uval, tol = std_val.reshape(1, -1), 0
    else:
        ndarray_slice = isinstance(std_val, np.ndarray)
        std_val = np.asarray(std_val)
        len_std_val = 1 if std_val.shape == () else len(std_val)
        std_val = std_val.reshape(1, len_std_val)

        if isinstance(std_val[0, 0], (np.complex, np.complexfloating)):
            assert np.all(std_val.real == 0)
            std_val = std_val.imag

        if crd_arr is None:
            crd_arr = np.array([0], dtype=std_val.dtype)
        else:
            crd_arr = np.asarray(crd_arr)

        # ----
        if isinstance(crd_arr[0], type(std_val[0, 0])):
            uval = std_val
        elif isinstance(crd_arr[0], np.timedelta64):
            if isinstance(std_val[0, 0], np.floating):
                uval = as_timedelta64(std_val, unit=tdunit)
            elif isinstance(std_val[0, 0], np.datetime64):
                if epoch is None:
                    raise NotImplementedError("Can't slice-by-location a timedelta64 "
                                              "axis using datetime64 value without "
                                              "epoch")
                else:
                    uval = time_diff(std_val, epoch, most_precise=True)
        elif isinstance(crd_arr[0], np.datetime64):
            if isinstance(std_val[0, 0], np.floating):
                std_val = as_timedelta64(std_val, unit=tdunit)
                uval = crd_arr[0] + std_val
            elif isinstance(std_val[0, 0], np.timedelta64):
                uval = crd_arr[0] + std_val
        elif isinstance(crd_arr[0], np.floating):
            if isinstance(std_val[0, 0], np.timedelta64):
                uval = std_val / as_timedelta64(1, unit=tdunit)
            elif isinstance(std_val[0, 0], np.datetime64):
                if epoch is None:
                    raise NotImplementedError("Can't slice-by-location a floating pt. "
                                              "axis using datetime64 value without "
                                              "epoch")
                else:
                    uval = time_diff(std_val, epoch, most_precise=True)
                    uval = uval / as_timedelta64(1, unit=tdunit)
            elif isinstance(std_val[0, 0], np.floating):
                uval = std_val.astype(crd_arr.dtype)
        else:
            raise NotImplementedError("coordinates dtype {0} can not be "
                                      "sliced by value".format(type(crd_arr[0])))

        # ----
        if uval is None:
            tol = None
        elif isinstance(uval[0, 0], np.floating):
            if len(crd_arr) > 1:
                tol = 0.0001 * np.min(np.diff(crd_arr))
            else:
                tol = 1.0
        elif isinstance(uval[0, 0], np.datetime64):
            tol = as_timedelta64(0, 's')
        elif isinstance(uval[0, 0], np.timedelta64):
            tol = as_timedelta64(0, 's')

        if uval is not None and not ndarray_slice:
            uval = uval[0, 0]

    return uval, tol

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

#######################################################################

def raw_sel2values(arrs, selection, epoch=None, tdunit='s'):
    return sel_list2values(arrs, raw_sel2sel_list(selection),
                           epoch=epoch, tdunit=tdunit)

def sel_list2values(arrs, sel_list, epoch=None, tdunit='s'):
    """find the extrema values for a given sel list"""
    if arrs is None:
        arrs = [None] * len(sel_list)
    return [sel2values(arr, sel, epoch=epoch, tdunit=tdunit)
            for arr, sel in zip(arrs, sel_list)]

def sel2values(arr, sel, epoch=None, tdunit='s'):
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
    ret = None

    std_sel = standardize_sel(sel)

    if isinstance(std_sel, slice):
        if std_sel.start is None or hasattr(std_sel.start, '__index__'):
            if arr is None:
                if std_sel.start is None and std_sel.step > 0:
                    ustart = -np.inf
                elif std_sel.start is None and std_sel.step < 0:
                    ustart = np.inf
                else:
                    ustart = np.nan
            else:
                if std_sel.start is None:
                    idx = 0 if std_sel.step > 0 else -1
                else:
                    idx = int(std_sel.start)
                ustart = arr[idx]
        else:
            ustart, _ = _unify_sbv_types(std_sel.start, arr, tdunit=tdunit,
                                         epoch=epoch)

        if std_sel.stop is None or hasattr(std_sel.stop, '__index__'):
            if arr is None:
                if std_sel.stop is None and std_sel.step > 0:
                    ustop = np.inf
                elif std_sel.stop is None and std_sel.step < 0:
                    ustop = -np.inf
                else:
                    ustop = np.nan
            else:
                if std_sel.stop is None:
                    idx = -1 if std_sel.step > 0 else 0
                else:
                    idx = int(std_sel.stop)
                ustop = arr[idx]
        else:
            ustop, _ = _unify_sbv_types(std_sel.stop, arr, tdunit=tdunit,
                                        epoch=epoch)

        ret = (ustart, ustop)
    else:
        if std_sel in (None, np.newaxis):
            ret = (-np.inf, np.inf)
        elif hasattr(std_sel, "__index__"):
            if arr is None:
                ret = (np.nan, np.nan)
            else:
                ret = tuple([arr[std_sel]] * 2)
        else:
            uval, _ = _unify_sbv_types(std_sel, arr, tdunit=tdunit, epoch=epoch)
            ret = tuple([uval] * 2)

    return ret

selection2values = sel2values

#######################################################################

# FIXME: is this graceful on slice-by-ndarray?

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

#######################################################################

def all_slices_none(slices):
    '''true iff all slices have no effect'''
    return all(not isinstance(s, (list, np.ndarray))
               and s in (slice(None), slice(None, None, 1))
               for s in slices)

def _user_written_stack_frame():
    """get the frame of first stack frame outside Viscid"""
    import inspect
    import os
    stack = inspect.stack()
    frame_info = None
    for frame_info in stack:
        if 'viscid' not in os.path.normpath(frame_info[1]):
            break
    return frame_info

##
## EOF
##
