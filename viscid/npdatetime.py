#!/usr/bin/env python
"""This is a shim to use numpy datetime64 and timedelta64 types

Times must be given in ISO 8601 format as per numpy >= 1.7.

All arguments as strings are assumed to be ZULU (UTC) time. This
keeps the interface the same for numpy versions both before and
after 1.11 when datetime64 became timezone agnostic.

Note:
    There is some trickery here. For numpy versions prior to 1.11, this
    module will force ZULU (UTC) time. Note that when these datetimes
    are printed, numpy will display the local time. This will not be
    a problem for matplotlib plots if arr.astype(datetime.datetime) is
    used. This is because python's datetime objects are timezone
    agnostic, so matplotlib won't know a thing about your local TZ.

See Also:
    Information about the time span of each time unit is available at
    [1]_.

References:
    .. [1] http://docs.scipy.org/doc/numpy/reference/arrays.datetime.html#datetime-units

This module is completely orthogonal to Viscid, so that it can be
ripped out and used more generally. Please note that Viscid is MIT
licensed, which requires attribution.

The MIT License (MIT)
Copyright (c) 2017 Kristofor Maynard

"""

from __future__ import print_function, division
from datetime import datetime, timedelta
from distutils.version import LooseVersion
try:
    from itertools import izip
except ImportError:
    izip = zip
import re
import sys

import numpy as np


if sys.version_info[0] == 3:
    string_types = str,
else:
    string_types = basestring,  # pylint: disable=undefined-variable


__all__ = ['PrecisionError',
           'as_datetime64', 'as_timedelta64', 'as_datetime', 'as_timedelta',
           'to_datetime64', 'to_timedelta64', 'to_datetime', 'to_timedelta',
           'as_isotime', 'to_isotime', 'format_time', 'format_datetime',
           'is_valid_datetime64', 'is_valid_timedelta64',
           'datetime_as_seconds', 'timedelta_as_seconds', 'time_as_seconds',
           'datetime64_as_years',
           'asarray_datetime64', 'linspace_datetime64',
           'round_time', 'regularize_time',
           'time_sum', 'time_diff',
           'is_datetime_like', 'is_timedelta_like', 'is_time_like']


_NP_TZ = LooseVersion(np.__version__) < LooseVersion('1.11')
TIME_UNITS = ('as', 'fs', 'ps', 'ns', 'us', 'ms', 's', 'm', 'h')
TIME_SCALE = (1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 60, 60)
DATE_UNITS = ('D', 'W', 'M', 'Y')
ORDERED_UNITS = list(TIME_UNITS) + list(DATE_UNITS)

DATETIME_BASE = "datetime64"
DELTA_BASE = "timedelta64"


class PrecisionError(ArithmeticError):
    """Used if conversion to a time unit truncates value to 0"""
    pass


# This class is for Python2.6 compatability since in 2.6, timedelta has
# no total_seconds method
class TimeDeltaCompat(timedelta):
    @staticmethod
    def __new__(*args, **kwargs):  # pylint: disable=no-method-argument
        if len(args) == 2 and isinstance(args[1], (timedelta, np.timedelta64)):
            return timedelta.__new__(args[0], args[1].days, args[1].seconds,
                                     args[1].microseconds)
        else:
            return timedelta.__new__(*args, **kwargs)

    def total_seconds(self):
        try:
            return super(TimeDeltaCompat, self).total_seconds()
        except AttributeError:
            return (self.microseconds + (self.seconds + self.days * 24 * 3600)
                    * 10**6) / 10**6


def _format_unit(unit, base=DATETIME_BASE):
    if unit:
        return "{0}[{1}]".format(base, unit)
    else:
        return base

def _as_dtype(t):
    if isinstance(t, np.dtype):
        return t
    else:
        return t.dtype

def _get_base_unit(t):
    name = _as_dtype(t).name
    base = name[:name.rfind('[')]
    unit = name[name.rfind('[') + 1:name.rfind(']')]
    return base, unit

def _get_unit(t):
    return _get_base_unit(t)[1]

def _is_datetime64(t):
    return _as_dtype(t).type == np.datetime64

def _is_dateunit(t):
    return _is_datetime64(t) and _get_unit(t) in DATE_UNITS

def _is_timeunit(t):
    return _is_datetime64(t) and _get_unit(t) in TIME_UNITS

def _as_datetime64_scalar(time, unit=None):
    unit_args = [unit] if unit else []

    if is_timedelta_like(time):
        scalar = as_timedelta64(time, unit=unit).astype(_format_unit(None))
    elif isinstance(time, string_types):
        try:
            time = as_isotime(time)
        except (TypeError, ValueError):
            pass  # Let ValueErrors happen in numpy constructors below

        if _is_timeunit(np.datetime64(time).dtype) and _NP_TZ:
            has_tz = bool(re.match(r".*([+-][0-9]{2,4}|Z)$", time))
            if not has_tz:
                time += 'Z'

        scalar = np.datetime64(time, *unit_args)
    elif unit_args and hasattr(time, 'astype'):
        scalar = time.astype(_format_unit(unit_args[0]))
    else:
        scalar = np.datetime64(time, *unit_args)
    return scalar

def _as_timedelta64_scalar(time, unit=None):
    unit_args = [unit] if unit else []
    flt_unit = unit if unit else 's'
    # turn 'H:M:S.ms', 'M:S.ms', 'S.ms' into floating point seconds
    if isinstance(time, string_types):# and ':' in time:
        time = [float(t) for t in time.lstrip('T').split(':')][::-1]
        if len(time) > 1 and unit is not None:
            raise ValueError("When giving time as a string, units are automatic")
        if len(time) > 3:
            raise ValueError("Timedelta as string only goes up to hours")
        t_flt = 0.0
        for factor, t in zip([1, 60, 60 * 60], time):
            t_flt += factor * t
        time = t_flt
        flt_unit = 's'
    # turn floating point time into integer with the correct unit
    if is_datetime_like(time):
        time = as_datetime64(time) - as_datetime64(np.timedelta64(0, 's'))
    elif isinstance(time, (np.timedelta64, timedelta)):
        time = np.timedelta64(time).astype(_format_unit(unit, base=DELTA_BASE))
    elif isinstance(time, (int, float, np.integer, np.floating)):
        orig_time, orig_flt_unit = time, flt_unit
        unit_idx = TIME_UNITS.index(flt_unit)
        while not np.isclose(time, int(np.round(time)), rtol=1e-4, atol=1e-18):
            if unit_idx <= 0:
                raise ValueError("Floating point time {0} [{1}] is too precise "
                                 "for any time unit?".format(orig_time, orig_flt_unit))
            unit_idx -= 1
            time *= TIME_SCALE[unit_idx]
            flt_unit = TIME_UNITS[unit_idx]
        time = np.timedelta64(int(np.round(time)), flt_unit)
        unit, unit_args = flt_unit, [flt_unit]
    return np.timedelta64(time, *unit_args)

def as_isotime(time):
    """Try to convert times in string format to ISO 8601

    Raises:
        TypeError: Elements are not strings
        ValueError: numpy.datetime64(time) fails
    """
    if isinstance(time, (list, tuple, np.ndarray)):
        scalar = False
    else:
        scalar = True
        time = [time]

    ret = [None] * len(time)
    for i, t in enumerate(time):
        if isinstance(t, string_types):
            t = t.strip().upper().lstrip('UT')
            if re.match(r"^[0-9]{2}([0-9]{2}:){3,5}[0-9]{1,2}(\.[0-9]*)?$", t):
                # Handle YYYY:MM:DD:hh:mm:ss.ms -> YYYY-MM-DDThh:mm:ss.ms
                #        YYYY:MM:DD:hh:mm:s.ms  -> YYYY-MM-DDThh:mm:s.ms
                #        YYYY:MM:DD:hh:mm:ss    -> YYYY-MM-DDThh:mm:ss
                #        YYYY:MM:DD:hh:mm       -> YYYY-MM-DDThh:mm
                #        YYYY:MM:DD:hh          -> YYYY-MM-DDThh
                # -- all this _tsp nonsense is to take care of s.ms; annoying
                _tsp = t.replace('.', ':').split(':')
                _tsp[0] = _tsp[0].zfill(4)
                _tsp[1:6] = [_s.zfill(2) for _s in _tsp[1:6]]
                t = ":".join(_tsp[:6])
                if len(_tsp) > 6:
                    t += "." + _tsp[6]
                # --
                ret[i] = t[:10].replace(':', '-') + 'T' + t[11:]

            elif re.match(r"^[0-9]{2}([0-9]{2}:){2}[0-9]{2}$", t):
                # Handle YYYY:MM:DD -> YYYY-MM-DD
                ret[i] = t.replace(':', '-')
            else:
                ret[i] = t

            try:
                np.datetime64(ret[i])
            except ValueError:
                raise
        else:
            raise TypeError("Can only turn strings to ISO 8601 time format "
                            "({0})".format(type(t)))

    if scalar:
        return ret[0]
    else:
        if isinstance(time, np.ndarray):
            return np.array(time, dtype=time.dtype)
        else:
            return ret

def as_datetime64(time, unit=None):
    """Convert to a Numpy datetime64 scalar or array

    Args:
        time: some python datetime or string in ISO 8601 format, could
            also be a sequence of these to return a Numpy ndarray
        unit (str): one of {Y,M,W,D,h,m,s,m,s,us,ns,ps,fs,as}

    Returns:
        np.datetime64[unit] or array with dtype np.datetime64[unit]
    """
    if isinstance(time, np.ndarray):
        time = time.astype(_format_unit(unit))
    elif isinstance(time, (list, tuple)):
        time = np.array([_as_datetime64_scalar(ti, unit=unit) for ti in time],
                        dtype=_format_unit(unit))
        if time.dtype.kind == 'f':
            unit = 'ns' if unit is None else unit
            time = time.astype(_format_unit(unit, base=DATETIME_BASE))
    else:
        time = _as_datetime64_scalar(time, unit=unit)
    return time

def as_timedelta64(time, unit=None):
    """Convert to a timedelta64 type


    Args:
        time (timedelta-like): an int/float/string/... to convert
        unit (None): This is the unit of the input, the result
            will be the most coarse unit that can store the time
    """
    if isinstance(time, (np.ndarray, list, tuple)):
        time = np.array([_as_timedelta64_scalar(ti, unit=unit) for ti in time])
        if time.dtype.kind == 'f':
            unit = 'ns' if unit is None else unit
            time = time.astype(_format_unit(unit, base=DELTA_BASE))
    else:
        time = _as_timedelta64_scalar(time, unit=unit)
    return time

def as_datetime(time, unit=None):
    """Convert time to a Numpy ndarray of datetime.datetime objects

    Args:
        time: some python datetime or string in ISO 8601 format, could
            also be a sequence of these to return a Numpy ndarray
        unit (str): one of {Y,M,W,D,h,m,s,m,s,us,ns,ps,fs,as}

    Returns:
        np.ndarray of native datetime.datetime objects (dtype = object)
    """
    try:
        dt64 = as_datetime64(time, unit=unit)
    except ValueError:
        dt64 = as_datetime64(as_timedelta64(time, unit=unit))
    return round_time(dt64, 'us').astype(datetime)

def as_timedelta(time, unit=None, allow0=True):
    """Convert time to a Numpy ndarray of datetime.datetime objects

    Note:
        Python timedelta objects are accurate up to microseconds

    Args:
        time: some python datetime or string in ISO 8601 format, could
            also be a sequence of these to return a Numpy ndarray
        unit (str): one of {Y,M,W,D,h,m,s,m,s,us,ns,ps,fs,as}
        allow0 (bool): If False, then raise PrecisionError if a value
            has been rounded to 0

    Returns:
        np.ndarray of native datetime.datetime objects (dtype = object)
    """
    time = as_timedelta64(time, unit=unit)
    ret = round_time(time, unit='us', allow0=allow0).astype(timedelta)
    if not isinstance(ret, np.ndarray) and not hasattr(ret, "total_seconds"):
        ret = TimeDeltaCompat(ret)
    elif isinstance(ret, np.ndarray) and not hasattr(ret[0], "total_seconds"):
        ret = np.array([TimeDeltaCompat(r) for r in ret])
    return ret

def asarray_datetime64(arr, unit=None, conservative=False):
    """If is_valid_datetime64, then return a datetime64 array

    Args:
        arr (sequence): something that can become an arary
        unit (str): one of {Y,M,W,D,h,m,s,m,s,us,ns,ps,fs,as}
        conservative (bool): If True, then only turn arr into a
            date-time array if it really looks like it
    """
    if conservative:
        if is_datetime_like(arr):
            return as_datetime64(arr, unit=unit)
        else:
            return np.asarray(arr)
    else:
        try:
            return as_datetime64(arr, unit=unit)
        except ValueError:
            return np.asarray(arr)

def linspace_datetime64(start, stop, n, endpoint=True, unit=None):
    """Make an evenly space ndarray from start to stop with n values"""
    start = as_datetime64(start, unit=unit)
    stop = as_datetime64(stop, unit=unit)
    start, stop = regularize_time([start, stop], most_precise=True)

    fltarr = np.linspace(start.astype('i8'), stop.astype('i8'), n,
                         endpoint=endpoint, dtype='f8')
    return np.round(fltarr, 0).astype('i8').astype(start.dtype)

def _most_precise_t_unit(tlst):
    """Find the most precise time unit from the bunch

    Args:
        tlst: datetime64 or timedelta64 instances

    Returns:
        str: unit of the most precise time
    """
    units = [_get_base_unit(t)[1] for t in tlst]
    unit_idx = [ORDERED_UNITS.index(u) for u in units]
    return units[np.argmin(unit_idx)]

def _adjust_t_unit(t, unit, cfunc=None, allow0=True):
    """adjust the unit of t using cfunc

    Args:
        t (datetime64, timedelta64): time to convert
        unit: target unit
        cfunc (callable): one of `as_datetime64` or `as_timedelta64`
        allow0 (bool): If False, then raise PrecisionError if a value
            has been truncated to 0

    Raises:
        OverflowError: if all elements of tlst can't fit in the same
            unit
        PrecisionError: if rounding a time truncated it to 0 and not
            `allow0`
    """
    orig_base, orig_unit = _get_base_unit(t)

    if cfunc is None:
        cfunc_lookup = {'datetime64': as_datetime64,
                        'timedelta64': as_timedelta64}
        cfunc = cfunc_lookup[orig_base]

    if orig_unit == unit:
        t1 = t
    elif ORDERED_UNITS.index(unit) < ORDERED_UNITS.index(orig_unit):
        # we want a more precise unit... raise an OverflowError if the
        # new unit can not store the most coarse part of t
        t1 = cfunc(t, unit=unit)
        # converting back to orig_unit and checking t == t2 effectively
        # checks to make sure we haven't overflowed into the sign bit
        # since this isn't checked by Numpy internally. if the conversion
        # overflowed a 64-bit int completely, an OverflowError has already
        # been raised
        t2 = cfunc(t1, unit=orig_unit)
        try:
            if not np.all(t == t2):
                raise OverflowError("Time {0} could not be refined to unit '{1}' "
                                    "because it overflowed into the sign bit ({2})."
                                    "".format(str(t), unit, str(t1)))
        except ValueError:
            import viscid; viscid.interact()
            raise
    else:
        # we want a less precise unit, i.e., round t to the new unit... raise
        # a PrecisionError if t was rounded to 0
        t1 = cfunc(t, unit=unit)
        if not allow0 and t1.astype('i8') == 0 and t.astype('i8') != 0:
            raise PrecisionError("The time {0} was truncated to 0 when "
                                 "rounded to the nearest '{1}'"
                                 "".format(str(t), unit))
    return t1

def round_time(tlst, unit, allow0=True):
    """Round a time or list of times to minimum level of coarseness

    Note:
        * When rounding, some values might be rounded to 0. If you
          rather raise a PrecisionError, then give `allow0=False`.

    Args:
        tlst (timelike, list): single or list of datetime64 or
            timedelta64
        unit (str): units of result will be at least as coarse as
            this unit
        allow0 (bool): If False, then raise PrecisionError if a value
            has been truncated to 0

    Returns:
        timelike or list: `tlst` rounded to a unit at least as coarse
            as `unit`
    """
    cfunc_lookup = {'datetime64': as_datetime64,
                    'timedelta64': as_timedelta64}
    if not isinstance(tlst, (list, tuple, np.ndarray)):
        tlst = [tlst]
        single_val = True
    else:
        single_val = False
    bases_units = [_get_base_unit(t) for t in tlst]
    bases = [bu[0] for bu in bases_units]
    units = [bu[1] for bu in bases_units]
    unit_idxs = [ORDERED_UNITS.index(u) for u in units]
    unit0_idx = ORDERED_UNITS.index(unit)
    ret = []
    for t, base, unit_idx in izip(tlst, bases, unit_idxs):
        cfunc = cfunc_lookup[base]
        if unit_idx >= unit0_idx:
            ret.append(t)
        else:
            ret.append(_adjust_t_unit(t, unit, cfunc=cfunc, allow0=allow0))

    if single_val:
        return ret[0]
    else:
        if isinstance(tlst, np.ndarray):
            return np.asarray(ret)
        else:
            return ret

def regularize_time(tlst, unit=None, most_precise=False, allow_rounding=True,
                    allow0=True):
    """Convert a list of times to a common unit

    Notes:
        * If some times are too fine to fit in the same unit as the
          rest of the times, then they will be rounded to a more
          coarse unit. If you rather raise an OverflowError, then give
          `allow_rounding=False`.
        * When rounding, some values might be rounded to 0. If you
          rather raise a PrecisionError, then give `allow0=False`.

    Args:
        tlst (timelike, list): single or list of datetime64 or
            timedelta64
        unit (str): If given, regularize all times to this unit,
            otherwise, regularize them to the most precise of the
            bunch
        most_precise (bool): If True, then convert all times to the
            most precise unit that fits all the times
        allow_rounding (bool): if any time is too small to be
            represented in the desired unit, then use a more coarse
            unit and round values so that everything fits
        allow0 (bool): If False, then raise PrecisionError if a value
            has been rounded to 0

    Returns:
        timelike or list: single or list of times all in the same unit
    """
    cfunc_lookup = {'datetime64': as_datetime64,
                    'timedelta64': as_timedelta64}
    if not isinstance(tlst, (list, tuple, np.ndarray)):
        tlst = [tlst]
        single_val = True
    else:
        single_val = False

    if unit is None:
        unit = _most_precise_t_unit(tlst)

    bases_units = [_get_base_unit(t) for t in tlst]
    bases = [bu[0] for bu in bases_units]
    cfuncs = [cfunc_lookup[b] for b in bases]

    # round values to successively more coarse units until we get to
    # a unit that can contain all the times in our list
    for u in ORDERED_UNITS[ORDERED_UNITS.index(unit):]:
        ret = []
        try:
            for t, cfunc in izip(tlst, cfuncs):
                ret.append(_adjust_t_unit(t, u, cfunc, allow0=allow0))
        except OverflowError:
            if not allow_rounding:
                raise
        else:
            unit = u
            break

    # if we want the most precise unit that fits everything, then keep
    # refining the unit until we get an OverflowError
    if most_precise:
        for u in reversed(ORDERED_UNITS[:ORDERED_UNITS.index(unit)]):
            try:
                nxt = []
                for t, cfunc in izip(ret, cfuncs):
                    nxt.append(_adjust_t_unit(t, u, cfunc, allow0=allow0))
            except OverflowError:
                break
            else:
                ret = nxt

    if single_val:
        return ret[0]
    else:
        if isinstance(tlst, np.ndarray):
            return np.asarray(ret)
        else:
            return ret

def time_sum(t0, tdelta, unit=None, most_precise=False, allow_rounding=True,
             allow0=True):
    """Add timedelta64 to datetime64 at highest precision w/o overflow

    Notes:
        * If `allow_rounding`, then the result may not be in `unit`. If
          you rather raise an OverflowError, give `allow_rounding=False`
        * If t0 can not be represented using the same units as tdelta,
          then tdelta could be rounded to 0. If you rather raise a
          PrecisionError, then give `allow0=False`

    Args:
        t0 (datetime64): starting date
        tdelta (timedelta64): timedelta to add
        unit (str): If given, regularize all times to this unit,
            otherwise, regularize them to the most precise of the
            bunch
        most_precise (bool): If True, then convert all times to the
            most precise unit that fits all the times
        allow_rounding (bool): if tdelta is too small to be represented
            in the same unit as t0, then round it to the finest unit
            that fits both t0 and tdelta
        allow0 (bool): If False, and a value is rounded to 0 in a given
            unit, then raise a PrecisionError

    Returns:
        datetime64: t0 + tdelta
    """
    t0 = as_datetime64(t0)
    tdelta = as_timedelta64(tdelta)
    t0, tdelta = regularize_time([t0, tdelta], unit=unit,
                                 most_precise=most_precise,
                                 allow_rounding=allow_rounding,
                                 allow0=allow0)
    return t0 + tdelta

def time_diff(t1, t2, unit=None, most_precise=False):
    """Diff two datetime64s at highest precision w/o overflow

    Note:
        If `allow_rounding`, then the result may not be in `unit`. If
        you rather raise an OverflowError, give `allow_rounding=False`

    Args:
        t1 (datetime64): `t1` for `t1 - t2`
        t2 (datetime64): `t2` for `t1 - t2`
        unit (str): If given, regularize all times to this unit,
            otherwise, regularize them to the most precise of the
            bunch
        most_precise (bool): If True, then convert all times to the
            most precise unit that fits all the times

    Returns:
        timedelta64: t1 - t2
    """
    t1 = as_datetime64(t1)
    t2 = as_datetime64(t2)
    t1, t2 = regularize_time([t1, t2], unit=unit, most_precise=most_precise)
    return t1 - t2

def is_valid_datetime64(arr, unit=None):
    """Returns True iff arr can be made into a datetime64 array"""
    try:
        as_datetime64(arr, unit=unit)
        return True
    except ValueError:
        return False

def is_valid_timedelta64(arr, unit=None):
    """Returns True iff arr can be made into a timedelta64 array"""
    try:
        as_timedelta64(arr, unit=unit)
        return True
    except ValueError:
        return False

def datetime_as_seconds(a, decimals=0, unit=None):
    """round datetime a to the nearest decimals seconds"""
    a_as_dt64 = as_datetime64(a, unit=unit)
    _epoch = regularize_time(a_as_dt64, unit='s')
    frac = time_diff(a_as_dt64, _epoch) / np.timedelta64(1, 's')
    rounded = np.round(frac, decimals)
    return as_datetime64(_epoch + as_timedelta64(rounded, unit='s'))

def timedelta_as_seconds(a, decimals=0, unit=None):
    """round timedelta a to the nearest decimals seconds

    Note:
        works for 'fs', but not 'as'
    """
    a_td64 = as_timedelta64(a, unit=unit)
    rounded = np.round(a_td64 / as_timedelta64(1, 's'), decimals)
    return as_timedelta64(rounded, unit='s')

def time_as_seconds(a, decimals=0, unit=None):
    """round a to the nearest decimal seconds"""
    if is_datetime_like(a):
        return datetime_as_seconds(a, decimals=decimals, unit=unit)
    elif is_timedelta_like(a, conservative=True):
        return timedelta_as_seconds(a, decimals=decimals, unit=unit)
    else:
        return np.round(a, decimals=decimals)

def format_datetime(time, fmt="%Y-%m-%d %H:%M:%S.%.02f"):
    """Shortcut to :py:func:`format_time` for a datetime format"""
    return format_time(time, fmt=fmt)

def format_time(time, fmt='.02f', basetime=None):
    """Format time as a string

    Args:
        t (float): time
        style (str): for this method, can be::

              -----------------------   -------   ----------------------------
              style                     time      string
              -----------------------   -------   ----------------------------
              'hms'                     90015.0   "25:00:15"
              'hmss'                    90015.0   "25:00:15 (090015)"
              'dhms'                      900.0   "0 days 00:15:00"
              'dhmss'                     900.0   "0 days 00:15:00 (000900)"
              '.02f'                      900.0   '900.00'
              '%Y-%m-%d %H:%M:%S'         900.0   '1970-01-01 00:15:00'
              '%Y-%m-%d %H:%M:%S.%1f'     900.0   '1970-01-01 00:15:00.0'
              -----------------------   -------   ----------------------------

            Note that the last one can involve any formatting strings
            understood by datetime.strftime
        basetime (np.datetime64): if formatting just number of seconds
            from something like ".02f", then use this time as 0 seconds

    Returns:
        str
    """
    dttime = as_datetime(time)
    ret = ""

    if basetime is None:
        basetime = as_datetime64(0.0)

    if fmt.lower() == 'ut':
        fmt = '%Y-%m-%d %H:%M:%S'

    if fmt in ('dhms', 'dhmss', 'hms', 'hmss'):
        # These are special time-style formatters
        if fmt.startswith('d'):
            days = int(as_timedelta64(dttime) / np.timedelta64(1, 'D'))
            if days == 1:
                days_str = '{0} day'.format(days)
            else:
                days_str = '{0} days '.format(days)
        else:
            days_str = ''
        ret = datetime.strftime(dttime, days_str + '%H:%M:%S')
        if fmt.endswith('ss'):
            _tt = time_diff(dttime, basetime) / np.timedelta64(1, 's')
            ret += " ({0:06d})".format(int(_tt))
    elif '%' not in fmt:
        # if there's no % symbol, then it's probably not a strftime format,
        # so use fmt as normal string formatting of total_seconds
        _tt = (as_datetime64(time) - basetime) / np.timedelta64(1, 's')
        ret = "{0:{1}}".format(_tt, fmt.strip())
    else:
        if not fmt:
            msec_fmt = ['1']
            fmt = "%Y-%m-%d %H:%M:%S.%f"
        else:
            msec_fmt = re.findall(r"%\.?([0-9]*)f", fmt)
            fmt = re.sub(r"%\.?([0-9]*)f", "%f", fmt)

        tstr = datetime.strftime(dttime, fmt)

        # now go back and for any %f -> [0-9]{6}, reformat the precision
        it = list(izip(msec_fmt, re.finditer("[0-9]{6}", tstr)))
        for ffmt, m in reversed(it):
            a, b = m.span()
            val = float("0." + tstr[a:b])
            ifmt = int(ffmt) if len(ffmt) > 0 else 6
            f = "{0:0.{1}f}".format(val, ifmt)[2:]
            tstr = tstr[:a] + f + tstr[b:]
        ret = tstr
    return ret

def _check_like(val, _np_types, _native_types, check_str=None):  # pylint: disable=too-many-return-statements
    """
    Checks the follwing:
      - if val is instance of _np_types or _native_types
      - if val is a list or ndarray of _np_types or _native_types
      - if val is a string or list of strings that can be parsed by check_str
    Does not check:
      - if val is an ndarray of strings that can be parsed by check_str
    """
    _all_types = _np_types + _native_types

    if isinstance(val, _all_types):
        return True
    elif isinstance(val, string_types):
        return check_str and check_str(val)
    elif isinstance(val, (list, tuple)):
        for v in val:
            if isinstance(v, string_types):
                if check_str and check_str(v):
                    continue
            if not isinstance(v, _all_types):
                return False
        return True
    elif hasattr(val, 'dtype'):
        if val.dtype == np.object:
            return all(isinstance(v, _native_types) for v in val)
        else:
            return val.dtype.type in _np_types
    else:
        return False

def datetime64_as_years(time):
    """Get time as floating point years since the year 0"""
    time = as_datetime64(time)
    epoch_year = 1970
    epoch = as_datetime64("{0}-01-01T00:00:00.0".format(epoch_year))
    tdelta = time_diff(time, epoch, most_precise=True)
    years = tdelta / np.timedelta64(1, 'D') / 365.242 + epoch_year
    return years

def is_datetime_like(val, conservative=False):  # pylint: disable=unused-argument
    """Returns True iff val is datetime-like"""
    if conservative and val is None:
        return False
    if conservative:
        try:
            float(val)
            return False
        except (ValueError, TypeError):
            pass
    return _check_like(val, (np.datetime64, ), (datetime, ),
                       is_valid_datetime64)

def is_timedelta_like(val, conservative=False):
    """Returns True iff val is timedelta-like"""
    if conservative:
        if val is None:
            return False
        if isinstance(val, string_types):
            try:
                float(val)
                return False
            except (ValueError, TypeError):
                pass
        return _check_like(val, (np.timedelta64, ), (timedelta, ),
                           is_valid_timedelta64)
    else:
        return _check_like(val, (np.timedelta64, np.floating, np.integer),
                           (timedelta, float, int), is_valid_timedelta64)

def is_time_like(val, conservative=False):
    """Returns True iff val is datetime-like or timedelta-like"""
    return (is_datetime_like(val, conservative=conservative) or
            is_timedelta_like(val, conservative=conservative))

to_datetime64 = as_datetime64
to_timedelta64 = as_timedelta64
to_datetime = as_datetime
to_timedelta = as_timedelta
to_isotime = as_isotime


def _main():
    verb = True

    d0 = as_datetime64('2010-06-21')
    d1 = as_datetime64('2014-12-15T03:00:00.0003')
    d2 = as_datetime64('1970-01-01', 'as')
    t0 = as_timedelta64(60, 'm')
    t1 = as_timedelta64(121, 'us')
    t2 = as_timedelta64(1536, 'as')

    l0 = [d0, d1, t0, t1, t2]

    if verb:
        print("l0", l0, "\n")

    #
    # TEST `round_time` and `regularize_time`
    #

    l1 = regularize_time(l0, unit='us')
    if verb:
        print("l1", l1, "\n")

    l2 = round_time(l0, 'us')
    if verb:
        print("l1", l2, "\n")

    l3 = round_time(l1, 's')
    if verb:
        print("l3", l3, "\n")

    l4 = round_time(l0, 'fs', allow0=False)
    if verb:
        print("l4", l4, "\n")

    assert l1 == l2
    assert l1 != l3

    try:
        _ = regularize_time(l0, unit='us', allow0=False)
    except PrecisionError:
        pass
    else:
        assert 0, "rounding 1536 atto secs -> us should have caused an error"

    try:
        _ = regularize_time(l0, allow_rounding=False)
    except OverflowError:
        pass
    else:
        assert 0, "2010-06-21 should not be representable in atto secs"

    try:
        _ = round_time(l0, 's', allow0=False)
    except PrecisionError:
        pass
    else:
        assert 0, "rounding 1536 atto secs -> secs should have caused an error"

    #
    # TEST `time_sum`
    #

    print(d0, "+", t1, "=", time_sum(d0, t1))
    print(d0, "+", t2, "=", time_sum(d0, t2))

    try:
        time_sum(d0, t2, allow0=False)
    except PrecisionError:
        pass
    else:
        assert 0, "rounding 1536 atto secs -> us should have caused an error"

    try:
        time_sum(d0, t2, allow_rounding=False)
    except OverflowError:
        pass
    else:
        assert 0, "2010-06-21 should not be representable in atto secs"

    #
    # TEST `time_diff`
    #

    print(d0, "-", d1, "=", time_diff(d0, d1))
    print(d0, "-", d1, "=", time_diff(d0, d1, most_precise=True))
    print(d0, "-", d1, "=", time_diff(d0, d1, unit='s'))
    print(d0, "-", d2, "=", time_diff(d0, d2))
    print(d0, "-", d2, "=", time_diff(d0, d2, unit='s'))
    print(d0, "-", d2, "=", time_diff(d0, d2, unit='Y'))

    #
    # TEST linspace_datetime64
    #
    lst = linspace_datetime64(as_datetime64('1930-03-04'),
                               as_datetime64('2010-02-14T12:30:00'),
                               10)
    print(lst, lst.dtype)

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
