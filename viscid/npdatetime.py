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
Copyright (c) 2016 Kristofor Maynard

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


__all__ = ['as_datetime64', 'as_timedelta64', 'as_datetime', 'as_timedelta',
           'to_datetime64', 'to_timedelta64', 'to_datetime', 'to_timedelta',
           'as_isotime', 'to_isotime', 'format_time', 'format_datetime',
           'is_valid_datetime64', 'is_valid_timedelta64',
           'round_datetime', 'round_timedelta', 'round_time',
           'asarray_datetime64', 'linspace_datetime64', 'most_precise_tdiff',
           'datetime64_as_years',
           'is_datetime_like', 'is_timedelta_like',
           'is_time_like']


_NP_TZ = LooseVersion(np.__version__) < LooseVersion('1.11')
TIME_UNITS = ('as', 'fs', 'ps', 'ns', 'us', 'ms', 's', 'm', 'h')
TIME_SCALE = (1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 60, 60)
DATE_UNITS = ('D', 'W', 'M', 'Y')


DATETIME_BASE = "datetime64"
DELTA_BASE = "timedelta64"


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

def _get_unit(dtype):
    return dtype.name[len('datetime64'):][1:-1]

def _is_datetime64(dtype):
    return dtype.type == np.datetime64

def _is_dateunit(dtype):
    return _is_datetime64(dtype) and _get_unit(dtype) in DATE_UNITS

def _is_timeunit(dtype):
    return _is_datetime64(dtype) and _get_unit(dtype) in TIME_UNITS

def _most_precise_diff(dt1, dt2):
    """return dt1 - dt2 with the best precision it can and still be
    able to represent both dt1 and dt2
    """
    # try to represent the diff with various precision and stop when
    # we reach a unit that doesn't overflow the timedelta unit

    diff = dt1 - dt2
    unit_idx = -1
    for i, unit in enumerate(TIME_UNITS):
        try:
            diff_tmp = diff.astype(_format_unit(unit, base=DELTA_BASE))
            # overflowing into the sign bit doesn't raise an OverflowError,
            # so we do this the explicitly
            if diff == diff_tmp.astype(diff.dtype):
                diff = diff_tmp
                unit_idx = i
                break
        except OverflowError:
            pass

    if unit_idx < 0:
        raise OverflowError("The interval {0} - {1} can not be represented as "
                            "a timedelta with time units".format(dt1, dt2))

    # now make the resolution more coarse until it can represent both
    # dt1 and dt2 as an absolute time in the same unit
    best_unit = None
    ss = (dt1, dt2)
    for unit in TIME_UNITS[unit_idx:]:
        diff = diff.astype(_format_unit(unit, base=DELTA_BASE))
        if all(t.astype(datetime) == (t + diff - diff).astype(datetime) for t in ss):
            best_unit = unit
            break

    if best_unit is None:
        raise OverflowError("Could not find best unit for difference {0} - {1}"
                            "".format(dt1, dt2))

    return best_unit, diff.astype(_format_unit(unit, base=DELTA_BASE))

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
    else:
        scalar = np.datetime64(time, *unit_args)
    return scalar

def _as_timedelta64_scalar(time, unit=None):
    unit_args = [unit] if unit else []
    flt_unit = unit if unit else 's'
    # turn 'H:M:S.ms', 'M:S.ms', 'S.ms' into floating point seconds
    if isinstance(time, string_types):# and ':' in time:
        time = [float(t) for t in time.split(':')][::-1]
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
        try:
            eps = np.finfo(time).resolution
        except ValueError:
            eps = 1e-15
        while not np.isclose(time, int(time), rtol=eps, atol=1e-15):
            if unit_idx <= 0:
                raise ValueError("Floating point time {0} [{1}] is too precise "
                                 "for any time unit?".format(orig_time, orig_flt_unit))
            unit_idx -= 1
            time *= TIME_SCALE[unit_idx]
            flt_unit = TIME_UNITS[unit_idx]
        time = np.timedelta64(int(time), flt_unit)
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
            if re.match(r"^[0-9]{2}([0-9]{2}:){3,5}[0-9]{2}(\.[0-9]*)?$", t):
                # Handle YYYY:MM:DD:hh:mm:ss.ms -> YYYY-MM-DDThh:mm:ss.ms
                #        YYYY:MM:DD:hh:mm:ss    -> YYYY-MM-DDThh:mm:ss
                #        YYYY:MM:DD:hh:mm       -> YYYY-MM-DDThh:mm
                #        YYYY:MM:DD:hh          -> YYYY-MM-DDThh
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
    # if isinstance(time, np.ndarray):
    #     time = time.astype(_format_unit(None, base=DELTA_BASE))
    if isinstance(time, (np.ndarray, list, tuple)):
        time = np.array([_as_timedelta64_scalar(ti, unit=unit) for ti in time])
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
        return as_datetime64(time, unit=unit).astype(datetime)
    except ValueError:
        return as_datetime64(as_timedelta64(time, unit=unit)).astype(datetime)

def as_timedelta(time, unit=None):
    """Convert time to a Numpy ndarray of datetime.datetime objects

    Args:
        time: some python datetime or string in ISO 8601 format, could
            also be a sequence of these to return a Numpy ndarray
        unit (str): one of {Y,M,W,D,h,m,s,m,s,us,ns,ps,fs,as}

    Returns:
        np.ndarray of native datetime.datetime objects (dtype = object)
    """
    # FIXME: this is known not to work in all cases, for instance when
    # time is in units of ns
    ret = as_timedelta64(time, unit=unit).astype(timedelta)
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

    # make start - stop as precise as you can
    if unit:
        diff = stop - start
    else:
        unit, diff = _most_precise_diff(stop, start)
        start = start.astype(_format_unit(unit))
        stop = stop.astype(_format_unit(unit))

    if endpoint:
        dx = diff / max(n - 1, 1)
    else:
        dx = diff / max(n, 1)

    arr = np.empty((n,), dtype=start.dtype)
    arr[:] = start + np.arange(n) * dx
    return arr

def most_precise_tdiff(t1, t2):
    """return t1 - t2 with the best precision it can and still be
    able to represent both dt1 and dt2
    """
    return _most_precise_diff(as_datetime64(t1), as_datetime64(t2))

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

def round_datetime(a, decimals=0):
    """round datetime a to the nearest dicimals seconds"""
    a_as_dt64 = as_datetime64(a)
    _epoch = as_datetime64(a_as_dt64, unit='s')
    frac = (a_as_dt64 - _epoch) / np.timedelta64(1, 's')
    rounded = np.round(frac, decimals)
    return as_datetime64(_epoch + as_timedelta64(rounded, unit='s'))

def round_timedelta(a, decimals=0):
    """round timedelta a to the nearest dicimals seconds"""
    rounded = np.round(as_timedelta(a).total_seconds(), decimals)
    return as_timedelta64(rounded, unit='s')

def round_time(a, decimals=0):
    """round a to the nearest dicimals seconds"""
    if is_datetime_like(a):
        return round_datetime(a, decimals=decimals)
    elif is_timedelta_like(a, conservative=True):
        return round_timedelta(a, decimals=decimals)
    else:
        return np.round(a, decimals=decimals)

def format_datetime(time, fmt="%Y-%m-%d %H:%M:%S.%.02f"):
    """Shortcut to :py:func:`format_time` for a datetime format"""
    return format_time(time, fmt=fmt)

def format_time(time, fmt='.02f'):
    """Format time as a string

    Args:
        t (float): time
        style (str): for this method, can be::
          -------------------   -------   ------------------------------
          style                   time    string
          -------------------   -------   ------------------------------
          'hms'                 90015.0   "25:00:15.000"
          'hmss'                90015.0   "25:00:15.000 (090015)"
          'dhms'                  900.0   "0 days 00:15:00.000"
          'dhmss'                 900.0   "0 days 00:15:00.000 (000900)"
          '.02f'                  900.0   '900.00'
          '%Y-%m-%d %H:%M:%S'     900.0   '1970-01-01 00:15:00'
          -------------------   -------   ------------------------------

          Note that the last one can involve any formatting strings
          understood by datetime.strftime

    Returns:
        str
    """
    time = as_datetime(time)
    ret = ""

    if fmt.lower() == 'ut':
        fmt = '%Y-%m-%d %H:%M:%S'

    if fmt in ('dhms', 'dhmss', 'hms', 'hmss'):
        # These are special time-style formatters
        if fmt.startswith('d'):
            days = int(as_timedelta64(time) / np.timedelta64(1, 'D'))
            if days == 1:
                days_str = '{0} day'.format(days)
            else:
                days_str = '{0} days '.format(days)
        else:
            days_str = ''
        ret = datetime.strftime(time, days_str + '%H:%M:%S')
        if fmt.endswith('ss'):
            ret += " ({0:06d})".format(int(as_timedelta(time).total_seconds()))
    elif '%' not in fmt:
        # if there's no % symbol, then it's probably not a strftime format,
        # so use fmt as normal string formatting of total_seconds
        ret = "{0:{1}}".format(as_timedelta(time).total_seconds(), fmt.strip())
    else:
        if not fmt:
            msec_fmt = ['1']
            fmt = "%Y-%m-%d %H:%M:%S.%f"
        else:
            msec_fmt = re.findall(r"%\.?([0-9]*)f", fmt)
            fmt = re.sub(r"%\.?([0-9]*)f", "%f", fmt)

        tstr = datetime.strftime(time, fmt)

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
    """Get time as floating point years since the year 0000"""
    time = as_datetime64(time)
    _epoch_year = 1900
    _epoch = as_datetime64("{0}-01-01T00:00:00.0".format(_epoch_year))
    _epoch = _epoch.astype(time.dtype)
    tdelta = as_timedelta(time - _epoch).total_seconds()
    years = tdelta / (365.242 * 24 * 3600) + _epoch_year
    return years

def is_datetime_like(val, conservative=False):  # pylint: disable=unused-argument
    """Returns True iff val is datetime-like"""
    if conservative and val is None:
        return False
    if conservative:
        try:
            int(val)
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
        try:
            int(val)
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

##
## EOF
##
