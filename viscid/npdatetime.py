"""This is a shim to use numpy datetime in a standard way

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
"""

# info about time span of each time unit is available at:
# http://docs.scipy.org/doc/numpy/reference/arrays.datetime.html#datetime-units

from __future__ import print_function, division
from datetime import datetime, timedelta
from distutils.version import LooseVersion
import re

import numpy as np
import viscid


__all__ = ['as_datetime64', 'as_timedelta64', 'as_datetime', 'as_timedelta',
           'to_datetime64', 'to_timedelta64', 'to_datetime', 'to_timedelta',
           'as_isotime', 'to_isotime',
           'is_valid_datetime64', 'is_valid_timedelta64',
           'asarray_datetime64', 'linspace_datetime64',
           'is_datetime_like', 'is_timedelta_like', 'is_time_like']

_NP_TZ = LooseVersion(np.__version__) < LooseVersion('1.11')
TIME_UNITS = ('as', 'fs', 'ps', 'ns', 'us', 'ms', 's', 'm', 'h')
DATE_UNITS = ('D', 'W', 'M', 'Y')

DATETIME_BASE = "datetime64"
DELTA_BASE = "timedelta64"


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

    if isinstance(time, viscid.string_types):
        try:
            time = as_isotime(time)
        except (TypeError, ValueError):
            pass  # Let ValueErrors happen in numpy constructors below

        if _is_timeunit(np.datetime64(time).dtype) and _NP_TZ:
            has_tz = bool(re.match(r".*([+-][0-9]{2,4}|Z)$", time))
            if not has_tz:
                time += 'Z'

        scalar = np.datetime64(time, *unit_args)
    elif isinstance(time, np.timedelta64):
        scalar = np.array([time]).astype(_format_unit(unit))[0]
    elif isinstance(time, timedelta):
        scalar = np.array([time], dtype='timedelta64').astype(_format_unit(unit))[0]
    else:
        scalar = np.datetime64(time, *unit_args)
    return scalar

def _as_timedelta64_scalar(time, unit=None):
    unit_args = [unit] if unit else []
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
        if isinstance(t, viscid.string_types):
            t = t.strip()
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
    if isinstance(time, np.ndarray):
        time = time.astype(_format_unit(unit, base=DELTA_BASE))
    elif isinstance(time, (list, tuple)):
        time = np.array([_as_timedelta64_scalar(ti, unit=unit) for ti in time],
                        dtype=_format_unit(unit, base=DELTA_BASE))
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
    return as_datetime64(time, unit=unit).astype(datetime)

def as_timedelta(time, unit=None):
    """Convert time to a Numpy ndarray of datetime.datetime objects

    Args:
        time: some python datetime or string in ISO 8601 format, could
            also be a sequence of these to return a Numpy ndarray
        unit (str): one of {Y,M,W,D,h,m,s,m,s,us,ns,ps,fs,as}

    Returns:
        np.ndarray of native datetime.datetime objects (dtype = object)
    """
    return as_timedelta64(time, unit=unit).astype(timedelta)

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

def is_valid_datetime64(arr, unit=False):
    """Returns True iff arr can be made into a datetime64 array"""
    try:
        as_datetime64(arr, unit=unit)
        return True
    except ValueError:
        return False

def is_valid_timedelta64(arr, unit=False):
    """Returns True iff arr can be made into a datetime64 array"""
    try:
        as_timedelta64(arr, unit=unit)
        return True
    except ValueError:
        return False

def _check_like(val, _np_types, _native_types, check_str=None):  # pylint: disable=too-many-return-statements
    _all_types = _np_types + _native_types

    if isinstance(val, _all_types):
        return True
    elif isinstance(val, viscid.string_types):
        return check_str and check_str(val)
    elif isinstance(val, (list, tuple)):
        for v in val:
            if isinstance(v, viscid.string_types):
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

def is_datetime_like(val):
    """Returns True iff val is datetime-like"""
    return _check_like(val, (np.datetime64, ), (datetime, ),
                       is_valid_datetime64)

def is_timedelta_like(val):
    """Returns True iff val is timedelta-like"""
    return _check_like(val, (np.timedelta64, ), (timedelta, ),
                       is_valid_timedelta64)

def is_time_like(val):
    """Returns True iff val is datetime-like or timedelta-like"""
    return is_datetime_like(val) or is_timedelta_like(val)

to_datetime64 = as_datetime64
to_timedelta64 = as_timedelta64
to_datetime = as_datetime
to_timedelta = as_timedelta
to_isotime = as_isotime

##
## EOF
##
