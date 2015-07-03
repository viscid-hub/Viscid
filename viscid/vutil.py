#!/usr/bin/env python

from __future__ import print_function, division

from datetime import datetime
import fnmatch
from glob import glob
from itertools import count
import logging
from operator import itemgetter
import os.path
import re
import subprocess as sub
import sys
from timeit import default_timer as time

from viscid import logger
from viscid.compat import izip, string_types

import numpy as np


tree_prefix = ".   "

def find_field(vfile, fld_name_lst):
    """ convenience function to get a field that could be called many things
    returns the first fld_name in the list that is in the file """
    for fld_name in fld_name_lst:
        if fld_name in vfile:
            return vfile[fld_name]
    raise KeyError("file {0} contains none of {1}".format(vfile, fld_name_lst))

def split_floats(arg_str):
    return [float(s) for s in arg_str.split(',')]

def add_animate_arguments(parser):
    """ add common options for animating, you may want to make sure parser was
    constructed with conflict_handler='resolve' """
    anim = parser.add_argument_group("Options for creating animations")
    anim.add_argument("-a", "--animate", default=None,
                      help="animate results")
    anim.add_argument("--prefix", default=None,
                        help="Prefix of the output image filenames")
    anim.add_argument('-r', '--rate', dest='framerate', type=int, default=5,
                      help="animation frame rate (default 5).")
    anim.add_argument('--qscale', dest='qscale', default='2',
                      help="animation quality flag (default 2).")
    anim.add_argument('-k', dest='keep', action='store_true',
                      help="keep temporary files.")
    return parser

def add_mpl_output_arguments(parser):
    """ add common options for tuning matplotlib output, you may want to make
    sure parser was constructed with conflict_handler='resolve' """
    mplargs = parser.add_argument_group("Options for tuning matplotlib")
    mplargs.add_argument("-s", "--size", dest="plot_size", type=split_floats,
                         default=None, help="size of mpl plot (inches)")
    mplargs.add_argument("--dpi", dest="dpi", type=float, default=None,
                         help="dpi of plot")
    parser.add_argument("--prefix", default=None,
                        help="Prefix of the output image filenames")
    parser.add_argument("--format", "-f", default="png",
                        help="output format, as in 'png'|'pdf'|...")
    parser.add_argument('-w', '--show', dest='show', action="store_true",
                        help="show plots with plt.show()")
    return parser

def common_argparse(parser, default_verb=0, logging_fmt=None):
    """ add some common verbosity stuff to argparse, parse the
    command line args, and setup the logging levels
    parser should be an ArgumentParser instance, and kwargs
    should be options that get passed to logger.basicConfig
    returns the args namespace  """
    general = parser.add_argument_group("Viscid general options")
    general.add_argument("--log", action="store", type=str, default=None,
                         help="Logging level (overrides verbosity)")
    general.add_argument("-v", action="count", default=default_verb,
                         help="increase verbosity")
    general.add_argument("-q", action="count", default=0,
                         help="decrease verbosity")
    args = parser.parse_args()

    # setup the logging level
    if args.log is not None:
        logger.setLevel(getattr(logging, args.log.upper()))
    else:
        # default = 30 WARNING
        verb = args.v - args.q
        logger.setLevel(int(30 - 10 * verb))

    # the 0th handler going to stderr should always be setup
    if not logging_fmt:
        logging_fmt = "(%(levelname)s): %(message)s"
    logger.handlers[0].setFormatter(logging.Formatter(logging_fmt))

    return args

def make_animation(args, program="ffmpeg"):
    """ make animation by calling program (only ffmpeg works for now) using
    args, which is a namespace filled by the argparse options from
    add_animate_arguments. Plots are expected to be named
    ${args.prefix}_000001.png where the number is in order from 1 up """
    if args.animate:
        if program == "ffmpeg":
            sub.Popen("ffmpeg -r {0} -i {2}_%06d.png -pix_fmt yuv420p "
                      "-qscale {1} {3}".format(args.framerate, args.qscale,
                      args.prefix, args.animate), shell=True).communicate()
    if args.animate is None and args.prefix is not None:
        args.keep = True
    if not args.keep:
        sub.Popen("rm -f {0}_*.png".format(args.prefix),
                  shell=True).communicate()
    return None

def subclass_spider(cls):
    """ return recursive list of subclasses of cls (depth first) """
    lst = [cls]
    # reversed gives precedence to the more recently declared classes
    for c in reversed(cls.__subclasses__()):
        lst += subclass_spider(c)
    return lst

def timereps(reps, func, *args, **kwargs):
    arr = [None] * reps
    for i in range(reps):
        start = time()
        func(*args, **kwargs)
        end = time()
        arr[i] = end - start
    return min(arr), max(arr), sum(arr) / reps

def timeit(f, *args, **kwargs):
    t0 = time()
    ret = f(*args, **kwargs)
    t1 = time()

    print("Took {0:.03g} secs.".format(t1 - t0))
    return ret

def str_to_value(s):
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

def format_datetime(dt, fmt):
    r"""Wrapper around datetime.datetime.strftime

    This function allows one to specify the precision of fractional
    seconds (mircoseconds) using something like '%2f' to round to
    the nearest hundredth of a second.

    Args:
        dt (datetime.datetime): time to format
        fmt (str): format string that strftime understands with the
            addition of '%\.?[0-9]f' to the syntax.

    Returns:
        str
    """
    if len(fmt) == 0:
        msec_fmt = ['1']
        fmt = "%Y-%m-%d %H:%M:%S.%f"
    else:
        msec_fmt = re.findall(r"%\.?([0-9]*)f", fmt)
        fmt = re.sub(r"%\.?([0-9]*)f", "%f", fmt)

    tstr = datetime.strftime(dt, fmt)

    # now go back and for any %f -> [0-9]{6}, reformat the precision
    it = list(izip(msec_fmt, re.finditer("[0-9]{6}", tstr)))
    for ffmt, m in reversed(it):
        a, b = m.span()
        val = float("0." + tstr[a:b])
        ifmt = int(ffmt) if len(ffmt) > 0 else 6
        f = "{0:0.{1}f}".format(val, ifmt)[2:]
        tstr = tstr[:a] + f + tstr[b:]
    return tstr

def format_time(t, style='.02f'):
    """Format time as a string

    Note:
        If t is a datetime.datetime instance, then this function
        returns the result of format_datetime(t, style), which is
        basically datetime.datetime.strftime

    Args:
        t (float): time
        style (str): for this method, can be::
                style          |   time   | string
                --------------------------------------------------
                'hms'          | 90015.0  | "25:00:15.000"
                'hms'          | 90000.0  | "1 day 01:00:15.000"
                'hmss'         | 90015.0  | "25:00:15.000 (90015)"
                'dhms'         |   900.0  | "0 days 00:15:00.000"
                '.02f'         |   900.0  | '900.00'

    Returns:
        str
    """
    lstyle = style.lower()
    if t is None:
        return ""
    elif isinstance(t, datetime):
        return format_datetime(t, style)
    elif "hms" in lstyle:
        days = int(t // (24 * 3600))
        hrs = int((t // 3600) % 24)
        mins = int((t // 60) % 60)
        secs = t % 60

        if lstyle == "dhms":
            daystr = "day" if days == 1 else "days"
            return "{0} {1}, {2}:{3:02d}:{4:04.01f}".format(days, daystr,
                                                            hrs, mins, secs)
        elif lstyle.startswith("hms"):
            hrs += 24 * days
            s = "{0:02d}:{1:02d}:{2:05.2f}".format(hrs, mins, secs)
            if lstyle == "hmss":
                s += " ({0:d})".format(int(t))
            return s
        else:
            raise ValueError("Unknown time style: {0}".format(style))
    elif lstyle == "none":
        return ""
    else:
        return "{0:{1}}".format(t, style)
    raise NotImplementedError("should never be here")

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

    # ya know, lets just go through all the dimensions in shape
    # just to be safe and default to an empty slice / no reverse
    slices = slices + [slice(None)] * (len(shape) - len(slices))
    reverse = reverse + [False] * (len(slices) - len(reverse))

    first_slc = [slice(None)] * len(shape)
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

        first_slc[i] = slc

    if cull_second:
        second_slc = [s for s in second_slc if s is not None]
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

    _step = 1 if step is None else step
    epsilon_step = epsilon if _step > 0 else -epsilon

    start = convert_deprecated_floats(start, "start")
    stop = convert_deprecated_floats(stop, "stop")

    # print("?!? |{0}|  |{1}|".format(start, stop))

    try:
        start = start.rstrip()
        if len(start) == 0:
            start = None
        elif start[-1] == 'f':
            start = float(start[:-1])
            diff = arr - start + epsilon_step
            if _step > 0:
                diff = np.ma.masked_less_equal(diff, 0)
            else:
                diff = np.ma.masked_greater_equal(diff, 0)

            if np.ma.count(diff) == 0:
                # start value is past the wrong end of the array
                if _step > 0:
                    start = len(arr)
                else:
                    # start = -len(arr) - 1
                    # having a value < -len(arr) won't play
                    # nice with make_fwd_slice, but in this
                    # case, the slice will have no data, so...
                    return 0, 0, step
            else:
                start = int(np.argmin(np.abs(diff)))
    except AttributeError:
        pass

    try:
        stop = stop.rstrip()
        if len(stop) == 0:
            stop = None
        elif stop[-1] == 'f':
            stop = float(stop.rstrip()[:-1])
            diff = arr - stop - epsilon_step
            if _step > 0:
                diff = np.ma.masked_greater_equal(diff, 0)
                if np.ma.count(diff) == 0:
                    # stop value is past the wong end of the array
                    stop = 0
                else:
                    stop = int(np.argmin(np.abs(diff)))
                    if endpoint:
                        stop += 1
            else:
                diff = np.ma.masked_less_equal(diff, 0)
                if np.ma.count(diff) == 0:
                    # stop value is past the wrong end of the array
                    stop = len(arr)
                else:
                    stop = int(np.argmin(np.abs(diff)))
                    if endpoint:
                        if stop > 0:
                            stop -= 1
                        else:
                            # 0 - 1 == -1 which would wrap to the end of
                            # of the array... instead, just make it None
                            stop = None
    except AttributeError:
        pass

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

def to_slices(arrs, slices, endpoint=True, tol=100):
    """Wraps :py:func:`to_slice` for multiple arrays / slices

    Args:
        arrs (list, None): list of arrays for float lookups, must be
            the same length as s.split(','). If all slices are by
            index, then `arrs` can be `None`.
        slices (list, str): list of things that
            :py:func:`viscid.vutil.str2slice` understands, or a comma
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
        arrs = [None] * len()
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

    if hasattr(s, "__index__"):
        ret = s
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

        if len(slclst) == 1:
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

def slice_globbed_filenames(glob_pattern):
    """Apply a slice to a glob pattern

    Note:
        Slice by value works by adding an 'f' to a value, as like the
        rest of Viscid.

    Args:
        glob_pattern (str): A string

    Returns:
        list of filenames

    Examples:
        If a directory contains files,

            file.010.txt  file.020.txt  file.030.txt  file.040.txt

        then sliced globs can look like

        >>> expand_glob_slice("f*.[:2].txt")
        ["file.010.txt", "file.020.txt"]

        >>> expand_glob_slice("f*.[10.0f::2].txt")
        ["file.010.txt", "file.030.txt"]

        >>> expand_glob_slice("f*.[20f:2].txt")
        ["file.020.txt", "file.040.txt"]
    """
    glob_pattern = os.path.expanduser(os.path.expandvars(glob_pattern))
    glob_pattern = os.path.abspath(glob_pattern)

    # construct a regex to match the results
    # verify glob pattern has only one
    number_re = r"(?:[-+]?[0-9]*\.?[0-9]+f?|[-+]?[0-9+])"
    slc_re = r"\[({0})?(:({0})?){{0,2}}\]".format(number_re)
    n_slices = len(re.findall(slc_re, glob_pattern))

    if n_slices > 1:
        print("Multiple filename slices found, only using the first.",
              file=sys.stderr)

    if n_slices:
        m = re.search(slc_re, glob_pattern)
        slcstr = glob_pattern[m.start() + 1:m.end() - 1]
        edited_glob = glob_pattern[:m.start()] + "*" + glob_pattern[m.end():]
        res_re = glob_pattern[:m.start()] + "TSLICE" + glob_pattern[m.end():]
        res_re = fnmatch.translate(res_re)
        res_re = res_re.replace("TSLICE", r"(?P<TSLICE>.*?)")
    else:
        edited_glob = glob_pattern
        slcstr = ""

    fnames = glob(edited_glob)

    if n_slices:
        times = [float(re.match(res_re, fn).group('TSLICE')) for fn in fnames]
        fnames = [fn for fn, t in sorted(zip(fnames, times), key=itemgetter(1))]
        times.sort()
        slc = to_slice(times, slcstr)
    else:
        times = [None] * len(fnames)
        slc = slice(None)

    return fnames[slc]

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

def convert_deprecated_floats(value, varname="value"):
    if value_is_float_not_int(value):
        # TODO: eventually, a ValueError should be raised here
        print("Deprecation Warning!\n"
              "  Slicing by float is deprecated. The slice by value syntax is \n"
              "  now a string that has a trailing 'f', as in 'x=0f' [{0} = {1}]"
              "".format(varname, value), file=sys.stderr)
        value = "{0}f".format(value)
    return value

##
## EOF
##
