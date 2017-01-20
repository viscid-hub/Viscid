#!/usr/bin/env python

# FIXME: this module is way too long and disorganized

from __future__ import print_function, division

import datetime
import fnmatch
from glob import glob
from itertools import count, chain
import logging
from operator import itemgetter
import os.path
import re
import subprocess as sub
import sys
from timeit import default_timer as time

import viscid
from viscid import logger
from viscid.compat import izip, string_types

import numpy as np


__all__ = ["timeit", "resolve_path", "find_item", "find_items",
           "slice_globbed_filenames", "meshlab_convert"]


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

def common_argparse(parser, default_verb=0):
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

    return args

def make_animation(movie_fname, prefix, framerate=5, qscale=2, keep=False,
                   args=None, frame_idx_fmt="_%06d", program="ffmpeg",
                   yes=False):
    """ make animation by calling program (only ffmpeg works for now) using
    args, which is a namespace filled by the argparse options from
    add_animate_arguments. Plots are expected to be named
    ${args.prefix}_000001.png where the number is in order from 1 up """
    if args is not None:
        prefix = args.prefix
        framerate = args.framerate
        qscale = args.qscale
        movie_fname = args.animate
        keep = args.keep

    if movie_fname:
        cmd = "yes | {0}".format(program) if yes else program
        if program == "ffmpeg":
            sub.Popen("{0} -r {1} -i {3}{4}.png -pix_fmt yuv420p "
                      "-qscale {2} {5}".format(cmd, framerate, qscale, prefix,
                                               frame_idx_fmt, movie_fname),
                      shell=True).communicate()
    if movie_fname is None and prefix is not None:
        keep = True
    if not keep:
        sub.Popen("rm -f {0}_*.png".format(prefix), shell=True).communicate()
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
    """overly simple timeit wrapper

    Arguments:
        f: callable to timeit
        *args: positional arguments for `f`
        **kwargs: keyword arguments for `f`

    Keyword arguments:
        timeit_repeat (int): number of times to call `f` (Default: 1)
        timeit_print_stats (bool): print min/max/mean/median when done
        timeit_quet (bool): quiets all output (useful if you only want
            the timeit_stats dict filled)
        timeit_stats (dict): Stats will be stuffed into here

    Returns:
        The result of `f(*args, **kwargs)`
    """
    timeit_repeat = kwargs.pop('timeit_repeat', 1)
    timeit_print_stats = kwargs.pop('timeit_print_stats', True)
    timeit_quiet = kwargs.pop('timeit_quiet', False)
    timeit_stats = kwargs.pop('timeit_stats', dict())

    times = np.empty((timeit_repeat,), dtype='f8')

    for i in range(timeit_repeat):
        ret = None
        t0 = time()
        ret = f(*args, **kwargs)
        t1 = time()

        s = "{0:.03g}".format(t1 - t0)
        times[i] = t1 - t0
        if not timeit_quiet and (timeit_repeat == 1 or not timeit_print_stats):
            secs = "second" if s == "1" else "seconds"
            print("<function {0}.{1}>".format(f.__module__, f.__name__),
                  "took", s, secs)

    timeit_stats['min'] = np.min(times)
    timeit_stats['max'] = np.max(times)
    timeit_stats['mean'] = np.mean(times)
    timeit_stats['median'] = np.median(times)
    timeit_stats['repeat'] = timeit_repeat

    if not timeit_quiet and timeit_repeat > 1 and timeit_print_stats:
        print("<function {0}.{1}> stats ({2} runs):"
              "".format(f.__module__, f.__name__, timeit_repeat))
        print("  Min: {min:.3g}, Mean: {mean:.3g}, Median: {median:.3g}, "
              "Max: {max:.3g}".format(**timeit_stats))

    return ret

def resolve_path(dset, loc, first=False):
    """Search for globbed paths in a nested dict-like hierarchy

    Args:
        dset (dict): Root of some nested dict-like hierarchy
        loc (str): path as a glob pattern
        first (bool): Stop at first match and return a single value

    Raises:
        KeyError: If there are no glob matches

    Returns:
        If first == True, (value, path)
        else, ([value0, value1, ...], [path0, path1, ...])
    """
    try:
        if first:
            return dset[loc], loc
        else:
            return [dset[loc]], [loc]
    except KeyError:
        searches = [loc.strip('/').split('/')]
        dsets = [dset]
        paths = [[]]

        while any(searches):
            next_dsets = []
            next_searches = []
            next_paths = []
            for dset, search, path in viscid.izip(dsets, searches, paths):
                try:
                    next_dsets.append(dset[search[0]])
                    next_searches.append(search[1:])
                    next_paths.append(path + [search[0]])
                except (KeyError, TypeError, IndexError):
                    s = [{}.items()]
                    if hasattr(dset, 'items'):
                        s.append(dset.items())
                    if hasattr(dset, 'attrs'):
                        s.append(dset.attrs.items())
                    for key, val in chain(*s):
                        if fnmatch.fnmatchcase(key, search[0]):
                            next_dsets.append(val)
                            next_searches.append(search[1:])
                            next_paths.append(path + [key])
                            if first:
                                break
            dsets = next_dsets
            searches = next_searches
            paths = next_paths

    if dsets:
        dsets, paths = dsets, ['/'.join(p) for p in paths]
        if first:
            return dsets[0], paths[0]
        else:
            return dsets, paths
    else:
        raise KeyError("Path {0} has no matches".format(loc))

def find_item(dset, loc):
    """Shortcut for first :py:func:`resolve_path`, item only"""
    return resolve_path(dset, loc, first=True)[0]

def find_items(dset, loc):
    """Shortcut for :py:func:`resolve_path`, items only"""
    return resolve_path(dset, loc)[0]

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

        if viscid.is_datetime_like(s, conservative=True):
            byval = s.astype(arr.dtype)
            _epsilon_step = 0
        elif viscid.is_timedelta_like(s, conservative=True):
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
        viscid.logger.warn("Multiple filename slices found, only using the "
                           "first.")

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
        if not fnames:
            raise IOError("the glob {0} matched no files".format(edited_glob))

        times = []
        _newfn = []
        for fn in fnames:
            try:
                times.append(float(re.match(res_re, fn).group('TSLICE')))
                _newfn.append(fn)
            except ValueError:
                pass
        fnames = _newfn
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

def convert_timelike(value):
    if value in (None, ''):
        return None
    elif viscid.is_datetime_like(value, conservative=True):
        return viscid.as_datetime64(value)
    elif viscid.is_timedelta_like(value, conservative=True):
        return viscid.as_timedelta64(value)
    else:
        return value

def convert_deprecated_floats(value, varname="value"):
    if value_is_float_not_int(value):
        # TODO: eventually, a ValueError should be raised here
        s = ("DEPRECATION...\n"
             "Slicing by float is deprecated. The slice by value syntax is \n"
             "now a string that has a trailing 'f', as in 'x=0f' [{0} = {1}]"
             "".format(varname, value))
        viscid.logger.warn(s)
        value = "{0}f".format(value)
    return value

def prepare_lines(lines, scalars=None, do_connections=False, other=None):
    """Concatenate and standardize a list of lines

    Args:
        lines (list): Must be a list of 3xN or 4xN ndarrays of xyz(s)
            data for N points along the line. N need not be the same
            for all lines. Can alse be 6xN such that lines[:][3:, :]
            are interpreted as rgb colors
        scalars (ndarray, list): Can have shape 1xN for a single scalar
            or 3xN for an rgb color for each point. If the shape is
            1xNlines, the scalar is broadcast so the whole line gets
            the same value, and likewise for 3xNlines and rgb colors.
            Can also be a list of hex color (#ffffff) strings.
            Otherwise, scalars is reshaped to -1xN.
        do_connections (bool): Whether or not to make connections array
        other (dict): a dictionary of other arrays that should be
            reshaped and the like the same way scalars is

    Returns:
        (vertices, scalars, connections, other)

        * vertices (ndarray): 3xN array of N xyz points. N is the sum
            of the lengths of all the lines
        * scalars (ndarray): N array of scalars, 3xN array of uint8
            rgb values, or None
        * connections (ndarray): Nx2 array of ints (indices along
            axis 1 of vertices) describing the forward and backward
            connectedness of the lines, or None
        * other (dict): a dict of N length arrays

    Raises:
        ValueError: If rgb data is not in a valid range or the shape
            of scalars is not understood
    """
    nlines = len(lines)
    npts = [line.shape[1] for line in lines]
    N = np.sum(npts)
    first_idx = np.cumsum([0] + npts[:-1])
    vertices = [np.asarray(line) for line in lines]
    vertices = np.concatenate(lines, axis=1)
    if vertices.dtype.kind not in 'fc':
        vertices = np.asarray(vertices, dtype='f')

    if vertices.shape[0] > 3:
        if scalars is not None:
            viscid.logger.warn("Overriding line scalars with scalars kwarg")
        else:
            scalars = vertices[3:, :]
        vertices = vertices[:3, :]

    if scalars is not None:
        if isinstance(scalars, viscid.field.Field):
            scalars = viscid.interp_trilin(scalars, vertices)
            if scalars.size != N:
                raise ValueError("Scalars was not a scalar field")

        scalars = np.atleast_2d(scalars)

        if scalars.shape == (1, 1):
            scalars = scalars.repeat(N, axis=1)
        elif scalars.shape == (1, nlines) or scalars.shape == (nlines, 1):
            # one scalar for each line, so broadcast it
            scalars = scalars.reshape(nlines, 1)
            scalars = [scalars[i].repeat(ni) for i, ni in enumerate(npts)]
            scalars = np.concatenate(scalars, axis=0).reshape(1, N)
        elif scalars.shape == (N, 1) or scalars.shape == (1, N):
            # catch these so they're not interpreted as colors if
            # nlines == 1 and N == 3; ie. 1 line with 3 points
            scalars = scalars.reshape(1, N)
        elif scalars.shape == (3, nlines) or scalars.shape == (nlines, 3):
            # one rgb color for each line, so broadcast it
            if scalars.shape == (3, nlines):
                scalars = scalars.T
            colors = []
            for i, ni in enumerate(npts):
                c = scalars[i].reshape(3, 1).repeat(ni, axis=1)
                colors.append(c)
            scalars = np.concatenate(colors, axis=1)
        else:
            scalars = scalars.reshape(-1, N)

        if scalars.dtype.kind in ['S', 'U']:
            # translate hex colors (#ff00ff) into rgb values
            scalars = np.char.lstrip(scalars, '#').astype('S6')
            scalars = np.char.zfill(scalars, 6)
            # this np.char.decode(..., 'hex') doesn't work for py3k; kinda silly
            try:
                scalars = np.frombuffer(np.char.decode(scalars, 'hex'), dtype='u1')
            except LookupError:
                import codecs
                scalars = np.frombuffer(codecs.decode(scalars, 'hex'), dtype='u1')
            scalars = scalars.reshape(-1, 3).T
        elif scalars.shape[0] == 1:
            # normal scalars
            scalars = scalars.reshape(-1)
        elif scalars.shape[0] == 3:
            # The scalars encode rgb data, standardize the result to a
            # 3xN ndarray of 1 byte unsigned ints (chars)
            if np.all(scalars >= 0) and np.all(scalars <= 1):
                scalars = (255 * scalars).round().astype('u1')
            elif np.all(scalars >= 0) and np.all(scalars < 256):
                scalars = scalars.round().astype('u1')
            else:
                raise ValueError("Rgb data should be in range [0, 1] or "
                                 "[0, 255], range given is [{0}, {1}]"
                                 "".format(np.min(scalars), np.max(scalars)))
        else:
            raise ValueError("Scalars should either be a number, or set of "
                             "rgb values, shape is {0}".format(scalars.shape))

    # broadcast / reshape additional arrays given in other
    if other:
        for key, arr in other.items():
            if arr is None:
                pass
            elif arr.shape == (1, nlines) or arr.shape == (nlines, 1):
                arr = arr.reshape(nlines, 1)
                arr = [arr[i].repeat(ni) for i, ni in enumerate(npts)]
                other[key] = np.concatenate(arr, axis=0).reshape(1, N)
            else:
                try:
                    other[key] = arr.reshape(-1, N)
                except ValueError:
                    viscid.logger.warn("Unknown dimension, dropping array {0}"
                                       "".format(key))

    if do_connections:
        connections = [None] * nlines
        for i, ni in enumerate(npts):
            # i0 is the index of the first point of the i'th line in lines
            i0 = first_idx[i]
            connections[i] = np.vstack([np.arange(i0, i0 + ni - 1.5),
                                        np.arange(i0 + 1, i0 + ni - 0.5)]).T
        connections = np.concatenate(connections, axis=0).astype('i')
    else:
        connections = None

    return vertices, scalars, connections, other

def meshlab_convert(fname, fmt="dae", quiet=True):
    """Run meshlabserver to convert 3D mesh files

    Uses `MeshLab <http://meshlab.sourceforge.net/>`_, which is a great
    little program for playing with 3D meshes. The best part is that
    OS X's Preview can open the COLLADA (`*.dae`) format. How cool is
    that?

    Args:
        fname (str): file to convert
        fmt (str): extension of result, defaults to COLLADA format
        quiet (bool): redirect output to :py:attr:`os.devnull`

    Returns:
        None
    """
    iname = fname
    oname = '.'.join(iname.split('.')[:-1]) + "." + fmt.strip()
    redirect = "&> {0}".format(os.devnull) if quiet else ""
    cmd = ("meshlabserver -i {0} -o {1} -m vc vn fc fn {2}"
           "".format(iname, oname, redirect))
    sub.Popen(cmd, shell=True, stdout=None, stderr=None)

##
## EOF
##
