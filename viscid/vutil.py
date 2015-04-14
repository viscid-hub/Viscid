#!/usr/bin/env python

from __future__ import print_function, division
from itertools import count
from timeit import default_timer as time
import subprocess as sub
import logging
from datetime import datetime, timedelta
import re

from viscid import logger
from viscid.compat import izip

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
        msec_fmt = []
        fmt = "%Y-%m-%d %H:%M:%S"
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
    if isinstance(t, datetime):
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
            assert start is None or stop is None or start < stop
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

def convert_floating_slice(arr, start, stop=None, step=None,
                           endpoint=True, tol=100):
    """Be able to slice using floats relative to an array

    The idea is that [0.1, 0.2, 0.3, 0.4, 0.5, 0.6][0.2:0.6:2] should
    be able to give you [0.2, 0.4, 0.6]. This function will handle all
    the paticulars, like when to include what and what if step < 1 etc.

    The rules for when something is included are:
        - The slice will never include elements whose value in arr
          are < start (or > if the slice is backward)
        - The slice will never include elements whose value in arr
          are > stop (or < if the slice is backward)
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
        start (None, int, float): like slice().start
        stop (None, int, float): like slice().start
        step (None, int, float): like slice().start
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
    try:
        epsilon = tol * np.finfo(arr.dtype).eps
    except ValueError:
        # array is probably of type numpy.int*
        epsilon = 0

    _step = 1 if step is None else step
    epsilon_step = epsilon if _step > 0 else -epsilon

    if isinstance(start, (float, np.floating)):
        diff = arr - start + epsilon_step
        if epsilon_step > 0:
            diff = np.ma.masked_less_equal(diff, 0)
        else:
            diff = np.ma.masked_greater_equal(diff, 0)

        if np.ma.count(diff) == 0:
            # start value is past the wrong end of the array
            if epsilon_step > 0:
                start = len(arr)
            else:
                # start = -len(arr) - 1
                # having a value < -len(arr) won't play
                # nice with make_fwd_slice, but in this
                # case, the slice will have no data, so...
                return 0, 0, step
        else:
            start = int(np.argmin(np.abs(diff)))

    if isinstance(stop, (float, np.floating)):
        diff = arr - stop - epsilon_step
        if epsilon_step > 0:
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

##
## EOF
##
