#!/usr/bin/env python

from __future__ import print_function
from itertools import islice
from timeit import default_timer as time
import logging
import subprocess as sub

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
    """ add common options for animating """
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
    """ add common options for tuning matplotlib output """
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

def common_argparse(parser, **kwargs):
    """ add some common verbosity stuff to argparse, parse the
    command line args, and setup the logging levels
    parser should be an ArgumentParser instance, and kwargs
    should be options that get passed to logging.basicConfig
    returns the args namespace  """
    general = parser.add_argument_group("Viscid general options")
    general.add_argument("--log", action="store", type=str, default=None,
                         help="Logging level (overrides verbosity)")
    general.add_argument("-v", action="count", default=0,
                         help="increase verbosity")
    general.add_argument("-q", action="count", default=0,
                         help="decrease verbosity")
    args = parser.parse_args()

    # setup the logging level
    if not "level" in kwargs:
        if args.log is not None:
            kwargs["level"] = getattr(logging, args.log.upper())
        else:
            # default = 30 WARNING
            verb = args.v - args.q
            kwargs["level"] = int(30 - 10 * verb)
    if not "format" in kwargs:
        kwargs["format"] = "%(levelname)s: %(message)s"
    logging.basicConfig(**kwargs)

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
    sub_classes = cls.__subclasses__()
    lst = [cls]
    for c in sub_classes:
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

##
## EOF
##
