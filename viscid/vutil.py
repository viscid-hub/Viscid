#!/usr/bin/env python

from __future__ import print_function
from timeit import default_timer as time
import logging

from . import verror

def find_field(vfile, fld_name_lst):
    """ convenience function to get a field that could be called many things
    returns the first fld_name in the list that is in the file """
    for fld_name in fld_name_lst:
        if fld_name in vfile:
            return vfile[fld_name]
    raise verror.FieldNotFound("file {0} contains none of "
                               "{1}".format(vfile, fld_name_lst))

def common_argparse(parser, **kwargs):
    """ add some common verbosity stuff to argparse, parse the
    command line args, and setup the logging levels
    parser should be an ArgumentParser instance, and kwargs
    should be options that get passed to logging.basicConfig
    returns the args namespace  """
    parser.add_argument("--log", action="store", type=str, default=None,
                        help="Logging level (overrides verbosity)")
    parser.add_argument("-v", action="count", default=0,
                        help="increase verbosity")
    parser.add_argument("-q", action="count", default=0,
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

def subclass_spider(cls):
    """ return recursive list of subclasses of cls """
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
