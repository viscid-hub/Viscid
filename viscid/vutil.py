#!/usr/bin/env python

from __future__ import print_function
from itertools import islice
from timeit import default_timer as time
import logging

tree_prefix = ".   "

def find_field(vfile, fld_name_lst):
    """ convenience function to get a field that could be called many things
    returns the first fld_name in the list that is in the file """
    for fld_name in fld_name_lst:
        if fld_name in vfile:
            return vfile[fld_name]
    raise KeyError("file {0} contains none of {1}".format(vfile, fld_name_lst))

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

def chunk_list(seq, nchunks):
    """
    slice seq into chunks if nchunks size, seq can be a anything sliceable
    such as lists, numpy arrays, etc.
    Note: Use chunk_iterator to chunk up iterators
    Returns: nchunks slices of length N = (len(lst) // nchunks) or N - 1

    ex: it1, it2, it3 = chunk_list(range(8), 3)
    it1 == range(0, 2)  # 2 vals
    it2 == range(2, 5)  # 3 vals
    it3 == range(5, 8)  # 3 vals
    """
    nel = len(seq)
    nlong = nel % nchunks  # nshort guarenteed < nchunks
    nshort = nchunks - nlong
    lenshort = nel // nchunks
    ret = [None] * nchunks
    for i in range(nshort):
        start = i * lenshort
        ret[i] = seq[start:start + lenshort]
    for i in range(nlong):
        start = nshort * lenshort + i * (lenshort + 1)
        ind = nshort + i
        ret[ind] = seq[start:start + lenshort + 1]
    return ret

def chunk_iterator(iter_list, nel):
    """
    iter_list: list of independant iterators (not pointers to the
               same iterator a la [it]*nchunks), one for each chunk
               you want. They should all contain the same data... the
               returned iterators will be these iterators isliced to the
               right location / length
    nel: how many elements are in one pass of the original iterators
    nchunks: is inferred from the length of iter_list
    Returns: a list of nchunks iterators with N or N-1 elements
             where N is nel // nchunks

    ex: it1, it2 = chunk_iterable([range(5) for i in range(2)], 5)
    -> it1 == (i for i in range(0, 2))  # 2 vals
    -> it2 == (i for i in range(2, 5))  # 3 vals
    """
    nchunks = len(iter_list)
    nlong = nel % nchunks  # nshort guarenteed < nchunks
    nshort = nchunks - nlong
    lenshort = nel // nchunks
    ret = [None] * nchunks
    for i in range(nshort):
        start = i * lenshort
        ret[i] = islice(iter_list[i], start, start + lenshort)
    for i in range(nlong):
        start = nshort * lenshort + i * (lenshort + 1)
        ind = nshort + i
        ret[ind] = islice(iter_list[ind], start, start + lenshort + 1)
    return ret

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
