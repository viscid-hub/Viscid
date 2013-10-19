#!/usr/bin/env python

from __future__ import print_function
from itertools import islice
from timeit import default_timer as time
import logging

import numpy as np

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
    such as lists, numpy arrays, etc. These chunks will be 'contiguous', see
    chunk_interslice for picking every nth element.

    Note: Use chunk_iterator to chunk up iterators

    Returns: nchunks slices of length N = (len(lst) // nchunks) or N - 1

    ex: it1, it2, it3 = chunk_list(range(8), 3)
    it1 == range(0, 3)  # 3 vals
    it2 == range(3, 6)  # 3 vals
    it3 == range(6, 8)  # 2 vals
    """
    nel = len(seq)
    ret = chunk_slices(nel, nchunks)
    for i in range(nchunks):
        ret[i] = seq[slice(*ret[i])]
    return ret

def chunk_slices(nel, nchunks):
    """
    Get the slice info (can be unpacked & passed to the slice builtin as in
    slice(*ret[i])) for nchunks contiguous chunks in a list with nel elements

    nel: how many elements are in one pass of the original list
    nchunks: how many chunks to make
    Returns: a list of (start, stop) tuples with length nchunks

    ex: sl1, sl2 = chunk_slices(5, 2)
    -> sl1 == (0, 3)  # 3 vals
    -> sl2 == (3, 5)  # 2 vals
    """
    nlong = nel % nchunks  # nshort guarenteed < nchunks
    lenshort = nel // nchunks
    lenlong = lenshort + 1

    ret = [None] * nchunks
    start = 0
    for i in range(nlong):
        ret[i] = (start, start + lenlong)
        start += lenlong
    for i in range(nlong, nchunks):
        ret[i] = (start, start + lenshort)
        start += lenshort
    return ret

def chunk_interslices(nchunks):
    """
    Similar to chunk_slices, but pick every nth element instead of getting
    a contiguous block for each chunk

    nchunks: how many chunks to make
    Returns: a list of (start, stop, step) tuples with length nchunks

    ex: chunk_slices(2) == [(0, None, 2), (1, None, 2)]
    """
    ret = [None] * nchunks
    for i in range(nchunks):
        ret[i] = (i, None, nchunks)
    return ret

def chunk_sizes(nel, nchunks):
    """
    nel: how many elements are in one pass of the original list
    nchunks: is inferred from the length of iter_list
    Returns: an ndarray of the number of elements in each chunk, this
             should be the same for chunk_list, chunk_slices and
             chunk_interslices

    ex: nel1, nel2 = chunk_sizes(5, 2)
    -> nel1 == 2
    -> nel2 == 3
    """
    nlong = nel % nchunks  # nshort guarenteed < nchunks
    lenshort = nel // nchunks
    lenlong = lenshort + 1
    ret = np.empty((nchunks,), dtype="int")
    ret[:nlong] = lenlong
    ret[nlong:] = lenshort
    return ret

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
