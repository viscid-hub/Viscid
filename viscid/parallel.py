"""common tools for parallel processing"""

from __future__ import print_function, division
from math import ceil
import threading
import multiprocessing as mp
import multiprocessing.pool
from contextlib import closing
from itertools import repeat
import sys

import numpy as np

import viscid
from viscid.compat import izip, futures, string_types


__all__ = ["chunk_list", "chunk_slices", "chunk_interslices", "chunk_sizes",
           "map", "map_async"]


# Non daemonic processes are probably a really bad idea
class NoDaemonProcess(mp.Process):
    """Using this is probably a bad idea"""
    # make 'daemon' attribute always return False
    @staticmethod
    def _get_daemon():
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class NoDaemonPool(multiprocessing.pool.Pool):  # pylint: disable=W0223
    """ I am vulnerable to armies of undead worker processes, chances
    are you don't actually want to use me
    """
    Process = NoDaemonProcess


class _MapThread(threading.Thread):
    def __init__(self, result_container, index, **kwargs):
        self.results = result_container
        self.index = index
        self.target = kwargs.pop("target")
        self.args = kwargs.pop("args", [])
        self.kwargs = kwargs.pop("kwargs", {})
        super(_MapThread, self).__init__(**kwargs)

    def run(self):
        self.results[self.index] = self.target(*self.args, **self.kwargs)


def chunk_list(seq, nchunks, size=None):
    """Chunk a list

    slice seq into chunks of nchunks size, seq can be a anything
    sliceable such as lists, numpy arrays, etc. These chunks will be
    'contiguous', see :meth:`chunk_interslice` for picking every nth
    element.

    Parameters:
        size: if given, set nchunks such that chunks have about 'size'
            elements

    Returns:
        nchunks slices of length N = (len(lst) // nchunks) or N - 1

    See Also:
        Use :meth:`chunk_iterator` to chunk up iterators

    Example:
        >>> it1, it2, it3 = chunk_list(range(8), 3)
        >>> it1 == range(0, 3)  # 3 vals
        True
        >>> it2 == range(3, 6)  # 3 vals
        True
        >>> it3 == range(6, 8)  # 2 vals
        True
    """
    nel = len(seq)

    if size is not None:
        nchunks = int(ceil(nel / nchunks))

    ret = chunk_slices(nel, nchunks)
    for i in range(nchunks):
        ret[i] = seq[slice(*ret[i])]
    return ret

def chunk_slices(nel, nchunks, size=None):
    r"""Make continuous chunks

    Get the slice info (can be unpacked and passed to the slice builtin
    as in slice(\*ret[i])) for nchunks contiguous chunks in a list with
    nel elements

    Parameters:
        nel: how many elements are in one pass of the original list
        nchunks: how many chunks to make
        size: if given, set nchunks such that chunks have about 'size'
            elements

    Returns:
        a list of (start, stop) tuples with length nchunks

    Example:
        >>> sl1, sl2 = chunk_slices(5, 2)
        >>> sl1 == (0, 3)  # 3 vals
        True
        >>> sl2 == (3, 5)  # 2 vals
        True
    """
    if size is not None:
        nchunks = int(ceil(nel / nchunks))

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
    """Make staggered chunks

    Similar to chunk_slices, but pick every nth element instead of
    getting a contiguous patch for each chunk

    Parameters:
        nchunks: how many chunks to make

    Returns:
        a list of (start, stop, step) tuples with length nchunks

    Example:
        >>> chunk_slices(2) == [(0, None, 2), (1, None, 2)]
        True
    """
    ret = [None] * nchunks
    for i in range(nchunks):
        ret[i] = (i, None, nchunks)
    return ret

def chunk_sizes(nel, nchunks, size=None):
    """For chunking up lists, how big is each chunk

    Parameters:
        nel: how many elements are in one pass of the original list
        nchunks: is inferred from the length of iter_list
        size: if given, set nchunks such that chunks have about 'size'
            elements
    Returns:
        an ndarray of the number of elements in each chunk, this
        should be the same for chunk_list, chunk_slices and
        chunk_interslices

    Example:
        >>> nel1, nel2 = chunk_sizes(5, 2)
        >>> nel1 == 2
        True
        >>> nel2 == 3
        True
    """
    if size is not None:
        nchunks = int(ceil(nel / nchunks))

    nlong = nel % nchunks  # nshort guarenteed < nchunks
    lenshort = nel // nchunks
    lenlong = lenshort + 1
    ret = np.empty((nchunks,), dtype="int")
    ret[:nlong] = lenlong
    ret[nlong:] = lenshort
    return ret

def _star_passthrough(args):
    """ this is so we can give a zipped iterable to func """
    # args[0] is function, args[1] is positional args, and args[2] is kwargs
    return args[0](*(args[1]), **(args[2]))

def sanitize_nr_procs(nr_procs):
    if isinstance(nr_procs, string_types):
        nr_procs = nr_procs.strip().lower()

    if nr_procs == "all" or nr_procs == "auto":
        nr_procs = mp.cpu_count()

    return int(nr_procs)

def map(nr_procs, func, args_iter, args_kw=None, timeout=1e8,
        daemonic=True, threads=False, pool=None, force_subprocess=False):
    """Just like ``subprocessing.map``?

    same as :meth:`map_async`, except it waits for the result to
    be ready and returns it

    Note:
        When using threads, this is WAY faster than map_async since
        map_async uses the builtin python ThreadPool. I have no idea
        why that's slower than making threads by hand.
    """
    nr_procs = sanitize_nr_procs(nr_procs)
    if args_kw is None:
        args_kw = {}

    # don't waste time spinning up a new process
    if threads:
        args = [(func, ai, args_kw) for ai in args_iter]
        with futures.ThreadPoolExecutor(max_workers=nr_procs) as executor:
            ret = [val for val in executor.map(_star_passthrough, args)]
    elif pool is None and nr_procs == 1 and not force_subprocess:
        args_iter = izip(repeat(func), args_iter, repeat(args_kw))
        ret = [_star_passthrough(args) for args in args_iter]
    else:
        p, r = map_async(nr_procs, func, args_iter, args_kw=args_kw,
                         daemonic=daemonic, threads=threads, pool=pool)
        ret = r.get(int(timeout))
        # in principle this join should return almost immediately since
        # we already called r.get
        p.join()

    return ret

def map_async(nr_procs, func, args_iter, args_kw=None, daemonic=True,
              threads=False, pool=None):
    """Wrap python's ``map_async``

    This has some utility stuff like star passthrough

    Run func on nr_procs with arguments given by args_iter. args_iter
    should be an iterable of the list of arguments that can be unpacked
    for each invocation. kwargs are passed to func as keyword arguments

    Returns:
        (tuple) (pool, multiprocessing.pool.AsyncResult)

    Note:
        When using threads, this is WAY slower than map since
        map_async uses the builtin python ThreadPool. I have no idea
        why that's slower than making threads by hand.

    Note: daemonic can be set to False if one needs to spawn child
        processes in func, BUT this could be vulnerable to creating
        an undead army of worker processes, only use this if you
        really really need it, and know what you're doing

    Example:
        >>> func = lambda i, letter: print i, letter
        >>> p, r = map_async(2, func, itertools.izip(itertools.count(), 'abc'))
        >>> r.get(1e8)
        >>> p.join()
        >>> # the following is printed from 2 processes
        0 a
        1 b
        2 c
    """
    nr_procs = sanitize_nr_procs(nr_procs)
    if args_kw is None:
        args_kw = {}

    if not threads and sys.platform == 'darwin' and ("mayavi.mlab" in sys.modules or
                                                     "mayavi" in sys.modules):
        import mayavi
        if mayavi.ETSConfig.toolkit == 'qt4':
            viscid.logger.critical("Using multiprocessing with Mayavi + Qt4 "
                                   "will cause segfaults on join.\n"
                                   "A workaround is to use the wx backend "
                                   "(`os.environ['ETS_TOOLKIT'] = 'wx'`).")

    args_iter = izip(repeat(func), args_iter, repeat(args_kw))

    # if given a pool, don't close it when we're done delegating tasks
    if pool is not None:
        return pool, pool.map_async(_star_passthrough, args_iter)
    else:
        if threads:
            pool = mp.pool.ThreadPool(nr_procs)
        elif daemonic:
            pool = mp.Pool(nr_procs)
        else:
            pool = NoDaemonPool(nr_procs)

        with closing(pool) as p:
            return p, p.map_async(_star_passthrough, args_iter)

##
## EOF
##
