"""Execute computations asynchronously using threads or processes."""

try:
    from concurrent.futures import _base
    from concurrent.futures import *  # pylint: disable=wildcard-import
except ImportError:
    # Copyright 2009 Brian Quinlan. All Rights Reserved.
    # Licensed to PSF under a Contributor Agreement.

    __author__ = 'Brian Quinlan (brian@sweetapp.com)'

    from viscid.compat.futures import _base
    from viscid.compat.futures._base import (FIRST_COMPLETED,
                                             FIRST_EXCEPTION,
                                             ALL_COMPLETED,
                                             CancelledError,
                                             TimeoutError,
                                             Future,
                                             Executor,
                                             wait,
                                             as_completed)
    from viscid.compat.futures.thread import ThreadPoolExecutor

    try:
        from viscid.compat.futures.process import ProcessPoolExecutor
    except ImportError:
        # some platforms don't have multiprocessing
        pass
