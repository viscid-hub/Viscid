#!/usr/bin/env python
"""Test parallel streamlines with both threads and processes

This test asserts that the result on any number of processes is almost
equal. In principle, they should be exactly equal.
"""

from __future__ import print_function
import argparse
import sys

import numpy as np

import viscid_test_common  # pylint: disable=unused-import

import viscid


def do(timeit, func, *args, **kwargs):
    if timeit:
        return viscid.vutil.timeit(func, *args, **kwargs)
    else:
        return func(*args, **kwargs)

def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeit", '-t', action="store_true")
    args = viscid.vutil.common_argparse(parser, default_verb=0)

    for nu in (False, True):
        viscid.logger.info("Test set, nonuniform = {0}".format(nu))
        b = viscid.make_dipole(l=(-20, -6.4, -6.4), h=(20, 6.4, 6.4),
                               n=(256, 128, 128), dtype='f4', nonuniform=nu)
        seed = viscid.Circle(p0=(0, 0, 0), pole=(0, 0, 1), r=5.5, n=int(1e4))

        kwargs = dict(method='euler1')

        viscid.logger.info("Serial test...")
        l0, t0 = do(args.timeit, viscid.calc_streamlines, b, seed,
                    nr_procs=1, threads=False, **kwargs)
        viscid.logger.info("Parallel test (processes)...")
        l1, t1 = do(args.timeit, viscid.calc_streamlines, b, seed,
                    nr_procs=2, threads=False, **kwargs)
        viscid.logger.info("Parallel test (threads)...")
        l2, t2 = do(args.timeit, viscid.calc_streamlines, b, seed,
                    nr_procs=2, threads=True, **kwargs)

        np.testing.assert_almost_equal(t0.data, t1.data)
        np.testing.assert_almost_equal(t0.data, t2.data)
        assert len(l0) > 0
        assert len(l0) == len(l1)
        assert len(l0) == len(l2)
        for i in range(len(l0)):
            np.testing.assert_almost_equal(l0[i], l1[i])
            np.testing.assert_almost_equal(l0[i], l2[i])

    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
