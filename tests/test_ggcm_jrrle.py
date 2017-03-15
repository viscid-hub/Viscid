#!/usr/bin/env python
"""Test reading / plotting OpenGGCM jrrle files

Note that the business end of this test is in `ggcm_test_common`. Also,
in `ggcm_test_common`, you'll find how to subclass a standard grid or
reader to inject specific behavior.
"""

from __future__ import print_function
import argparse
import sys
import os

import ggcm_test_common

import viscid
from viscid import sample_dir
from viscid import vutil


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)

    f3d = viscid.load_file(os.path.join(sample_dir, 'sample_jrrle.3df.*'),
                           grid_type=ggcm_test_common.MyGGCMGrid)
    ggcm_test_common.run_test_3d(f3d, __file__, show=args.show)

    f2d = viscid.load_file(os.path.join(sample_dir, 'sample_jrrle.py_0.*'),
                           grid_type=ggcm_test_common.MyGGCMGrid)
    ggcm_test_common.run_test_2d(f2d, __file__, show=args.show)
    ggcm_test_common.run_test_timeseries(f2d, __file__, show=args.show)

    fiof = viscid.load_file(os.path.join(sample_dir, 'sample_jrrle.iof.*'))
    ggcm_test_common.run_test_iof(fiof, __file__, show=args.show)

    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
