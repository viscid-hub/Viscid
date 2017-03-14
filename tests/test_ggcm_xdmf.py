#!/usr/bin/env python
""" test a ggcm grid wrapper """

from __future__ import print_function
import argparse
import os

import ggcm_test_common

import viscid
from viscid import sample_dir
from viscid import vutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)

    f3d = viscid.load_file(os.path.join(sample_dir, 'sample_xdmf.3d.xdmf'),
                           grid_type=ggcm_test_common.MyGGCMGrid)
    ggcm_test_common.run_test_3d(f3d, __file__, show=args.show)

    f2d = viscid.load_file(os.path.join(sample_dir, 'sample_xdmf.py_0.xdmf'),
                           grid_type=ggcm_test_common.MyGGCMGrid)
    ggcm_test_common.run_test_2d(f2d, __file__, show=args.show)
    ggcm_test_common.run_test_timeseries(f2d, __file__, show=args.show)

    fiof = viscid.load_file(os.path.join(sample_dir, 'sample_xdmf.iof.xdmf'))
    ggcm_test_common.run_test_iof(fiof, __file__, show=args.show)

if __name__ == "__main__":
    main()

##
## EOF
##
