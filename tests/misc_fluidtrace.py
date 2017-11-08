#!/usr/bin/env python
"""Test reading / plotting OpenGGCM xdmf files

Note that the business end of this test is in `ggcm_test_common`. Also,
in `ggcm_test_common`, you'll find how to subclass a standard grid or
reader to inject specific behavior.
"""

from __future__ import print_function
import argparse
import sys
import os

import numpy as np

import viscid
from viscid import sample_dir
from viscid import vutil


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)

    f = viscid.load_file(os.path.join(sample_dir, 'sample_xdmf.py_0.xdmf'))

    xl, xh = f.get_grid().xl_nc, f.get_grid().xh_nc
    seeds0 = viscid.Circle(p0=0.5 * (xl + xh), r=0.2 * np.max(xh - xl),
                           pole=(0, 0, 1), n=10)
    seeds1 = viscid.Circle(p0=0.5 * (xl + xh), r=0.4 * np.max(xh - xl),
                           pole=(0, 0, 1), n=10)
    seeds = viscid.Point(np.concatenate([seeds0.get_points(),
                                         seeds1.get_points()], axis=1))

    viscid.follow_fluid(f, seeds, dt=1.0, speed_scale=1 / 6.4e3,
                        callback_kwargs=dict(show=args.show,
                                             series_fname='plots/fluidtrace'))

    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
