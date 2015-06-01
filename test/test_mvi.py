#!/usr/bin/env python
""" Try to convert a Field to a mayavi type and plot
streamlines or something """

from __future__ import print_function
import sys
import os
import argparse

from mayavi import mlab

_viscid_root = os.path.realpath(os.path.dirname(__file__) + '/../viscid/')
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)

import viscid
from viscid import vutil
from viscid.plot import mvi

def main():
    parser = argparse.ArgumentParser(description="Test calc")
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)

    f3d = viscid.load_file(_viscid_root + '/../sample/sample.3df.xdmf')

    if "b" in f3d:
        b = f3d["b"]
        bx, by, bz = b.component_fields() #pylint: disable=W0612
    elif "bx" in f3d:
        bx = f3d["bx"]
        by = f3d["by"]
        bz = f3d["bz"]
        b = viscid.scalar_fields_to_vector([bx, by, bz], name='b')
    else:
        raise RuntimeError("Where have all the B fields gone...")

    b_src = mvi.field_to_point_source(b)
    bsl2 = mlab.pipeline.streamline(b_src, seedtype='sphere',
                                    integration_direction='both',
                                    seed_resolution=4)
    bsl2.stream_tracer.maximum_propagation = 20.
    bsl2.seed.widget.center = [-8, 0, 0]
    bsl2.seed.widget.radius = 1.0
    bsl2.streamline_type = 'tube'
    bsl2.tube_filter.radius = 0.03
    bsl2.stop()  # this stop/start was a hack to get something to work?
    bsl2.start()
    bsl2.seed.widget.enabled = True

    mvi.mlab_earth(mlab.pipeline, crd_system="gse")

    if args.show:
        mlab.show()

if __name__ == "__main__":
    main()

##
## EOF
##
