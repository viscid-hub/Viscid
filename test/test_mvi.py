#!/usr/bin/env python
# Try to convert a Field to a mayavi type and plot streamlines or something

from __future__ import print_function
import sys
import os

from mayavi import mlab

_viscid_root = os.path.realpath(os.path.dirname(__file__) + '/../src/viscid/')
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)

from viscid import readers
from viscid import field
from viscid.plot import mvi

verb = 0

def main():
    show = "--plot" in sys.argv or "--show" in sys.argv

    f3d = readers.load(_viscid_root + '/../../sample/sample.3df.xdmf')

    if "b" in f3d:
        b = f3d["b"]
        bx, by, bz = b.component_fields() #pylint: disable=W0612
    elif "bx" in f3d:
        bx = f3d["bx"]
        by = f3d["by"]
        bz = f3d["bz"]
        b = field.scalar_fields_to_vector("b", [bx, by, bz])
    else:
        raise RuntimeError("Where have all the B fields gone...")

    b_src = mvi.field_to_source(b)
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

    mvi.mlab_earth(mlab.pipeline)

    if show:
        mlab.show()

if __name__ == "__main__":
    if "-v" in sys.argv:
        verb += 1
    main()

##
## EOF
##
