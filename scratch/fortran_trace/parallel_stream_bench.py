#!/usr/bin/env python

from __future__ import print_function
import sys
import os
import cProfile
import pstats
from timeit import default_timer as time
import argparse

from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
from mayavi import mlab
import numpy as np

_viscid_root = os.path.realpath(os.path.dirname(__file__) + '/../../src/viscid/')  # pylint: disable=C0301
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)

import tracer
from viscid import vutil
from viscid import readers
from viscid import field
from viscid.plot import vpyplot as vlt
from viscid.plot import vlab
from viscid.calculator import streamline
from viscid.calculator import seed
import topo_numba

gsize = (16, 16, 16)
x1 = -10.0; x2 = -5.0  # pylint: disable=C0321
y1 = -5.0; y2 = 5.0  # pylint: disable=C0321
z1 = -5.0; z2 = 5.0  # pylint: disable=C0321
vol = seed.Volume((z1, y1, x1), (z2, y2, x2), gsize)

def trace_cython(B, nr_procs, force_subprocess=False):
    lines, topo = streamline.streamlines(B, vol, ds0=0.02, ibound=3.7,
                            maxit=5000, output=streamline.OUTPUT_BOTH,
                            method=streamline.EULER1,
                            nr_procs=nr_procs,
                            force_subprocess=force_subprocess)
    return lines, topo

def main():
    parser = argparse.ArgumentParser(description="Test xdmf")
    parser.add_argument("--show", "--plot", action="store_true")
    parser.add_argument('file', nargs="?", default=None)
    args = vutil.common_argparse(parser)

    # f3d = readers.load_file(_viscid_root + '/../../sample/sample.3df.xdmf')
    # b3d = f3d['b']
    # bx, by, bz = b3d.component_fields()  # pylint: disable=W0612

    if args.file is None:
        args.file = "/Users/kmaynard/dev/work/cen4000/cen4000.3d.xdmf"
    f3d = readers.load_file(args.file)

    bx = f3d["bx"]
    by = f3d["by"]
    bz = f3d["bz"]
    B = field.scalar_fields_to_vector([bx, by, bz], name="B_cc",
                                      _force_layout=field.LAYOUT_INTERLACED)

    t0 = time()
    lines_single, topo = trace_cython(B, nr_procs=1)
    t1 = time()
    topo_single = vol.wrap_field(topo, name="CyTopo1")
    print("single proc:", t1 - t0, "s")

    nr_procs_list = np.array([1, 2, 3])
    # nr_procs_list = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    times = np.empty_like(nr_procs_list, dtype="float")

    print("always parallel overhead now...")
    for i, nr_procs in enumerate(nr_procs_list):
        t0 = time()
        lines, topo = trace_cython(B, nr_procs=nr_procs, force_subprocess=True)
        t1 = time()
        fld = vol.wrap_field(topo, name="CyTopo")
        same_topo = (fld.data == topo_single.data).all()
        same_lines = True
        for j, line in enumerate(lines):
            same_lines = same_lines and (line == lines_single[j]).all()
        print("nr_procs:", nr_procs, "time:", t1 - t0, "s",
              "same topo:", same_topo, "same lines:", same_lines)
        times[i] = t1 - t0

    plt.plot(nr_procs_list, times, 'k^')
    plt.plot(nr_procs_list, times[0] / nr_procs_list, 'b--')
    plt.xscale("log")
    plt.yscale("log")
    plt.show()

    if False:
        cmap = plt.get_cmap('spectral')
        levels = [4, 5, 6, 7, 8, 13, 14, 16, 17]
        norm = BoundaryNorm(levels, cmap.N)
        vlt.plot(topo_flds[-1], "y=0", cmap=cmap, norm=norm, show=False)
        #vlt.plot_streamlines2d(lines[::5], "y", topology=topo[::5], show=False)
        #vlt.plot_streamlines(lines, topology=topo, show=False)
        vlt.mplshow()

        # topo_src = vlab.add_field(topo_fld, center='node')
        # vlab.plot_lines(mlab.pipeline, lines[::5], topo[::5], opacity=0.8,
        #                 tube_radius=0.02)
        # vlab.plot_earth_3d()
        # mlab.show()

if __name__ == "__main__":
    main()

##
## EOF
##
