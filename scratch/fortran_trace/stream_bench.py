#!/usr/bin/env python

from __future__ import print_function
import sys
import os
import cProfile
import pstats
from timeit import default_timer as time
import argparse

import numpy as np
from mayavi import mlab
from matplotlib.colors import BoundaryNorm

_viscid_root = os.path.realpath(os.path.dirname(__file__) + '/../../src/viscid/')  # pylint: disable=C0301
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)

from viscid import logger
import tracer
from viscid import vutil
from viscid import readers
from viscid import field
from viscid.plot import mpl
from viscid.plot import mvi
from viscid.plot.mpl import plt
from viscid.calculator import streamline
from viscid.calculator import seed
# import topo_numba

gsize = (32, 32, 16)
x1 = -10.0; x2 = -5.0  # pylint: disable=C0321
y1 = -5.0; y2 = 5.0  # pylint: disable=C0321
z1 = -5.0; z2 = 5.0  # pylint: disable=C0321
vol = seed.Volume((z1, y1, x1), (z2, y2, x2), gsize)

def trace_fortran(fld_bx, fld_by, fld_bz):
    gx, gy, gz = fld_bx.get_crds_cc(('x', 'y', 'z'))
    nz, ny, nx = fld_bx.shape

    t0 = time()
    bx_farr = np.array(np.ravel(fld_bx.data, order='K').reshape((nx, ny, nz), order="F"))  # pylint: disable=C0301
    by_farr = np.array(np.ravel(fld_by.data, order='K').reshape((nx, ny, nz), order="F"))  # pylint: disable=C0301
    bz_farr = np.array(np.ravel(fld_bz.data, order='K').reshape((nx, ny, nz), order="F"))  # pylint: disable=C0301

    topo = np.zeros(gsize, order='F', dtype='int32')
    nsegs = np.zeros((1,), order='F', dtype='int32')

    # logger.info((np.ravel(bx_farr, order='K') == np.ravel(fld_bx.data, order='K')).all())  # pylint: disable=C0301
    # logger.info((np.ravel(by_farr, order='K') == np.ravel(fld_by.data, order='K')).all())  # pylint: disable=C0301
    # logger.info((np.ravel(bz_farr, order='K') == np.ravel(fld_bz.data, order='K')).all())  # pylint: disable=C0301
    # logger.info("bx_arr")
    # logger.info(bx_farr.strides)
    # logger.info(bx_farr.flags)
    # logger.info("topo")
    # logger.info(topo.strides)
    # logger.info(topo.shape)

    tracer.get_topo(gx, gy, gz, bx_farr, by_farr, bz_farr, topo,
                    x1, x2, y1, y2, z1, z2, nsegs)
    t1 = time()

    t = t1 - t0
    print("total segments calculated: {0:.05e}".format(float(nsegs)))
    print("time: {0:.4}s ... {1:.4}s/segment".format(t, t / float(nsegs)))

    topo_fld = vol.wrap_field(topo, name="FortTopo")
    return None, topo_fld

def trace_cython(fld_bx, fld_by, fld_bz):
    # print("Cython...")
    B = field.scalar_fields_to_vector([fld_bx, fld_by, fld_bz], name="B_cc",
                                      _force_layout=field.LAYOUT_INTERLACED)
    t0 = time()
    lines, topo = None, None
    lines, topo = streamline.streamlines(B, vol, ds0=0.02, ibound=3.7,
                            maxit=5000, output=streamline.OUTPUT_BOTH,
                            method=streamline.EULER1,
                            tol_lo=0.005, tol_hi=0.1,
                            fac_refine=0.75, fac_coarsen=1.5)
    t1 = time()
    topo_fld = vol.wrap_field(topo, name="CyTopo")

    # cmap = plt.get_cmap('spectral')
    # levels = [4, 5, 6, 7, 8, 13, 14, 16, 17]
    # norm = BoundaryNorm(levels, cmap.N)
    # mpl.plot(topo_fld, "y=0", cmap=cmap, norm=norm, show=False)
    # # mpl.plot_streamlines2d(lines[::5], "y", topology=topo[::5], show=False)
    # # mpl.plot_streamlines(lines, topology=topo, show=False)
    # mpl.mplshow()

    # topo_src = mvi.add_field(topo_fld, center='node')
    # mvi.plot_lines(mlab.pipeline, lines[::5], topo[::5], opacity=0.8,
    #                tube_radius=0.02)
    # mvi.plod_earth_3d(mlab.pipeline)
    # mlab.show()

    nsegs = 1  # keep from divding by 0 is no streamlines
    if lines is not None:
        nsegs = 0
        for line in lines:
            nsegs += len(line[0])

    t = t1 - t0
    print("total segments calculated: ", nsegs)
    print("time: {0:.4}s ... {1:.4}s/segment".format(t, t / float(nsegs)))

    return lines, topo_fld

def trace_numba(fld_bx, fld_by, fld_bz):
    B = field.scalar_fields_to_vector([fld_bx, fld_by, fld_bz], name="B_cc",
                                      _force_layout=field.LAYOUT_INTERLACED)
    topo_arr = np.empty(gsize, order='C', dtype='int')
    lines, topo = None, None
    t0 = time()
    nsegs = topo_numba.get_topo(B, topo_arr, x1, x2, y1, y2, z1, z2)
    t1 = time()
    topo_fld = vol.wrap_field(topo, name="CyTopo")
    return t1 - t0, nsegs, lines, topo_fld

def main():
    parser = argparse.ArgumentParser(description="Test xdmf")
    parser.add_argument("--show", "--plot", action="store_true")
    parser.add_argument('file', nargs="?", default=None)
    args = vutil.common_argparse(parser)

    # f3d = readers.load_file(_viscid_root + '/../../sample/sample.3df.xdmf')
    # b3d = f3d['b']
    # bx, by, bz = b3d.component_fields()  # pylint: disable=W0612

    if args.file is None:
        # args.file = "/Users/kmaynard/dev/work/cen4000/cen4000.3d.xdmf"
        # args.file = "/Users/kmaynard/dev/work/tmp/cen2000.3d.004045.xdmf"
        args.file = "~/dev/work/api_05_180_0.00_5e2.3d.007200.xdmf"
    f3d = readers.load_file(args.file)

    bx = f3d["bx"]
    by = f3d["by"]
    bz = f3d["bz"]

    profile = False

    print("Fortran...")
    if profile:
        cProfile.runctx("lines, topo_fort = trace_fortran(bx, by, bz)",
                        globals(), locals(), "topo_fort.prof")
        s = pstats.Stats("topo_fort.prof")
        s.strip_dirs().sort_stats("cumtime").print_stats(10)
    else:
        lines, topo_fort = trace_fortran(bx, by, bz)

    print("Cython...")
    if profile:
        cProfile.runctx("lines, topo_cy = trace_cython(bx, by, bz)",
                        globals(), locals(), "topo_cy.prof")
        s = pstats.Stats("topo_cy.prof")
        s.strip_dirs().sort_stats("cumtime").print_stats(15)
    else:
        lines, topo_cy = trace_cython(bx, by, bz)
        # print("Same? ",(np.ravel(topo_fort.data, order='K') ==
        #                 np.ravel(topo_cy.data, order='K')).all())

    # print("Numba...")
    # t, nsegs, lines, topo_nb = trace_numba(bx, by, bz)
    # print("total segments calculated: ", nsegs)
    # print("time: {0:.4}s ... {1:.4}s/segment".format(t, t / float(nsegs)))
    # print("Same? ",(np.ravel(topo_fort.data, order='K') ==
    #                 np.ravel(topo_nb.data, order='K')).all())

if __name__ == "__main__":
    main()

##
## EOF
##
