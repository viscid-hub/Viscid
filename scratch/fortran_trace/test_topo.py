#!/usr/bin/env python

from __future__ import print_function
import sys
import os
import cProfile
import pstats
from timeit import default_timer as time
import argparse
import logging

import numpy as np
from mayavi import mlab

_viscid_root = os.path.realpath(os.path.dirname(__file__) + '/../../src/viscid/') #pylint: disable=C0301
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)

import tracer
from viscid import vutil
from viscid import readers
from viscid import field
from viscid.plot import mpl
from viscid.plot import mvi
from viscid.calculator import streamline
from viscid.calculator import seed
import topo_numba

gsize = (8, 8, 8)
x1 = -10.0; x2 = -5.0 #pylint: disable=C0321
y1 = -5.0; y2 = 5.0 #pylint: disable=C0321
z1 = -5.0; z2 = 5.0 #pylint: disable=C0321
vol = seed.Volume((z1, y1, x1), (z2, y2, x2), gsize)

def trace_fortran(fld_bx, fld_by, fld_bz):
    gx, gy, gz = fld_bx.crds.get_crd(('xcc', 'ycc', 'zcc'))
    nz, ny, nx = fld_bx.shape

    bx_farr = np.array(np.ravel(fld_bx.data, order='K').reshape((nx, ny, nz), order="F")) #pylint: disable=C0301
    by_farr = np.array(np.ravel(fld_by.data, order='K').reshape((nx, ny, nz), order="F")) #pylint: disable=C0301
    bz_farr = np.array(np.ravel(fld_bz.data, order='K').reshape((nx, ny, nz), order="F")) #pylint: disable=C0301
    topo = np.zeros(gsize, order='F', dtype='int32')
    nsegs = np.zeros((1,), order='F', dtype='int32')

    logging.info((np.ravel(bx_farr, order='K') == np.ravel(fld_bx.data, order='K')).all()) #pylint: disable=C0301
    logging.info((np.ravel(by_farr, order='K') == np.ravel(fld_by.data, order='K')).all()) #pylint: disable=C0301
    logging.info((np.ravel(bz_farr, order='K') == np.ravel(fld_bz.data, order='K')).all()) #pylint: disable=C0301
    logging.info("bx_arr")
    logging.info(bx_farr.strides)
    logging.info(bx_farr.flags)
    logging.info("topo")
    logging.info(topo.strides)
    logging.info(topo.shape)
    
    t0 = time()
    tracer.get_topo(gx, gy, gz, bx_farr, by_farr, bz_farr, topo, 
                    x1, x2, y1, y2, z1, z2, nsegs)
    t1 = time()

    t = t1 - t0
    print("total segments calculated: ", nsegs)
    print("time: {0:.4}s ... {1:.4}s/segment".format(t, t / float(nsegs)))

    topo_fld = vol.wrap_field("Scalar", "FortTopo", topo)
    return None, topo_fld

def trace_cython(fld_bx, fld_by, fld_bz):
    # print("Cython...")
    B = field.scalar_fields_to_vector("B_cc", [fld_bx, fld_by, fld_bz],
                                      info={"force_layout": field.LAYOUT_INTERLACED})
    t0 = time()
    lines, topo = None, None
    lines, topo = streamline.streamlines(B, vol, ds0=0.02, ibound=3.7,
                            maxit=5000, output=streamline.OUTPUT_TOPOLOGY)
    t1 = time()

    # mpl.plot_streamlines(lines, show=True)
    # mpl.plot(topo_cy, "y=0", show=True)

    # b_src = mvi.field_to_source(B)
    # topo_src = mvi.field_to_source(topo_fld)
    # mvi.plot_lines(mlab.pipeline, lines, color=(0.0, 0.8, 0.0), tube_radius=0.05)
    # mvi.mlab_earth(mlab.pipeline)
    # mlab.show()

    nsegs = 1  # keep from divding by 0 is no streamlines
    if lines is not None:
        nsegs = 0
        for line in lines:
            nsegs += len(line[0])

    t = t1 - t0
    print("total segments calculated: ", nsegs)
    print("time: {0:.4}s ... {1:.4}s/segment".format(t, t / float(nsegs)))

    topo_fld = vol.wrap_field("Scalar", "CyTopo", topo)

    return lines, topo_fld

def trace_numba(fld_bx, fld_by, fld_bz):
    B = field.scalar_fields_to_vector("B_cc", [fld_bx, fld_by, fld_bz],
                                      info={"force_layout": field.LAYOUT_INTERLACED})
    topo_arr = np.empty(gsize, order='C', dtype='int')
    lines, topo = None, None
    t0 = time()
    nsegs = topo_numba.get_topo(B, topo_arr, x1, x2, y1, y2, z1, z2)
    t1 = time()
    topo_fld = vol.wrap_field("Scalar", "CyTopo", topo)
    return t1 - t0, nsegs, lines, topo_fld

def main():
    parser = argparse.ArgumentParser(description="Test xdmf")
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser) #pylint: disable=W0612

    # f3d = readers.load(_viscid_root + '/../../sample/sample.3df.xdmf')
    # b3d = f3d['b']
    # bx, by, bz = b3d.component_fields() #pylint: disable=W0612

    f3d = readers.load("/Users/kmaynard/dev/work/t1/t1.3df.004320.xdmf")
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
        print("Same? ",(np.ravel(topo_fort.data, order='K') == 
                        np.ravel(topo_cy.data, order='K')).all())        

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
