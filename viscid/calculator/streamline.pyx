# cython: boundscheck=False, wraparound=False, cdivision=True, profile=False

from __future__ import print_function
from timeit import default_timer as time
import logging

import numpy as np

from .. import field
from . import seed

###########
# cimports
cimport cython
# cimport numpy as cnp

from cycalc_util cimport *
from cycalc cimport *
from integrate cimport *
from streamline cimport *

EULER1 = 1
RK2 = 2
RK12 = 3
EULER1A = 4

DIR_FORWARD = 1
DIR_BACKWARD = 2
DIR_BOTH = 3

OUTPUT_STREAMLINES = 1
OUTPUT_TOPOLOGY = 2
OUTPUT_BOTH = 3

END_NONE = 0  # not ended yet
END_IBOUND_NORTH = 1  # unused
END_IBOUND_SOUTH = 2  # unused
END_IBOUND = 4
END_OBOUND = 8
END_OTHER = 16  # ??
END_MAXIT = 32
END_ZERO_LENGTH = 64
END_CYCLIC = 128

# these are not used, and are too specialized to be here
TOPOLOGY_INVALID = 1 # 1, 2, 3, 4, 9, 10, 11, 12, 15
TOPOLOGY_CLOSED = 7 # 5 (both N), 6 (both S), 7(both hemispheres)
TOPOLOGY_SW = 8
TOPOLOGY_OPEN_NORTH = 13
TOPOLOGY_OPEN_SOUTH = 14
TOPOLOGY_OTHER = 16 # >= 16


#####################
# now the good stuff

def streamlines(fld, seeds, *args, **kwargs):
    if not fld.layout == field.LAYOUT_INTERLACED:
        raise ValueError("Streamlines only written for interlaced data.")
    if fld.dim != 3:
        raise ValueError("Streamlines are only written in 3D.")
    if fld.center != "Cell":
        raise ValueError("Can only trace cell centered things...")
    
    dat = fld.data
    dtype = dat.dtype
    center = "Cell"
    crdz, crdy, crdx = fld.crds.get_crd(center=center)
    return _py_streamline(dtype, dat, crdz, crdy, crdx, seeds, center,
                          *args, **kwargs)

@cython.wraparound(True)
def _py_streamline(dtype, real_t[:,:,:,::1] v_mv,
                   real_t[:] crdz, real_t[:] crdy, real_t[:] crdx,
                   seeds, center, ds0=-1.0, ibound=0.0,
                   obound0=None, obound1=None, maxit=10000,
                   stream_dir=DIR_BOTH, output=OUTPUT_BOTH, method=EULER1,
                   tol_lo=1e-3, tol_hi=1e-2, fac_refine=0.75, fac_coarsen=2.0):
    """ Start calculating a streamline at x0
    stream_dir:  DIR_FORWARD, DIR_BACKWARD, DIR_BOTH
    ibound:      stop streamline if within inner_bound of the origin
                 ignored if 0
    obound0:     corner of box beyond which to stop streamline (smallest values)
    obound1:     corner of box beyond which to stop streamline (smallest values)
    ds0:         initial spatial step for the streamline """
    cdef:
        # cdefed versions of arguments
        real_t c_ds0 = ds0
        real_t c_ibound = ibound
        real_t c_obound0_carr[3]
        real_t c_obound1_carr[3]
        real_t[:] c_obound0 = c_obound0_carr
        real_t[:] c_obound1 = c_obound1_carr
        real_t[:] py_obound0
        real_t[:] py_obound1
        int c_stream_dir = stream_dir
        int c_maxit = maxit
        real_t c_tol_lo = tol_lo
        real_t c_tol_hi = tol_hi
        real_t c_fac_refine = fac_refine
        real_t c_fac_coarsen = fac_coarsen

        # just for c
        int (*integrate_func)(real_t[:,:,:,::1], real_t[:]*,
                      real_t*, real_t[:], real_t, real_t, real_t, real_t,
                      int[3])
        int i, j, it
        int i_stream
        int n_streams = seeds.n_points(center=center)
        int nprogress = max(n_streams / 50, 1)  # progerss at every 5%
        int nsegs = 0

        int ret   # return status of euler integrate
        int end_flags
        int done  # streamline has ended for some reason
        real_t px, py, pz
        int d  # direction of streamline 1 | -1
        real_t ds
        real_t rsq, distsq
        double t0inner, t1inner
        double tinner = 0.0

        real_t x0_carr[3]
        int line_ends_carr[2]
        real_t s_carr[3]
        int *start_inds = [0, 0, 0]

        int[:] topology_mv = None
        real_t[:,:,::1] line_mv = None
        real_t[:] x0_mv = x0_carr
        int[:] line_ends_mv = line_ends_carr
        real_t[:] s_mv = s_carr
    cdef real_t[:] *crds = [crdz, crdy, crdx]

    lines = None
    line_ndarr = None
    topology_ndarr = None

    if obound0 is None:
        c_obound0[0] = crds[0][0]
        c_obound0[1] = crds[1][0]
        c_obound0[2] = crds[2][0]
    else:
        py_obound0 = obound0
        c_obound0[...] = py_obound0

    if obound1 is None:
        c_obound1[0] = crds[0][-1]
        c_obound1[1] = crds[1][-1]
        c_obound1[2] = crds[2][-1]
    else:
        py_obound1 = obound1
        c_obound1[...] = py_obound1

    if c_ds0 <= 0.0:
        # FIXME: calculate something reasonable here
        c_ds0 = 0.01

    if method == EULER1:
        integrate_func = _c_euler1[real_t]
    elif method == RK2:
        integrate_func = _c_rk2[real_t]
    elif method == RK12:
        integrate_func = _c_rk12[real_t]
    elif method == EULER1A:
        integrate_func = _c_euler1a[real_t]
    else:
        raise ValueError("unknown integration method")

    if output & OUTPUT_STREAMLINES:
        line_ndarr = np.empty((2, 3, c_maxit), dtype=dtype)
        line_mv = line_ndarr
        lines = []
    if output & OUTPUT_TOPOLOGY:
        topology_ndarr = np.empty((n_streams,), dtype="i")
        topology_mv = topology_ndarr

    # first one is for timing, second for status
    # t0_all = time()
    t0 = time()

    for i_stream, seed_pt in enumerate(seeds.iter_points(center=center)):
        if i_stream % nprogress == 0:
            t1 = time()
            logging.info("Streamline {0} of {1}: {2}% done, {3:.03e}".format(i_stream,
                         n_streams, int(100.0 * i_stream / n_streams),
                         t1 - t0))
            t0 = time()

        x0_mv[0] = seed_pt[0]
        x0_mv[1] = seed_pt[1]
        x0_mv[2] = seed_pt[2]

        if line_mv is not None:
            line_mv[0, 0, c_maxit - 1] = x0_mv[0]
            line_mv[0, 1, c_maxit - 1] = x0_mv[1]
            line_mv[0, 2, c_maxit - 1] = x0_mv[2]
        line_ends_mv[0] = c_maxit - 2
        line_ends_mv[1] = 0
        end_flags = END_NONE

        for i, d in enumerate([-1, 1]):
            if d < 0 and not (c_stream_dir & DIR_BACKWARD):
                continue
            elif d > 0 and not (c_stream_dir & DIR_FORWARD):
                continue

            ds = d * c_ds0

            s_mv[0] = x0_mv[0]
            s_mv[1] = x0_mv[1]
            s_mv[2] = x0_mv[2]

            it = line_ends_mv[i]

            done = END_NONE
            while 0 <= it and it < c_maxit:
                nsegs += 1
                ret = integrate_func(v_mv, crds, &ds, s_mv,
                    c_tol_lo, c_tol_hi, c_fac_refine , c_fac_coarsen,
                    start_inds)
                # if i_stream == 0:
                #     print(s_mv[2], s_mv[1], s_mv[0])
                # ret is non 0 when |v_mv| == 0
                if ret != 0:
                    done = END_OTHER | END_ZERO_LENGTH
                    break

                if line_mv is not None:
                    line_mv[i, 0, it] = s_mv[0]
                    line_mv[i, 1, it] = s_mv[1]
                    line_mv[i, 2, it] = s_mv[2]
                it += d

                # end conditions
                rsq = s_mv[0]**2 + s_mv[1]**2 + s_mv[2]**2

                # hit the inner boundary
                if rsq <= c_ibound**2:
                    # print("inner boundary")
                    if s_mv[0] >= 0.0:
                        done = END_IBOUND | END_IBOUND_NORTH
                    else:
                        done = END_IBOUND | END_IBOUND_SOUTH
                    break

                for j from 0 <= j < 3:
                    # hit the outer boundary
                    if s_mv[j] <= c_obound0[j] or s_mv[j] >= c_obound1[j]:
                        # print("outer boundary")
                        done = END_OBOUND
                        break

                # if we are within 0.99 * ds0 of the initial position
                # distsq = (x0_mv[0] - s_mv[0])**2 + \
                #          (x0_mv[1] - s_mv[1])**2 + \
                #          (x0_mv[2] - s_mv[2])**2
                # if distsq < (0.99 * ds0)**2:
                #     # print("cyclic field line")
                #     done = END_OTHER | END_CYCLIC
                #     break

                if done:
                    break

            if done == END_NONE:
                done = END_OTHER | END_MAXIT

            line_ends_mv[i] = it
            end_flags |= done

        # now we have forward and background traces, process this streamline
        if line_mv is not None:
            # if i_stream == 0:
            #     print("myzero", line_ends_mv[0], line_ends_mv[1], end_flags)
            line_cat = np.concatenate((line_mv[0, :, line_ends_mv[0] + 1:], 
                                       line_mv[1, :, :line_ends_mv[1]]), axis=1)
            lines.append(line_cat)

        if topology_mv is not None:
            topology_mv[i_stream] = end_flags
            # logging.info("{0}: {1}, [{2}, {3}]".format(i_stream, end_flags, 
            #                         line_ends_mv[0], line_ends_mv[1]))

    # for timing
    # t1_all = time()
    # t = t1_all - t0_all
    # print("=> in cython time: {0:.03e} s  {1:.03e} s/seg".format(t, t / nsegs))

    return lines, topology_ndarr

##
## EOF
##
