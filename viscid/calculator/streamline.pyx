# cython: boundscheck=False, wraparound=False, cdivision=True, profile=False
r"""All streamlines all the time

Calculate streamlines of a vector field on as many processors your
heart desires. The only function you need here is :meth:`streamlines`

Note:
    When nr_procs > 1, fields are shared to child processes using
    a global variable so that on \*nix there is no need to picklel and
    copy the entire field. This will only work on \*nix, and I have
    absolutely no idea what will happen on Windows.
"""

from __future__ import print_function
from timeit import default_timer as time
from logging import info, warning
from multiprocessing import Pool, cpu_count
from contextlib import closing
from itertools import islice, repeat
try:
    from itertools import izip
except ImportError:
    izip = zip

import numpy as np

from viscid import parallel
from viscid import field
from viscid.calculator import seed

###########
# cimports
cimport cython
from libc.math cimport fabs
# cimport numpy as cnp

from cycalc_util cimport *
from cycalc cimport *
from integrate cimport *
from streamline cimport *

EULER1 = 1  # euler1 non-adaptive
RK2 = 2  # rk2 non-adaptive
RK12 = 3  # euler1 + rk2 adaptive
EULER1A = 4  # euler 1st order adaptive
METHOD = {"euler": EULER1, "euler1": EULER1, "rk2": RK2, "rk12": RK12,
          "euler1a": EULER1A}

DIR_FORWARD = 1
DIR_BACKWARD = 2
DIR_BOTH = 3  # = DIR_FORWARD | DIR_BACKWARD

OUTPUT_STREAMLINES = 1
OUTPUT_TOPOLOGY = 2
OUTPUT_BOTH = 3  # = OUTPUT_STREAMLINES | OUTPUT_TOPOLOGY

# topology will be 1+ of these flags binary or-ed together
#                                bit #   4 2 0 8 6 4 2 0  Notes         bit
END_NONE = 0                         # 0b000000000000000 not ended yet    X
END_IBOUND = 1                       # 0b000000000000001                  0
END_IBOUND_NORTH = 2 | END_IBOUND    # 0b000000000000011  == 3            1
END_IBOUND_SOUTH = 4 | END_IBOUND    # 0b000000000000101  == 5            2
END_OBOUND = 8                       # 0b000000000001000                  3
END_OBOUND_XL = 16 | END_OBOUND      # 0b000000000011000  == 24           4
END_OBOUND_XH = 32 | END_OBOUND      # 0b000000000101000  == 40           5
END_OBOUND_YL = 64 | END_OBOUND      # 0b000000001001000  == 72           6
END_OBOUND_YH = 128 | END_OBOUND     # 0b000000010001000  == 136          7
END_OBOUND_ZL = 256 | END_OBOUND     # 0b000000100001000  == 264          8
END_OBOUND_ZH = 512 | END_OBOUND     # 0b000001000001000  == 520          9
END_CYCLIC = 1024                    # 0b000010000000000  !!NOT USED!!   10
END_OTHER = 2048                     # 0b000100000000000                 11
END_MAXIT = 4096 | END_OTHER         # 0b001100000000000  == 6144        12
END_MAX_LENGTH = 8192 | END_OTHER    # 0b010100000000000  == 10240       13
END_ZERO_LENGTH = 16384 | END_OTHER  # 0b100100000000000  == 18432       14

# ok, this is over complicated, but the goal was to or the topology value
# with its neighbors to find a separator line... To this end, or-ing two
# END_* values doesn't help, so before streamlines returns, it will
# replace the numbers that mean closed / open with powers of 2, that way
# we end up with topology as an actual bit mask
TOPOLOGY_MS_NONE = 0  # no translation needed
TOPOLOGY_MS_CLOSED = 1  # translated from 5, 6, 7(4|5|6)
TOPOLOGY_MS_OPEN_NORTH = 2  # translated from 13 (8|5)
TOPOLOGY_MS_OPEN_SOUTH = 4  # translated from 14 (8|6)
TOPOLOGY_MS_SW = 8  # no translation needed
# TOPOLOGY_MS_CYCLIC = 16  # no translation needed

# these are set if there is a pool of workers doing streamlines, they are
# always set back to None when the streamlines are done
# they need to be global so that the memory is shared with subprocesses
_global_dat = None
_global_crds = None

#####################
# now the good stuff
def streamlines(fld, seed, nr_procs=1, force_parallel=False,
                nr_chunks_factor=1, **kwargs):
    r"""Trace streamlines

    Parameters:
        fld: Some vector field
        seed: can be a Seeds instance or a Coordinates instance, or
            anything that exposes an iter_points method
        nr_procs: how many processes to calc streamlines on (>1 only
            works on \*nix systems)
        force_parallel: always calc streamlines in a separate process,
            even if nr_procs == 1
        nr_chunks_factor: If streamlines are really unbalanced in
            length, try bumping this up
        kwargs: more arguments for streamlines, described below

    Keyword Arguments:
        ds0 (float): initial spatial step for streamlines (if 0.0, it
            will be half the minimum d[xyz])
        ibound (float): Inner boundary as distance from (0, 0, 0)
        obound0 (array-like): lower corner of outer boundary (z, y, x)
        obound1 (array-like): upper corner of outer boundary (z, y, x)
        maxit (int): maximum number of line segments
        max_length (float): maximum streamline length
        stream_dir (int): one of DIR_FORWARD, DIR_BACKWARD, DIR_BOTH
        output (int): which output to provide, one of
            OUTPUT_STREAMLINE, OUTPUT_TOPOLOGY, or OUTPUT_BOTH
        method (int): integrator, one of EULER1, EULER1a (adaptive),
            RK2, RK12 (adaptive)
        tol_lo (float): lower acuracy tolerance for adaptive
            integrators. More acurate than, this ds goes up.
        tol_hi (float): upper acuracy tolerance for adaptive
            integrators. Less acurate than, this ds goes down.
        fac_refine (float): When refining ds, ds \*= fac_refine
        fac_coarsen (float): When coarsening ds, ds \*= fac_coarsen
        smallest_step (float): smallest spatial step
        largest_step (float): largest spatial step
        topo_style (str): how to map end point bitmask to a topology.
            'msphere' to map to ``TOPOLOGY_MS_*``, or 'generic' to
            leave topology as a bitmask of ``END_*``

    Returns:
        (lines, topo), either can be ``None`` depending on ``output``

        * lines, list of nr_streams ndarrays, each ndarray has shape
          (3, nr_points_in_stream) the nr_points_in_stream can be
          different for each line
        * topo, ndarray with shape (nr_streams,) of topology
          bitmask with values depending on the topo_style

    """
    if not fld.layout == field.LAYOUT_INTERLACED:
        raise ValueError("Streamlines only written for interlaced data.")
    if fld.nr_sdims != 3 or fld.nr_comps != 3:
        raise ValueError("Streamlines are only written in 3D.")
    if not fld.iscentered("Cell"):
        raise ValueError("Can only trace cell centered things...")

    dat = fld.data
    dtype = dat.dtype
    center = "cell"
    crdz, crdy, crdx = fld.get_crds_cc()
    nr_streams = seed.nr_points(center=center)
    kwargs["center"] = center

    if nr_procs == "all" or nr_procs == "auto":
        nr_procs = cpu_count()

    if nr_procs == 1 and not force_parallel:
        lines, topo = _py_streamline(dtype, dat, crdz, crdy, crdx, nr_streams,
                                     seed, **kwargs)
    else:
        # wrap the above around some parallelizing logic that is way more cumbersome
        # than it needs to be
        nr_chunks = nr_chunks_factor * nr_procs
        seed_slices = parallel.chunk_interslices(nr_chunks)  # every nr_chunks seed points
        # seed_slices = parallel.chunk_slices(nr_streams, nr_chunks)  # contiguous chunks
        chunk_sizes = parallel.chunk_sizes(nr_streams, nr_chunks)

        global _global_dat, _global_crds
        # if they were already set, then some other process sharing this
        # global memory is doing streamlines
        if _global_dat is not None or _global_crds is not None:
            raise RuntimeError("Another process is doing streamlines in this "
                               "global memory space")
        _global_dat = dat
        _global_crds = [crdz, crdy, crdx]
        grid_iter = izip(chunk_sizes, repeat(seed), seed_slices)
        args = izip(grid_iter, repeat(kwargs))

        with closing(Pool(nr_procs)) as p:
            r = p.map_async(_do_streamline_star, args).get(1e8)
        p.join()

        _global_dat = None
        _global_crds = None

        # rearrange the output to be the exact same as if we just called
        # _py_streamline straight up (like for nr_procs == 1)
        if r[0][0] is not None:
            lines = np.empty((nr_streams,), dtype=np.ndarray)  # [None] * nr_streams
            for i in range(nr_chunks):
                lines[slice(*seed_slices[i])] = r[i][0]
        else:
            lines = None

        if r[0][1] is not None:
            topo = np.empty((nr_streams,), dtype=r[0][1].dtype)
            for i in range(nr_chunks):
                topo[slice(*seed_slices[i])] = r[i][1]
        else:
            topo = None
    return lines, topo

@cython.wraparound(True)
def _do_streamline_star(args):
    ret = _py_streamline(_global_dat.dtype, _global_dat,
                         _global_crds[0], _global_crds[1], _global_crds[2],
                         *(args[0]), **(args[1]))
    return ret

@cython.wraparound(True)
def _py_streamline(dtype, real_t[:,:,:,::1] v_mv, crdz_in, crdy_in, crdx_in,
                   int nr_streams, seed, seed_slice=(None, ), center="Cell",
                   ds0=0.0, ibound=0.0, obound0=None, obound1=None,
                   maxit=10000, max_length=1e30, stream_dir=DIR_BOTH,
                   output=OUTPUT_BOTH, method=EULER1, tol_lo=1e-3, tol_hi=1e-2,
                   fac_refine=0.5, fac_coarsen=1.25, smallest_step=1e-4,
                   largest_step=1e2, topo_style="msphere"):
    r""" Start calculating a streamline at x0

    Parameters:
        dtype: anything to describe Numpy dtype of v_mv
        v_mv (typed memory view): some array
        crdz_in (ndarray): z coorinates, same centering as ``center``
        crdy_in (ndarray): z coorinates, same centering as ``center``
        crdx_in (ndarray): z coorinates, same centering as ``center``
        nr_streams (int):
        seed: something with an iter_points method
            (:class:`viscid.calculator.seed.Seed`,
            :class:`viscid.coordinate.Coordinates`, etc.)
        seed_slice (tuple): arguments for slice, (start, stop, [step])
        center (str): "Cell" | "Node"
        ds0 (float): initial spatial step for streamlines (if 0.0, it
            will be half the minimum d[xyz])
        ibound (float): Inner boundary as distance from (0, 0, 0)
        obound0 (array-like): lower corner of outer boundary (z, y, x)
        obound1 (array-like): upper corner of outer boundary (z, y, x)
        maxit (int): maximum number of line segments
        max_length (float): maximum streamline length
        stream_dir (int): one of DIR_FORWARD, DIR_BACKWARD, DIR_BOTH
        output (int): which output to provide, one of
            OUTPUT_STREAMLINE, OUTPUT_TOPOLOGY, or OUTPUT_BOTH
        method (int): integrator, one of EULER1, EULER1a (adaptive),
            RK2, RK12 (adaptive)
        tol_lo (float): lower acuracy tolerance for adaptive
            integrators. More acurate than, this ds goes up.
        tol_hi (float): upper acuracy tolerance for adaptive
            integrators. Less acurate than, this ds goes down.
        fac_refine (float): When refining ds, ds \*= fac_refine
        fac_coarsen (float): When coarsening ds, ds \*= fac_coarsen
        smallest_step (float): smallest spatial step
        largest_step (float): largest spatial step
        topo_style (str): how to map end point bitmask to a topology.
            'msphere' to map to ``TOPOLOGY_MS_*``, or 'generic' to
            leave topology as a bitmask of ``END_*``

    Returns:
        (lines, topo), either can be ``None`` depending on ``output``
        lines: list of nr_streams ndarrays, each ndarray has shape
            (3, nr_points_in_stream) the nr_points_in_stream can be
            different for each line
        topo: is an ndarray with shape (nr_streams,) of topology
            bitmask with values depending on the topo_style
    """
    cdef:
        # cdefed versions of arguments
        real_t[:] crdz = crdz_in
        real_t[:] crdy = crdy_in
        real_t[:] crdx = crdx_in
        real_t c_ds0 = ds0
        real_t c_ibound = ibound
        real_t c_obound0_carr[3]
        real_t c_obound1_carr[3]
        real_t min_dx[3]
        real_t[:] c_obound0 = c_obound0_carr
        real_t[:] c_obound1 = c_obound1_carr
        real_t[:] py_obound0
        real_t[:] py_obound1
        int c_stream_dir = stream_dir
        int c_maxit = maxit
        real_t c_max_length = max_length
        real_t c_tol_lo = tol_lo
        real_t c_tol_hi = tol_hi
        real_t c_fac_refine = fac_refine
        real_t c_fac_coarsen = fac_coarsen
        real_t c_smallest_step = smallest_step
        real_t c_largest_step = largest_step

        # just for c
        int (*integrate_func)(real_t[:,:,:,::1], real_t[:]*,
                real_t*, real_t[:], real_t, real_t, real_t, real_t,
                real_t, real_t, int[3]) except -1
        # int (*classify_endpoint)(real_t[3], real_t, real_t,
        #         real_t[3], real_t[3],
        #         real_t, real_t, real_t[3])
        int (*end_flags_to_topology)(int end_flags)
        int i, j, it
        int i_stream
        int nprogress = max(nr_streams / 50, 1)  # progeress at every 5%
        int nr_segs = 0

        int ret   # return status of euler integrate
        int end_flags
        int done  # streamline has ended for some reason
        real_t stream_length
        real_t px, py, pz
        int d  # direction of streamline 1 | -1
        real_t pre_ds
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
    crds_in = [crdz_in, crdy_in, crdx_in]

    lines = None
    line_ndarr = None
    topology_ndarr = None

    if c_ds0 == 0.0:
        find_ds0 = True
        c_ds0 = 1e30

    if obound0 is not None:
        py_obound0 = obound0
        c_obound0[...] = py_obound0

    if obound1 is not None:
        py_obound1 = obound1
        c_obound1[...] = py_obound1

    # these hoops are required for processing 2d fields
    for i in range(3):
        if len(crds[i]) == 1:
            if obound0 is None:
                c_obound0[i] = -1e30
            if obound1 is None:
                c_obound1[i] = 1e30
        else:
            if obound0 is None:
                c_obound0[i] = crds[i][0]
            if obound1 is None:
                c_obound1[i] = crds[i][-1]
            if find_ds0:
                c_ds0 = np.min([c_ds0, np.min(crds_in[i][1:] - crds_in[i][:-1])])

    if find_ds0:
        c_ds0 *= 0.5

    # which integrator are we using?
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

    # determine which functions to use for understanding endpoints
    # and
    if topo_style == "msphere":
        # classify_endpoint = classify_endpoint_msphere[real_t]
        end_flags_to_topology = end_flags_to_topology_msphere
    else:
        # classify_endpoint = classify_endpoint_generic[real_t]
        end_flags_to_topology = end_flags_to_topology_generic

    # establish arrays for output
    if output & OUTPUT_STREAMLINES:
        # 2 (0=backward, 1=forward), 3 z,y,x, c_maxit points in the line
        line_ndarr = np.empty((2, 3, c_maxit), dtype=dtype)
        line_mv = line_ndarr
        lines = []
    if output & OUTPUT_TOPOLOGY:
        topology_ndarr = np.empty((nr_streams,), dtype="i")
        topology_mv = topology_ndarr

    # first one is for timing, second for status
    t0_all = time()
    t0 = time()

    seed_iter = islice(seed.iter_points(center=center), *seed_slice)
    for i_stream, seed_pt in enumerate(seed_iter):
        if i_stream % nprogress == 0:
            t1 = time()
            info("Streamline {0} of {1}: {2}% done, {3:.03e}".format(i_stream,
                         nr_streams, int(100.0 * i_stream / nr_streams),
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
            stream_length = 0.0

            s_mv[0] = x0_mv[0]
            s_mv[1] = x0_mv[1]
            s_mv[2] = x0_mv[2]

            it = line_ends_mv[i]

            done = END_NONE
            while 0 <= it and it < c_maxit:
                nr_segs += 1
                pre_ds = fabs(ds)

                ret = integrate_func(v_mv, crds, &ds, s_mv,
                    c_tol_lo, c_tol_hi, c_fac_refine , c_fac_coarsen,
                    c_smallest_step, c_largest_step, start_inds)

                if fabs(ds) >= pre_ds:
                    stream_length += pre_ds
                else:
                    stream_length += fabs(ds)

                # if i_stream == 0:
                #     print(s_mv[2], s_mv[1], s_mv[0])
                # ret is non 0 when |v_mv| == 0
                if ret != 0:
                    done = END_ZERO_LENGTH
                    break

                if line_mv is not None:
                    line_mv[i, 0, it] = s_mv[0]
                    line_mv[i, 1, it] = s_mv[1]
                    line_mv[i, 2, it] = s_mv[2]
                it += d

                # end conditions
                done = classify_endpoint(s_carr, stream_length, c_ibound,
                                         c_obound0_carr, c_obound1_carr,
                                         c_max_length, ds, x0_carr)
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
            topology_mv[i_stream] = end_flags_to_topology(end_flags)

    # for timing
    # t1_all = time()
    # t = t1_all - t0_all
    # print("=> in cython nr_segments: {0:.05e}".format(nr_segs))
    # print("=> in cython time: {0:.03f}s {1:.03e}s/seg".format(t, t / nr_segs))

    return lines, topology_ndarr

cdef inline int classify_endpoint(real_t pt[3], real_t length, real_t ibound,
                           real_t obound0[3], real_t obound1[3],
                           real_t max_length, real_t ds, real_t pt0[3]):
    cdef int done = END_NONE
    cdef real_t rsq = pt[0]**2 + pt[1]**2 + pt[2]**2

    # print(pt[0], obound0[0], obound0[0])

    if rsq < ibound**2:
        if pt[0] >= 0.0:
            done = END_IBOUND_NORTH
        else:
            done = END_IBOUND_SOUTH
    elif pt[0] < obound0[0]:
        done = END_OBOUND_ZL
    elif pt[1] < obound0[1]:
        done = END_OBOUND_YL
    elif pt[2] < obound0[2]:
        done = END_OBOUND_XL
    elif pt[0] > obound1[0]:
        done = END_OBOUND_ZH
    elif pt[1] > obound1[1]:
        done = END_OBOUND_YH
    elif pt[2] > obound1[2]:
        done = END_OBOUND_XH
    elif length > max_length:
        done = END_MAX_LENGTH

    # if we are within 0.05 * ds of the initial position
    # distsq = (pt0[0] - pt[0])**2 + \
    #          (pt0[1] - pt[1])**2 + \
    #          (pt0[2] - pt[2])**2
    # if distsq < (0.05 * ds)**2:
    #     # print("cyclic field line")
    #     done = END_CYCLIC
    #     break

    return done

cdef int end_flags_to_topology_msphere(int end_flags):
    cdef int topo = 0
    cdef int mask_open_north = END_IBOUND_NORTH | END_OBOUND
    cdef int mask_open_south = END_IBOUND_SOUTH | END_OBOUND

    # order of these if statements matters!
    if (end_flags & END_OTHER):
        topo = end_flags
    # elif (topo & END_CYCLIC):
    #     return TOPOLOGY_MS_CYCLYC
    elif end_flags & (mask_open_north) == mask_open_north:
        topo = TOPOLOGY_MS_OPEN_NORTH
    elif end_flags & (mask_open_south) == mask_open_south:
        topo = TOPOLOGY_MS_OPEN_SOUTH
    elif end_flags == 3 or end_flags == 5 or end_flags == 7:
        topo = TOPOLOGY_MS_CLOSED
    else:
        topo = TOPOLOGY_MS_SW

    return topo

cdef int end_flags_to_topology_generic(int end_flags):
    return end_flags

##
## EOF
##
