# cython: boundscheck=False, wraparound=False, cdivision=True, profile=False
# cython: emit_code_comments=False

r"""All streamlines all the time

Calculate streamlines of a vector field on as many processors as your
heart desires. The only function you need here is :meth:`streamlines`;
everything else is for fused types (templates) or other cython
performance considerations.

Note:
    When nr_procs > 1, fields are shared to child processes using
    a global variable so that on \*nix there is no need to picklel and
    copy the entire field. This will only work on \*nix, and I have
    absolutely no idea what will happen on Windows.
"""
# NOTE: this take a minute to compile on account of _py_streamline makes way
#       WAY too many fused copies of itself, but i see no way to tell cython
#       which fused declarations are valid... the compiled version is like
#       2MB, that's rediculous

from __future__ import print_function
# from timeit import default_timer as time
from contextlib import closing
from itertools import islice, repeat
import os

import numpy as np

import viscid
from viscid import parallel
from viscid.seed import to_seeds
from viscid.compat import izip

###########
# cimports
cimport cython
from libc.math cimport fabs, NAN
from libc.time cimport time_t, time, clock_t, clock, CLOCKS_PER_SEC
cimport numpy as cnp

from viscid.cython.cyfield cimport MAX_FLOAT, real_t
from viscid.cython.cyamr cimport FusedAMRField, make_cyamrfield, activate_patch
from viscid.cython.cyfield cimport CyField, FusedField, make_cyfield
from viscid.cython.integrate cimport _c_euler1, _c_rk2, _c_rk12, _c_euler1a


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

# IMPORTANT! If TOPOLOGY_MS_* values change, make sure to also change the
# values in viscid/cython/__init__.py since those are used if the cython
# code is not built

# ok, this is over complicated, but the goal was to or the topology value
# with its neighbors to find a separator line... To this end, or-ing two
# _C_END_* values doesn't help, so before streamlines returns, it will
# replace the numbers that mean closed / open with powers of 2, that way
# we end up with topology as an actual bit mask
TOPOLOGY_MS_NONE = 0  # no translation needed
TOPOLOGY_MS_CLOSED = 1  # translated from 5, 6, 7(4|5|6)
TOPOLOGY_MS_OPEN_NORTH = 2  # translated from 13 (8|5)
TOPOLOGY_MS_OPEN_SOUTH = 4  # translated from 14 (8|6)
TOPOLOGY_MS_SW = 8  # no translation needed
# TOPOLOGY_MS_CYCLIC = 16  # no translation needed

TOPOLOGY_MS_SEPARATOR = (TOPOLOGY_MS_CLOSED | TOPOLOGY_MS_OPEN_NORTH |
                         TOPOLOGY_MS_OPEN_SOUTH | TOPOLOGY_MS_SW)

# ok, typing these masks gives a very, very small performance boost, but I
# guess there's no reason not to... just have to remember to add new values
# Note: DIR_*, OUTPUT_*, and the integrator constants provide 0 performance
#       benefit when typed
cdef:
    int _C_DIR_FORWARD = DIR_FORWARD
    int _C_DIR_BACKWARD = DIR_BACKWARD
    int _C_DIR_BOTH = DIR_BOTH
    int _C_OUTPUT_STREAMLINES = OUTPUT_STREAMLINES
    int _C_OUTPUT_TOPOLOGY = OUTPUT_TOPOLOGY
    int _C_OUTPUT_BOTH = OUTPUT_BOTH
    # end bitmask
    int _C_END_NONE = END_NONE
    int _C_END_IBOUND = END_IBOUND
    int _C_END_IBOUND_NORTH = END_IBOUND_NORTH
    int _C_END_IBOUND_SOUTH = END_IBOUND_SOUTH
    int _C_END_OBOUND = END_OBOUND
    int _C_END_OBOUND_XL = END_OBOUND_XL
    int _C_END_OBOUND_XH = END_OBOUND_XH
    int _C_END_OBOUND_YL = END_OBOUND_YL
    int _C_END_OBOUND_YH = END_OBOUND_YH
    int _C_END_OBOUND_ZL = END_OBOUND_ZL
    int _C_END_OBOUND_ZH = END_OBOUND_ZH
    int _C_END_CYCLIC = END_CYCLIC
    int _C_END_OTHER = END_OTHER
    int _C_END_MAXIT = END_MAXIT
    int _C_END_MAX_LENGTH = END_MAX_LENGTH
    int _C_END_ZERO_LENGTH = END_ZERO_LENGTH
    # topology bitmask
    int _C_TOPOLOGY_MS_NONE = TOPOLOGY_MS_NONE
    int _C_TOPOLOGY_MS_CLOSED = TOPOLOGY_MS_CLOSED
    int _C_TOPOLOGY_MS_OPEN_NORTH = TOPOLOGY_MS_OPEN_NORTH
    int _C_TOPOLOGY_MS_OPEN_SOUTH = TOPOLOGY_MS_OPEN_SOUTH
    int _C_TOPOLOGY_MS_SW = TOPOLOGY_MS_SW

# these are set if there is a pool of workers doing streamlines, they are
# always set back to None when the streamlines are done
# they need to be global so that the memory is shared with subprocesses
_global_fld = None

#####################
# now the good stuff
def calc_streamlines(vfield, seed, nr_procs=1, force_subprocess=False,
                     threads=True, chunk_factor=1, wrap=True, **kwargs):
    r"""Trace streamlines

    Args:
        vfield: A VectorField with 3 components
        seed: can be a Seeds instance or a Coordinates instance, or
            anything that exposes an iter_points method
        nr_procs: how many processes for streamlines (>1 only works on
            \*nix systems)
        force_subprocess (bool): always calc streamlines in a separate
            process, even if nr_procs == 1
        chunk_factor (int): Valid range is [1, nr_seeds // nr_procs]. 1
            indicates a single chunk per process; use this for
            perfectly balanced streamlines. nr_seeds // nr_procs
            indicates one chunk per seed; this solves load balancing,
            but the overhead of assembling results makes this
            undesirable. Start with 1 and bump this up if load
            balancing is still an issue.
        wrap (bool): if true, then call seed.wrap_field on topology
        **kwargs: more arguments for streamlines

    Keyword Arguments:
        ds0 (float): initial spatial step for streamlines (if 0.0, it
            will be half the minimum d[xyz])
        ibound (float): Inner boundary as distance from (0, 0, 0)
        obound0 (array-like): lower corner of outer boundary (x, y, z)
        obound1 (array-like): upper corner of outer boundary (x, y, z)
        maxit (int): maximum number of line segments
        max_length (float): maximum streamline length
        stream_dir (int): one of DIR_FORWARD, DIR_BACKWARD, DIR_BOTH
        output (int): which output to provide, one of
            OUTPUT_STREAMLINE, OUTPUT_TOPOLOGY, or OUTPUT_BOTH
        method (int): integrator, one of EULER1, EULER1a (adaptive),
            RK2, RK12 (adaptive)
        tol_lo (float): lower acuracy tolerance for adaptive
            integrators. More acurate than this, ds goes up.
        tol_hi (float): upper acuracy tolerance for adaptive
            integrators. Less acurate than this, ds goes down.
        fac_refine (float): When refining ds \*= fac_refine
        fac_coarsen (float): When coarsening ds \*= fac_coarsen
        smallest_step (float): smallest spatial step
        largest_step (float): largest spatial step
        topo_style (str): how to map end point bitmask to a topology.
            'msphere' means map to ``TOPOLOGY_MS_*`` and 'generic'
            means leave topology as a bitmask of ``END_*``

    Returns:
        (lines, topo), either can be ``None`` depending on ``output``

        * `lines`: list of nr_streams ndarrays, each ndarray has shape
          (3, nr_points_in_stream). The nr_points_in_stream can be
          different for each line
        * `topo`: ndarray with shape (nr_streams,) of topology
          bitmask with values depending on the topo_style
    """
    # if not fld.layout == field.LAYOUT_INTERLACED:
    #     raise ValueError("Streamlines only written for interlaced data.")
    if vfield.nr_sdims != 3 or vfield.nr_comps != 3:
        raise ValueError("Streamlines are only written in 3D.")

    # fld = make_cyfield(vfield)
    # fld = make_cyfield(vfield.as_cell_centered())
    fld = make_cyamrfield(vfield)

    seed = to_seeds(seed)

    seed_center = seed.center if hasattr(seed, 'center') else vfield.center
    if seed_center.lower() in ('face', 'edge'):
        seed_center = 'cell'
    kwargs['seed_center'] = seed_center

    nr_streams = seed.get_nr_points(center=seed_center)
    nr_procs = parallel.sanitize_nr_procs(nr_procs)

    nr_chunks = max(1, min(chunk_factor * nr_procs, nr_streams))
    seed_slices = parallel.chunk_interslices(nr_chunks)  # every nr_chunks seed points
    # seed_slices = parallel.chunk_slices(nr_streams, nr_chunks)  # contiguous chunks
    chunk_sizes = parallel.chunk_sizes(nr_streams, nr_chunks)

    global _global_fld
    if _global_fld is not None:
        raise RuntimeError("Another process is doing streamlines in this "
                           "global memory space")
    _global_fld = fld
    grid_iter = izip(chunk_sizes, repeat(seed), seed_slices)
    r = parallel.map(nr_procs, _do_streamline_star, grid_iter, args_kw=kwargs,
                     threads=threads, force_subprocess=force_subprocess)
    _global_fld = None

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

        if wrap:
            topo = seed.wrap_field(topo, name="Topology")
    else:
        topo = None

    return lines, topo

# for legacy code
streamlines = calc_streamlines

@cython.wraparound(True)
def _do_streamline_star(*args, **kwargs):
    """Wrapper for running in parallel using :py:module`Viscid.parallel`'s
    subprocessing helpers
    """
    # print("_global_fld type::", type(_global_fld))
    gfld = _global_fld
    return _streamline_fused_wrapper(gfld, *args, **kwargs)

def _streamline_fused_wrapper(FusedAMRField fld, int nr_streams, seed,
                              seed_slice=(None,), seed_center="cell", **kwargs):
    """Wrapper to make sure type specialization is same as fld's dtypes"""
    # cdef str amr_type = cython.typeof(amrfld)
    # # FIXME: **THUNDER-HACK** trim off the AMR part of the type name
    # #        This might be the only way to type-specialize AMR streamlines
    # cdef str fld_type = amr_type[3:]  #
    # cdef int nbits = 8 * int(fld_type.split("_")[1][1])
    # cdef str real_type = "float{0}_t".format(nbits)
    func = _py_streamline[cython.typeof(fld), cython.typeof(fld.active_patch),
                          cython.typeof(fld.min_dx)]
    return func(fld, fld.active_patch, nr_streams, seed, seed_slice=seed_slice,
                seed_center=seed_center, **kwargs)

@cython.wraparound(True)
def _py_streamline(FusedAMRField amrfld, FusedField active_patch,
                   int nr_streams, seed, seed_slice=(None, ),
                   real_t ds0=0.0, real_t ibound=0.0, obound0=None, obound1=None,
                   int stream_dir=_C_DIR_BOTH, int output=_C_OUTPUT_BOTH,
                   int method=EULER1,
                   int maxit=90000, real_t max_length=1e30,
                   real_t tol_lo=1e-3, real_t tol_hi=1e-2,
                   real_t fac_refine=0.5, real_t fac_coarsen=1.25,
                   real_t smallest_step=1e-4, real_t largest_step=1e2,
                   str topo_style="msphere", str seed_center="cell"):
    r""" Start calculating a streamline at x0

    Args:
        amrfld (FusedAMRField): Some Vector Field with 3 components
        active_patch (FusedField): amrfld.active_patch, needed for its
            ctype b/c integrate_funcs are cdef'd for performance
        nr_streams (int):
        seed: can be a Seeds instance or a Coordinates instance, or
            anything that exposes an iter_points method
        seed_slice (tuple): arguments for slice, (start, stop, [step])


    See Also:
        * :py:func:`streamline.streamlines`: Keyword Arguments are
            documented here.

    Returns:
        (lines, topo), either can be ``None`` depending on ``output``

        * `lines`: list of nr_streams ndarrays, each ndarray has shape
          (3, nr_points_in_stream). The nr_points_in_stream can be
          different for each line
        * `topo`: ndarray with shape (nr_streams,) of topology
          bitmask with values depending on the topo_style
    """
    cdef:
        # cdefed versions of arguments
        real_t c_obound0[3]
        real_t c_obound1[3]
        real_t min_dx[3]

        int nr_patch = amrfld.active_patch_index
        FusedField patch = amrfld.active_patch

        # just for c
        int (*integrate_func)(FusedField fld, real_t x[3], real_t *ds,
                              real_t tol_lo, real_t tol_hi,
                              real_t fac_refine, real_t fac_coarsen,
                              real_t smallest_step, real_t largest_step,
                              real_t vscale[3], int cached_idx3[3]) nogil except -1
        int (*end_flags_to_topology)(int _end_flags) nogil

        int i, j, it
        int n, nnc
        int i_stream
        int nprogress = max(nr_streams / 50, 1)  # progeress at every 5%
        int nr_segs = 0
        int maxit2

        int ret  # return status of euler integrate
        int end_flags
        int done  # streamline has ended for some reason
        real_t stream_length
        int d  # direction of streamline 1 | -1
        real_t pre_ds
        real_t ds
        real_t rsq, distsq

        int _dir_d[2]
        real_t x0[3]
        real_t s[3]
        real_t vscale[3]
        int line_ends[2]

        int[:] topology_mv = None
        real_t[:, ::1] line_mv = None
        real_t[:] dx
        real_t[:, ::1] seed_pts

        int cached_idx3[3]

    maxit2 = 2 * maxit + 1
    _dir_d[:] = [-1, 1]
    cached_idx3[:] = [0, 0, 0]

    lines = None
    line_ndarr = None
    topology_ndarr = None

    # set up ds0 and c_obound from the limits of fld if they're not already
    # given
    if obound0 is not None:
        py_obound0 = np.array(obound0, dtype=amrfld.crd_dtype)
        for i in range(3):
            c_obound0[i] = py_obound0[i]

    if obound1 is not None:
        py_obound1 = np.array(obound1, dtype=amrfld.crd_dtype)
        for i in range(3):
            c_obound1[i] = py_obound1[i]

    for i in range(3):
        # n = fld.n[i]
        # nnc = fld.nr_nodes[i]
        if active_patch.n[i] == 1:
            # these hoops are required for processing 2d fields
            if obound0 is None:
                c_obound0[i] = -MAX_FLOAT
            if obound1 is None:
                c_obound1[i] = MAX_FLOAT
            vscale[i] = 0.0
        else:
            if obound0 is None:
                c_obound0[i] = amrfld.global_xl[i]
            if obound1 is None:
                c_obound1[i] = amrfld.global_xh[i]
            vscale[i] = 1.0

    if ds0 == 0.0:
        ds0 = 0.5 * amrfld.min_dx

    # which integrator are we using?
    if method == EULER1:
        integrate_func = _c_euler1[FusedField, real_t]
    elif method == RK2:
        integrate_func = _c_rk2[FusedField, real_t]
    elif method == RK12:
        integrate_func = _c_rk12[FusedField, real_t]
    elif method == EULER1A:
        integrate_func = _c_euler1a[FusedField, real_t]
    else:
        raise ValueError("unknown integration method")

    # determine which functions to use for understanding endpoints
    if topo_style == "msphere":
        end_flags_to_topology = end_flags_to_topology_msphere
    else:
        end_flags_to_topology = end_flags_to_topology_generic

    # establish arrays for output
    if output & OUTPUT_STREAMLINES:
        # 2 (0=backward, 1=forward), 3 (x, y, z), maxit points in the line
        line_ndarr = np.empty((3, maxit2), dtype=amrfld.crd_dtype)
        line_mv = line_ndarr
        lines = [None] * nr_streams
    if output & OUTPUT_TOPOLOGY:
        topology_ndarr = np.empty((nr_streams,), dtype="i")
        topology_mv = topology_ndarr

    # for profiling
    # t0_all = time()

    seed_iter = islice(seed.iter_points(center=seed_center), *seed_slice)
    seed_pts = np.array(list(seed_iter), dtype=amrfld.crd_dtype)

    assert seed_pts.shape[1] == 3, "Seeds must have 3 spatial dimensions"

    with nogil:
        for i_stream in range(seed_pts.shape[0]):
            x0[0] = seed_pts[i_stream, 0]
            x0[1] = seed_pts[i_stream, 1]
            x0[2] = seed_pts[i_stream, 2]

            if line_mv is not None:
                # line_mv[:, :] = NAN  # Debug only
                line_mv[0, maxit] = x0[0]
                line_mv[1, maxit] = x0[1]
                line_mv[2, maxit] = x0[2]
            line_ends[0] = maxit - 1
            line_ends[1] = maxit + 1
            end_flags = _C_END_NONE

            for i in range(2):
                d = _dir_d[i]
                # i = 0, d = -1, backward ;; i = 1, d = 1, forward
                if d < 0 and not (stream_dir & _C_DIR_BACKWARD):
                    continue
                elif d > 0 and not (stream_dir & _C_DIR_FORWARD):
                    continue

                ds = d * ds0
                stream_length = 0.0

                s[0] = x0[0]
                s[1] = x0[1]
                s[2] = x0[2]

                it = line_ends[i]

                done = _C_END_NONE
                while 0 <= it and it < maxit2:
                    nr_segs += 1
                    pre_ds = fabs(ds)

                    if amrfld.nr_patches > 1:
                        # IMPORTANT: for AMR fields, this with gil block will
                        # absolutely destroy multithreaded performance because
                        # it happens each integration step, but there seems
                        # to be no gil-less way to change the active patch
                        # in cython the way things are currently structured
                        with gil:
                            activate_patch[FusedAMRField, real_t](amrfld, s)
                            patch = amrfld.active_patch

                    ret = integrate_func(patch, s, &ds,
                                         tol_lo, tol_hi, fac_refine , fac_coarsen,
                                         smallest_step, largest_step, vscale,
                                         cached_idx3)

                    if fabs(ds) >= pre_ds:
                        stream_length += pre_ds
                    else:
                        stream_length += fabs(ds)

                    # if i_stream == 0:
                    #     print(s[2], s[1], s[0])
                    # ret is non 0 when |v_mv| == 0
                    if ret != 0:
                        done = _C_END_ZERO_LENGTH
                        break

                    if line_mv is not None:
                        line_mv[0, it] = s[0]
                        line_mv[1, it] = s[1]
                        line_mv[2, it] = s[2]
                    it += d

                    # end conditions
                    done = classify_endpoint(s, stream_length, ibound,
                                             c_obound0, c_obound1,
                                             max_length, ds, x0)

                    if done:
                        break

                if done == _C_END_NONE:
                    done = _C_END_OTHER | _C_END_MAXIT

                line_ends[i] = it
                end_flags |= done

            # now we have forward and background traces, process this streamline
            if line_mv is not None:
                with gil:
                    lines[i_stream] = line_ndarr[:, line_ends[0] + 1:line_ends[1]].copy()

            if topology_mv is not None:
                topology_mv[i_stream] = end_flags_to_topology(end_flags)

    # for profiling
    # t1_all = time()
    # t = t1_all - t0_all
    # print("=> in cython nr_segments: {0:.05e}".format(nr_segs))
    # print("=> in cython time: {0:.03f}s {1:.03e}s/seg".format(t, t / nr_segs))

    return lines, topology_ndarr

cdef inline int classify_endpoint(real_t pt[3], real_t length, real_t ibound,
                           real_t obound0[3], real_t obound1[3],
                           real_t max_length, real_t ds, real_t pt0[3]) nogil:
    cdef int done = _C_END_NONE
    cdef real_t rsq = pt[0]**2 + pt[1]**2 + pt[2]**2

    if rsq < ibound**2:
        if pt[2] >= 0.0:
            done = _C_END_IBOUND_NORTH
        else:
            done = _C_END_IBOUND_SOUTH
    elif pt[0] < obound0[0]:
        done = _C_END_OBOUND_XL
    elif pt[1] < obound0[1]:
        done = _C_END_OBOUND_YL
    elif pt[2] < obound0[2]:
        done = _C_END_OBOUND_ZL
    elif pt[0] > obound1[0]:
        done = _C_END_OBOUND_XH
    elif pt[1] > obound1[1]:
        done = _C_END_OBOUND_YH
    elif pt[2] > obound1[2]:
        done = _C_END_OBOUND_ZH
    elif length > max_length:
        done = _C_END_MAX_LENGTH

    # if we are within 0.05 * ds of the initial position
    # distsq = (pt0[0] - pt[0])**2 + \
    #          (pt0[1] - pt[1])**2 + \
    #          (pt0[2] - pt[2])**2
    # if distsq < (0.05 * ds)**2:
    #     # print("cyclic field line")
    #     done = _C_END_CYCLIC
    #     break

    return done

cdef int end_flags_to_topology_msphere(int end_flags) nogil:
    cdef int topo = 0
    cdef int mask_open_north = _C_END_IBOUND_NORTH | _C_END_OBOUND
    cdef int mask_open_south = _C_END_IBOUND_SOUTH | _C_END_OBOUND

    # order of these if statements matters!
    if (end_flags & _C_END_OTHER):
        topo = end_flags
    # elif (topo & _C_END_CYCLIC):
    #     return _C_TOPOLOGY_MS_CYCLYC
    elif end_flags & (mask_open_north) == mask_open_north:
        topo = _C_TOPOLOGY_MS_OPEN_NORTH
    elif end_flags & (mask_open_south) == mask_open_south:
        topo = _C_TOPOLOGY_MS_OPEN_SOUTH
    elif end_flags == 3 or end_flags == 5 or end_flags == 7:
        topo = _C_TOPOLOGY_MS_CLOSED
    else:
        topo = _C_TOPOLOGY_MS_SW

    return topo

cdef int end_flags_to_topology_generic(int end_flags) nogil:
    return end_flags

##
## EOF
##
