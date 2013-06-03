# cython: boundscheck=True, wraparound=True, profile=False

from __future__ import print_function

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

#####################
# now the good stuff

def streamlines(fld, seeds, *args, **kwargs):
    if not fld.layout == field.LAYOUT_INTERLACED:
        raise ValueError("Streamlines only written for interlaced data.")
    if fld.dim != 3:
        raise ValueError("Streamlines are only written in 3D.")
    # nptype = fld.data.dtype.name

    dat = fld.data
    dtype = dat.dtype
    center = "Cell"
    crdz, crdy, crdx = fld.crds.get_crd(center=center)
    iter_seeds = seeds.iter_points(center=center)

    return _py_streamline(dtype, dat, crdz, crdy, crdx, iter_seeds,
                          *args, **kwargs)

def _py_streamline(dtype, real_t[:,:,:,:] v_arr, real_t[:] crdz, real_t[:] crdy,
                   real_t[:] crdx, iter_seeds, ds0=-1.0, ibound=0.0,
                   obound0=None, obound1=None, dir=BOTH,
                   maxit=10000):
    """ Start calculating a streamline at x0
    dir:         DIR_FORWARD, DIR_BACKWARD, DIR_BOTH
    ibound:      stop streamline if within inner_bound of the origin
                 ignored if 0
    obound0:     corner of box beyond which to stop streamline (smallest values)
    obound1:     corner of box beyond which to stop streamline (smallest values)
    ds0:         initial spatial step for the streamline """
    cdef:
        # cdefed versions of arguments
        real_t c_ds0 = ds0
        real_t c_ibound = ibound
        real_t c_obound0_arr[3]
        real_t c_obound1_arr[3]
        real_t[:] c_obound0 = c_obound0_arr
        real_t[:] c_obound1 = c_obound1_arr
        real_t[:] py_obound0
        real_t[:] py_obound1
        int c_dir = dir
        int c_maxit = maxit

        # just for c
        int i, j, it
        int ret
        int done
        real_t px, py, pz
        int d
        real_t rsq, distsq

        real_t[:,:,:] line_arr
        real_t[:,:] line
        real_t x0_arr[3]
        int line_ends_arr[2]
        real_t s_arr[3]

        real_t[:] x0 = x0_arr
        int[:] line_ends = line_ends_arr
        real_t[:] s = s_arr

    if obound0 is None:
        c_obound0[0] = crdz[0]
        c_obound0[1] = crdy[0]
        c_obound0[2] = crdx[0]
    else:
        py_obound0 = obound0
        c_obound0[...] = py_obound0

    if obound1 is None:
        c_obound1[0] = crdz[-1]
        c_obound1[1] = crdy[-1]
        c_obound1[2] = crdx[-1]
    else:
        py_obound1 = obound1
        c_obound1[...] = py_obound1

    if c_ds0 <= 0.0:
        # FIXME: calculate something reasonable here
        c_ds0 = 0.01

    lines = []
    line_arr = np.empty((2, 3, c_maxit), dtype=dtype)

    for seed_pt in iter_seeds:
        # print(seed_pt)
        x0[0] = seed_pt[0]
        x0[1] = seed_pt[1]
        x0[2] = seed_pt[2]

        line_arr[1, 0, c_maxit - 1] = x0[0]
        line_arr[1, 1, c_maxit - 1] = x0[1]
        line_arr[1, 2, c_maxit - 1] = x0[2]
        line_ends[0] = c_maxit - 2
        line_ends[1] = 0

        for i, d in enumerate([-1, 1]):
            if d < 0 and not (c_dir & BACKWARD):
                continue
            elif d > 0 and not (c_dir & FORWARD):
                continue

            ds = d * c_ds0

            s[0] = x0[0]
            s[1] = x0[1]
            s[2] = x0[2]

            it = line_ends[i]

            done = 0
            while 0 <= it and it < c_maxit:
                ret = _c_euler1[real_t](v_arr, crdz, crdy, crdx, ds, s)
                # ret is non 0 when |varr| == 0
                if ret != 0:
                    done = 1
                    break

                line_arr[i, 0, it] = s[0]
                line_arr[i, 1, it] = s[1]
                line_arr[i, 2, it] = s[2]
                it += d

                # end conditions
                rsq = s[0]**2 + s[1]**2 + s[2]**2

                # hit the inner boundary
                if rsq <= c_ibound**2:
                    # print("inner boundary")
                    done = 1
                    break

                for j from 0 <= j < 3:
                    # hit the outer boundary
                    if s[j] <= c_obound0[j] or s[j] >= c_obound1[j]:
                        # print("outer boundary")
                        done = 1
                        break

                # if we are within 0.99 * ds0 of the initial position
                distsq = (x0[0] - s[0])**2 + (x0[1] - s[1])**2 + (x0[2] - s[2])**2
                if distsq < (0.99 * ds0)**2:
                    # print("cyclic field line")
                    done = 1
                    break

                if done:
                    break
            if not done:
                pass

            line_ends[i] = it

        # reverse the 'backward' line segment
        line = np.concatenate((line_arr[0, :, line_ends[0] + 1:], 
                               line_arr[1, :, :line_ends[1]]), axis=1)

        lines.append(line)
    return lines

##
## EOF
##
