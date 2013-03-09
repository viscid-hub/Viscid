# cython: boundscheck=True, wraparound=True

from __future__ import print_function

from .. import field
from . import seed

###########
# cimports
cimport numpy as np

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
    nptype = fld.data.dtype.name

    dat = fld.data
    crdz, crdy, crdx = fld.crds.get_cc()

    if isinstance(seeds, seed.SeedGen):
        x0 = seeds.points
    else:
        x0 = np.array(seeds, dtype=dat.dtype).reshape((-1, 3))

    lines = []
    for start in x0:
        line = _py_streamline(dat, crdz, crdy, crdx,
                              start, *args, **kwargs)
        lines.append(line)
    return lines

def _py_streamline(real_t[:,:,:,:] v_arr, real_t[:] crdz, real_t[:] crdy,
                   real_t[:] crdx, real_t[:] x0, ds0=-1.0, ibound=0.0,
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
        real_t s_arr[3]
        real_t[:] s = s_arr
        real_t px, py, pz
        real_t d
        real_t rsq, distsq

    line = []

    if obound0 is None:
        c_obound0[0] = crdz[0]
        c_obound0[1] = crdy[0]
        c_obound0[2] = crdx[0]
    else:
        py_obound0 = obound0
        c_obound0[...] = py_obound0

    if obound0 is None:
        c_obound1[0] = crdz[-1]
        c_obound1[1] = crdy[-1]
        c_obound1[2] = crdx[-1]
    else:
        py_obound1 = obound1
        c_obound1[...] = py_obound1

    if c_ds0 <= 0.0:
        # FIXME: calculate something reasonable here
        c_ds0 = 0.01

    lseg = [[[x0[0], x0[1], x0[2]]], []]
    for i, d in enumerate([-1.0, 1.0]):
        if d < 0 and not (c_dir & BACKWARD):
            continue
        elif d > 0 and not (c_dir & FORWARD):
            continue

        ds = d * c_ds0

        s[0] = x0[0]
        s[1] = x0[1]
        s[2] = x0[2]

        it = 0
        done = 0
        while it <= c_maxit:
            # print("point (x, y, z): ", s[2], s[1], s[0])
            # components run x, y, z, but coords run z, y, x
            ret = _c_euler1[real_t](v_arr, crdz, crdy, crdx, ds, s)
            # ret is non 0 when |varr| == 0
            if ret != 0:
                done = 1
                break

            lseg[i].append([s[0], s[1], s[2]])
            it += 1

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
            # print("maxit")

    # reverse the 'backward' line segment
    # print("-- first: ", lseg[0][:4], " last: ", lseg[0][-4:])
    # print("++ first: ", lseg[1][:4], " last: ", lseg[1][-4:])
    line = (lseg[0][::-1] + lseg[1])
    return line

##
## EOF
##
