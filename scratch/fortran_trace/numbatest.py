#!/usr/bin/env python

from __future__ import print_function
from timeit import default_timer as time
import cProfile
import pstats

import numpy as np
import numba as nb

from viscid import readers
from viscid import field
from viscid.calculator import seed

# nb.float_ = nb.template("nb.float_")

@nb.jit(nb.int_(nb.float_[:], nb.float_, nb.int_[:], nb.int_),
        nopython=True)
def closest_ind(crd, point, startinds, m):
    i = 0
    fallback = 0
    n = crd.shape[0]
    forward = n > 1 and crd[1] > crd[0]

    if startinds[m] < 0:
        start = 0
    elif startinds[m] > n - 1:
        start = n - 1
    else:
        start = startinds[m]

    # search linearly... maybe branch prediction makes this better
    # than bisection for smallish arrays...
    # point is 'upward' (index wise) of crd[startind]... only search up
    if ((forward and crd[start] <= point) or \
        (not forward and crd[start] >= point)):
        for i in range(start, n - 1):
            if forward and crd[i + 1] >= point:
                startinds[m] = i
                return i
            if not forward and crd[i + 1] <= point:
                startinds[m] = i
                return i
        # if we've gone too far, pick the last index
        fallback = max(n - 2, 0)
        startinds[m] = fallback
        return fallback

    # startind was too large... go backwards
    for i in range(start - 1, -1, -1):
        if forward and crd[i] <= point:
            startinds[m] = i
            return i
        if not forward and crd[i] >= point:
            startinds[m] = i
            return i
    # if we've gone too far, pick the first index
    fallback = 0
    startinds[m] = fallback
    return fallback

@nb.jit(nb.float_(nb.float_[:,:,:,::1], nb.int_, nb.float_[:], nb.float_[:],
                      nb.float_[:], nb.float_[:], nb.int_[:]),
            nopython=False)
def trilin_interp(v, m, crdz, crdy, crdx, x, startinds):
    ix = np.array([0, 0, 0], dtype=nb.int_)
    p = np.array([0, 0, 0], dtype=nb.int_)
    xd = np.array([0.0, 0.0, 0.0], dtype=nb.float_)
    crds = [crdz, crdy, crdx]

    # find iz, iy, ix from startinds
    for i in range(3):
        ind = closest_ind(crds[i], x[i], startinds, i)
        ix[i] = ind
        p[i] = 1
        xd[i] = (x[i] - crds[i][ind]) / (crds[i][ind + 1] - crds[i][ind])

    c00 = v[ix[0], ix[1]       , ix[2]       , m] + xd[0] * (v[ix[0] + p[0], ix[1]       , ix[2]       , m] - v[ix[0], ix[1]       , ix[2]       , m])
    c10 = v[ix[0], ix[1] + p[1], ix[2]       , m] + xd[0] * (v[ix[0] + p[0], ix[1] + p[1], ix[2]       , m] - v[ix[0], ix[1] + p[1], ix[2]       , m])
    c01 = v[ix[0], ix[1]       , ix[2] + p[2], m] + xd[0] * (v[ix[0] + p[0], ix[1]       , ix[2] + p[2], m] - v[ix[0], ix[1]       , ix[2] + p[2], m])
    c11 = v[ix[0], ix[1] + p[1], ix[2] + p[2], m] + xd[0] * (v[ix[0] + p[0], ix[1] + p[1], ix[2] + p[2], m] - v[ix[0], ix[1] + p[1], ix[2] + p[2], m])
    c0 = c00 + xd[1] * (c10 - c00)
    c1 = c01 + xd[1] * (c11 - c01)
    c = c0 + xd[2] * (c1 - c0)

    return c

@nb.jit(nb.float_[:](nb.float_[:,:,:,::1], nb.float_[:], nb.float_[:],
                      nb.float_[:], nb.float_[:], nb.float_,
                      nb.int_[:]))
def euler1(v, crdx, crdy, crdz, x, ds, startinds):
    vx = trilin_interp(v, 0, crdz, crdy, crdx, x, startinds)
    vy = trilin_interp(v, 1, crdz, crdy, crdx, x, startinds)
    vz = trilin_interp(v, 2, crdz, crdy, crdx, x, startinds)
    vmag = np.sqrt(vx**2 + vy**2 + vz**2)
    # if vmag == 0.0 or isnan(vmag):
    #     logging.warning("vmag issue at: {0} {1} {2}".format(x[0], x[1], x[2]))
    #     return 1
    x[0] += ds * vz / vmag
    x[1] += ds * vy / vmag
    x[2] += ds * vx / vmag
    return x

@nb.jit(nb.int_(nb.float_[:,:,:,::1], nb.float_[:], nb.float_[:], nb.float_[:],
                nb.float_[:], nb.int_), nopython=True)
def trace(B, gx, gy, gz, r0, maxlen):
    ddir = 0.020
    ifl1 = 0
    ifl2 = 0
    nz, ny, nx, nc = B.shape
    maxst = int(maxlen / ddir)
    nsegs = 0
    start_ind = np.array([0, 0, 0], dtype=nb.int_)

    # trace backward
    ds = -ddir
    r = r0
    for k in range(maxst):
        r = euler1(B, gx, gy, gz, r, ds, start_ind)
        nsegs += 1
        rs2 = r[0]**2 + r[1]**2 + r[2]**2
        # inner bound
        if rs2 < 3.5**2:
            ifl1 = 1
            break
        # outer bound
        if r[0] < gz[0] or r[0] >= gz[nz - 1]:
            ifl1 = 2
            break
        if r[1] < gy[0] or r[1] >= gy[ny - 1]:
            ifl1 = 2
            break
        if r[2] < gx[0] or r[2] >= gx[nx - 1]:
            ifl1 = 2
            break

    # trace forward
    ds = ddir
    r = r0
    for k in range(maxst):
        r = euler1(B, gx, gy, gz, r, ds, start_ind)
        nsegs += 1
        rs2 = r[0]**2 + r[1]**2 + r[2]**2
        # inner bound
        if rs2 < 3.5**2:
            ifl2 = 1
            break
        # outer bound
        if r[0] < gz[0] or r[0] >= gz[nz - 1]:
            ifl2 = 2
            break
        if r[1] < gy[0] or r[1] >= gy[ny - 1]:
            ifl2 = 2
            break
        if r[2] < gx[0] or r[2] >= gx[nx - 1]:
            ifl2 = 2
            break

    topo = 3
    if ifl1 == 1 and ifl2 == 1:
        topo = 7
    if ifl1 == 1 and ifl2 == 2:
        topo = 14
    if ifl1 == 2 and ifl2 == 1:
        topo = 13
    if ifl1 == 2 and ifl2 == 2:
        topo = 8

    return topo

@nb.jit(nb.int_(nb.float_[:,:,:,::1], nb.float_[:], nb.float_[:], nb.float_[:],
                    nb.int_[:,:,::1], nb.float_[:], nb.float_[:]),
            locals=dict(nsegs=nb.int_), nopython=False)
def nb_get_topo(B, gx, gy, gz, topo_arr, x1, x2):
    outnx, outny, outnz = topo_arr.shape
    # nz, ny, nx, nc = B.shape
    nsegs = 0
    r = np.array([0.0, 0.0, 0.0], dtype=nb.float_)

    for iz in range(outnz):
        for iy in range(outny):
            for ix in range(outnx):
                print(iz, iy, ix)
                r[0] = x1[0] + float(iz - 1) * (x2[0] - x1[0]) / float(outnz - 1)
                r[1] = x1[1] + float(iy - 1) * (x2[1] - x1[1]) / float(outny - 1)
                r[2] = x1[2] + float(ix - 1) * (x2[2] - x1[2]) / float(outnx - 1)

                topo = trace(B, gx, gy, gz, r, 100)
                # nsegs += nsegsi
                topo_arr[iz, iy, ix] = topo

    # print("nsegs:", nsegs)
    return nsegs

@nb.autojit()
def py_get_topo(Bfld, topo_arr, x1, x2, y1, y2, z1, z2):
    B = Bfld.data
    gx, gy, gz = Bfld.crds.get_crd(('xcc', 'ycc', 'zcc'))
    x1 = np.array([z1, y1, x1], dtype=B.dtype)
    x2 = np.array([z2, y2, x2], dtype=B.dtype)
    nsegs = nb_get_topo(B, gx, gy, gz, topo_arr, x1, x2)
    return nsegs

def main():
    gsize = (2, 2, 2)
    x1 = -10.0; x2 = -5.0 #pylint: disable=C0321
    y1 = -5.0; y2 = 5.0 #pylint: disable=C0321
    z1 = -5.0; z2 = 5.0 #pylint: disable=C0321
    vol = seed.Volume((z1, y1, x1), (z2, y2, x2), gsize)

    f3d = readers.load("/Users/kmaynard/dev/work/t1/t1.3df.004320.xdmf")
    fld_bx = f3d["bx"]
    fld_by = f3d["by"]
    fld_bz = f3d["bz"]

    B = field.scalar_fields_to_vector("B_cc", [fld_bx, fld_by, fld_bz],
                            info={"force_layout": field.LAYOUT_INTERLACED})
    topo_arr = np.empty(gsize, order='C', dtype='int')
    lines, topo = None, None
    t0 = time()
    cProfile.runctx("nsegs = py_get_topo(B, topo_arr, x1, x2, y1, y2, z1, z2)",
                    globals(), locals(), "topo.prof")
    t1 = time()
    s = pstats.Stats("topo.prof")
    s.strip_dirs().sort_stats("tottime").print_stats()
    nsegs = py_get_topo(B, topo_arr, x1, x2, y1, y2, z1, z2)
    t = t1 - t0

    print(topo_arr)

    # print("numba time: {0}s, {1}s/seg".format(t, t / nsegs))
    print("numba time: {0}s".format(t))

if __name__ == "__main__":
    main()

##
## EOF
##
