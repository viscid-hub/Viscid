#!/usr/bin/env python

from __future__ import print_function
import logging

import numpy as np
import numba as nb
from numba.decorators import jit, autojit

# @jit(nb.int_(np.float_[:,:,:,::1], nb.int_[:,:,::1],
#              nb.float_, nb.float_, nb.float_, nb.float_, nb.float_, nb.float_))
# @autojit(locals=dict(x=nb.float_, y=nb.float_, z=nb.float_))
@autojit()
def get_topo(Bfld, topo_arr, x1, x2, y1, y2, z1, z2):
    B = Bfld.data
    gx, gy, gz = Bfld.crds.get_crd(('xcc', 'ycc', 'zcc'))
    outnx, outny, outnz = topo_arr.shape
    # nz, ny, nx, nc = B.shape
    nsegs = 0

    for ix in range(outnz):
        for iy in range(outny):
            for iz in range(outnx):
                x = x1 + float(ix - 1) * (x2 - x1) / float(outnx - 1)
                y = y1 + float(iy - 1) * (y2 - y1) / float(outny - 1)
                z = z1 + float(iz - 1) * (z2 - z1) / float(outnz - 1)

                topo, nsegsi = xcl4(B, gx, gy, gz, x, y, z, 100)
                nsegs += nsegsi
                topo_arr[iz, iy, ix] = topo
    # print("nsegs:", nsegs)

    return nsegs

# @jit(nb.int_(np.float_[:,:,:,::1], nb.int_[:,:,::1],
#              nb.float_, nb.float_, nb.float_, nb.float_, nb.float_, nb.int_))
# @jit("float_, int_ (float_[:,:,:,::1], float_[:], float_[:], float_[:],"
#                    "float_, float_, float_, int_)",
#      locals=dict(x=nb.float_, y=nb.float_, z=nb.float_,
#                  ix=nb.int_, iy=nb.int_, iz=nb.int_,))
@autojit()
def xcl4(B, gx, gy, gz, x0, y0, z0, xmaxlen):
    ddir = 0.020
    ifl1 = 0
    ifl2 = 0
    nz, ny, nx, nc = B.shape
    maxst = int(xmaxlen / ddir)
    nsegs = 0
    
    # trace backward
    ds = -ddir
    x, y, z = x0, y0, z0
    ix, iy, iz = 0, 0, 0

    for k in range(maxst):
        x, y, z = trace2(B, gx, gy, gz, x, y, z, ds, ix, iy, iz)
        nsegs += 1
        rs2 = x**2 + y**2 + z**2
        # inner bound
        if rs2 < 3.5**2:
            ifl1 = 1
            break
        # outer bound
        if x < gx[0] or x >= gx[nx - 1]:
            ifl1 = 2
            break
        if y < gy[0] or y >= gy[ny - 1]:
            ifl1 = 2
            break
        if z < gz[0] or z >= gz[nz - 1]:
            ifl1 = 2
            break

    # trace forward
    ds = ddir
    x, y, z = x0, y0, z0

    for k in range(maxst):
        x, y, z = trace2(B, gx, gy, gz, x, y, z, ds, ix, iy, iz)
        nsegs += 1
        rs2 = x**2 + y**2 + z**2
        # inner bound
        if rs2 < 3.5**2:
            ifl2 = 1
            break
        # outer bound
        if x < gx[0] or x >= gx[nx - 1]:
            ifl2 = 2
            break
        if y < gy[0] or y >= gy[ny - 1]:
            ifl2 = 2
            break
        if z < gz[0] or z >= gz[nz - 1]:
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

    return topo, nsegs

# @autojit(locals=dict(x2=nb.float_, y2=nb.float_, nz=nb.float_))
@autojit()
def trace2(B, gx, gy, gz, x, y, z, ds, ix, iy, iz):
    r2 = x**2 + y**2 + z**2
    if r2 <= 3.7**2:
        x2 = 0.0
        y2 = 0.0
        z2 = 0.0
    else:
        bbx, ix, iy, iz = ipol3a(B, 0, gx, gy, gz, x, y, z, ix, iy, iz)
        bby, ix, iy, iz = ipol3a(B, 1, gx, gy, gz, x, y, z, ix, iy, iz)
        bbz, ix, iy, iz = ipol3a(B, 2, gx, gy, gz, x, y, z, ix, iy, iz)

        bb = np.sqrt(bbx**2 + bby**2 + bbz**2)
        if bb > 0:
            s = ds / bb

            x2 = x + bbx * s
            y2 = y + bby * s
            z2 = z + bbz * s
        else:
            x2 = 1e6
            y2 = 1e6
            z2 = 1e6

    return x2, y2, z2#, ix, iy, iz

@autojit
def ipol3a(B, m, gx, gy, gz, x, y, z, ix, iy, iz):
    ix = closest_ind(gx, x, ix)
    iy = closest_ind(gy, y, iy)
    iz = closest_ind(gz, z, iz)

    dxi = (x - gx[ix]) / (gx[ix + 1] - gx[ix])
    dyi = (y - gy[iy]) / (gy[iy + 1] - gx[iy])
    dzi = (z - gz[iz]) / (gz[iz + 1] - gx[iz])

    bminterp = interp3(B, m, ix, iy, iz, dxi, dyi, dzi)
    return bminterp, ix, iy, iz

@autojit
def closest_ind(gx, x, ix):
    nx = gx.shape[0]

    # already there?
    if x > gx[ix] and x <= gx[ix + 1]:
        return ix
    # one cell ahead?
    ix = np.min(ix + 1, nx - 2)
    if x > gx[ix] and x <= gx[ix + 1]:
        return ix
    # one cell behind?
    ix = np.max(ix -2, 0)
    if x > gx[ix] and x <= gx[ix + 1]:
        return ix
    # ok then, search
    for i in range(nx):
        if x > gx[ix] and x <= gx[ix + 1]:
            return ix        

    return ix

@autojit
def interp3(a, m, ix, iy, iz, dxi, dyi, dzi):
    x00 = a[ix, iy    , iz    , m] + dxi * (a[ix + 1, iy    , iz    , m] - a[ix, iy    , iz    , m])
    x10 = a[ix, iy + 1, iz    , m] + dxi * (a[ix + 1, iy + 1, iz    , m] - a[ix, iy + 1, iz    , m])
    x01 = a[ix, iy    , iz + 1, m] + dxi * (a[ix + 1, iy    , iz + 1, m] - a[ix, iy    , iz + 1, m])
    x11 = a[ix, iy + 1, iz + 1, m] + dxi * (a[ix + 1, iy + 1, iz + 1, m] - a[ix, iy + 1, iz + 1, m])
    y0 = x00 + dyi * (x10 - x00)
    y1 = x01 + dyi * (x11 - x01)
    z = y0 + dzi * (y1 - y0)
    return z

##
## EOF
##
