#!/usr/bin/env python

from __future__ import division, print_function
import sys

import numpy as np
import viscid

from fort_tools import tracer


__all__ = ["fort_prep_field", "fort_interp_trilin", "fort_topology"]


def fort_prep_field(fld):
    fld = fld.atleast_3d().as_flat()
    x, y, z = fld.get_crds()
    # nc = fld.nr_comps
    nx, ny, nz = fld.sshape

    dat = np.array(np.ravel(fld.data, order='F').reshape((-1, nx, ny, nz), order="F"))

    dat = dat.astype('f4')
    x = x.astype('f4')
    y = y.astype('f4')
    z = z.astype('f4')

    assert dat.shape[-3:] == (nx, ny, nz)
    assert dat.shape[-3:] == (len(x), len(y), len(z))
    return dat, x, y, z, nx, ny, nz

# def fort_interp_trilin(dat, gx, gy, gz, nx, ny, nz, seeds, center, wrap=True):
def fort_interp_trilin(fld, seeds, wrap=True):
    dat, gx, gy, gz, nx, ny, nz = fort_prep_field(fld)
    dat = dat[0, ...]
    center = fld.center

    seed_center = seeds.center if hasattr(seeds, 'center') else center
    if seed_center.lower() in ('face', 'edge'):
        seed_center = 'cell'

    seeds = viscid.to_seeds(seeds)
    npts = seeds.get_nr_points(center=seed_center)
    pts = seeds.points(center=seed_center).astype('f4')
    ptsx = np.array(pts[0, :])
    ptsy = np.array(pts[1, :])
    ptsz = np.array(pts[2, :])

    result = np.empty((npts), dtype='f4', order='F')

    tracer.ffort_interp_trilin(dat, result, gx, gy, gz, ptsx, ptsy, ptsz,
                               nx, ny, nz, npts)

    if wrap:
        result = seeds.wrap_field(np.ascontiguousarray(result))

    return result

def fort_topology(fld, seeds, wrap=True):
    fldx, fldy, fldz = fld.component_fields()
    datx, gx, gy, gz, nx, ny, nz = fort_prep_field(fldx)
    daty, gx, gy, gz, nx, ny, nz = fort_prep_field(fldy)
    datz, gx, gy, gz, nx, ny, nz = fort_prep_field(fldz)
    center = fld.center

    seed_center = seeds.center if hasattr(seeds, 'center') else center
    if seed_center.lower() in ('face', 'edge'):
        seed_center = 'cell'

    seeds = viscid.to_seeds(seeds)
    npts = seeds.get_nr_points(center=seed_center)
    pts = seeds.points(center=seed_center).astype('f4')
    ptsx = np.array(pts[0, :], order='F', dtype='f4')
    ptsy = np.array(pts[1, :], order='F', dtype='f4')
    ptsz = np.array(pts[2, :], order='F', dtype='f4')

    topo = np.zeros((npts,), order='F', dtype='i4')
    nsegs = np.zeros((1,), order='F', dtype='i4')

    # tracer.get_topo_at(npts, nx, ny, nz, gx, gy, gz, datx, daty, datz,
    #                    ptsx, ptsy, ptsz, topo, nsegs)
    tracer.get_topo_at(topo, nsegs, ptsx, ptsy, ptsz, datx, daty, datz, gx, gy, gz, npts, nx, ny, nz)

    if wrap:
        topo = seeds.wrap_field(np.ascontiguousarray(topo), name="Topology")
        topo.set_info('nsegs', nsegs[0])

    return topo

def _main():
    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
