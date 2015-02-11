#!/usr/bin/env python
"""
Convert athena files (one file per time step per proc) to
xdmf files (one file per time step). The usual call will
look something like:

athena2xdmf id*/*.bin
"""

# TODO: make this more general, the only thing that is athena specific
# is hard coding an AthenaGrid, and the name of the output file (.ath.h5).

from __future__ import division, print_function
import sys
import argparse
from itertools import count

import numpy as np

import viscid
from viscid import vutil
from viscid.readers.athena import AthenaGrid
from viscid import coordinate
from viscid.compat import izip

def main():
    # tollerences for if endpoints of crd arrays are equal
    rtol, atol = 1e-14, 0.0
    parser = argparse.ArgumentParser(description=__doc__,
                                     conflict_handler='resolve')
    parser.add_argument('files', nargs='+', help='input file')
    args = vutil.common_argparse(parser)

    files = viscid.load_files(args.files)

    # sort of lucky that the 0th id is just the RUN, not RUN-id0
    runname = files[0].find_info('run')
    print(runname)

    times_iter = [f.iter_times(":") for f in files]
    for it in izip(count(), *times_iter):
        itime = it[0]
        grids = [f.get_grid() for f in it[1:]]

        print("working on itime", itime)

        # check that the times are the same
        for g in grids[1:]:
            if g.time != grids[0].time:
                raise ValueError("OH NOOOO, times don't match!")

        # go through the grids and come out with sorted lists of unique
        # coordinate arrays in each direction
        unique_crd_lst = [[], [], []]
        for g in grids:
            # i corresponds to dimension (zyx)
            for i, xi_nc in enumerate(g.get_crds_nc("zyx")):
                has_match = False
                for unique_xi in unique_crd_lst[i]:
                    # print(">>", i, xi_nc, unique_xi)
                    if np.isclose(xi_nc[0], unique_xi[0], rtol, atol):
                        if len(xi_nc) != len(unique_xi):
                            raise ValueError("different lengths :( "
                                             "SMR is not supported")
                        if not np.isclose(xi_nc[0], unique_xi[0], rtol, atol):
                            raise ValueError("different end point :( "
                                             "SMR is not supported")
                        has_match = True
                        break
                if not has_match:
                    ind = 0
                    for unique_xi in unique_crd_lst[i]:
                        if xi_nc[0] < unique_xi[0]:
                            if xi_nc[-1] > unique_xi[0]:
                                raise ValueError("overlapping crds :(")
                            break
                        else:
                            ind += 1
                    unique_crd_lst[i].insert(ind, xi_nc)

        # now go through the grids again and figure out which position each
        # grid is in... there should be no more surprises since weird grids
        # should have risen ValueErrors above
        ip = -999 * np.ones((3, len(grids)), dtype="int")
        nnodes = -999 * np.ones((3, len(grids)), dtype="int")
        ncells = -999 * np.ones((3, len(grids)), dtype="int")
        for igrid, g in enumerate(grids):
            for i, xi_nc in enumerate(g.get_crds_nc("zyx")):
                nnodes[i, igrid] = len(xi_nc)
                ncells[i, igrid] = nnodes[i, igrid] - 1
                if ncells[i, igrid] < 1:
                    raise ValueError("Um, 1D/2D? say what?")

                for k, unique_crd in enumerate(unique_crd_lst[i]):
                    if np.isclose(xi_nc[0], unique_crd[0], rtol, atol):
                        ip[i, igrid] = k
                        break
                if ip[i, igrid] == -999:
                    raise RuntimeError("I shouldn't be here")

        # we now have the index of a process in each dimension, lets figure
        # out the slices and dimensions of the whole domain
        slices_nc = []
        slices_cc = []
        for igrid, g in enumerate(grids):
            s_nc = []
            s_cc = []
            for i in range(3):
                start = np.sum(nnodes[i, :ip[i, igrid]])
                stop = start + nnodes[i, igrid]
                s_nc.append(slice(start, stop))
                start = np.sum(ncells[i, :ip[i, igrid]])
                stop = start + ncells[i, igrid]
                s_cc.append(slice(start, stop))
            slices_nc.append(s_nc)
            slices_cc.append(s_cc)

        # make coordinates
        clst = []
        for i, ax, ucl in izip(count(), "zyx", unique_crd_lst):
            # this removes the first element from all but the first unique
            # crd list, since that is a repeated node center
            to_cat = [uc[1:] if j > 0 else uc for j, uc in enumerate(ucl)]
            clst.append((ax, np.concatenate(to_cat)))
        crds = coordinate.wrap_crds("nonuniform_cartesian", clst)
        target_grid = AthenaGrid("AthenaGrid")
        target_grid.set_crds(crds)

        # print("clst", clst, "shape", crds.shape)
        # print("ip", ip)
        # print("ncells", ncells)
        # print("slices", slices)
        # print(crds.shape)

        for name in grids[0].field_names:
            fld0 = grids[0][name]
            dtyp = fld0.dtype
            if fld0.iscentered("Node"):
                target_arr = np.empty(crds.shape_nc, dtype=dtyp)
                slices = slices_nc
            else:
                target_arr = np.empty(crds.shape_cc, dtype=dtyp)
                slices = slices_cc

            for i, g in enumerate(grids):
                target_arr[slices[i]] = g[name].data

            fld = fld0.wrap(target_arr, context=dict(crds=crds))
            target_grid.add_field(fld)

        viscid.save_grid("{0}.{1:0{2}d}.ath.h5".format(runname, itime, 6),
                         target_grid)

if __name__ == "__main__":
    sys.exit(main())

##
## EOF
##
