#!/usr/bin/env python
"""Functions for clustering clouds of neighboring points"""

from __future__ import print_function, division
from itertools import count

import numpy as np


__all__ = ['distance_to_clusters', 'find_clusters']


def distance_to_clusters(point, clusters, alt=()):
    """L2 distance between point and clusters"""
    x, y = point

    dists = np.zeros((len(clusters),), dtype='f')
    wraps = np.zeros((len(clusters), len(point)), dtype='int')

    for i, (clx, cly) in enumerate(clusters):
        clx, cly = np.asarray(clx), np.asarray(cly)

        dists[i] = np.min(np.sqrt((clx - x)**2 +
                                  (cly - y)**2))
        for altx, alty in alt:
            newdist = np.min(np.sqrt((clx - (x + altx))**2 +
                                     (cly - (y + alty))**2))
            if newdist < dists[i]:
                dists[i] = newdist
                wraps[i, :] = (altx, alty)

    wraps = np.sign(wraps)
    return dists, wraps

def find_clusters(indx, indy, x, y, multiple=True, periodic=(False, False),
                  diagonals=True):
    """Cluster and average groups of neighboring points

    TODO: If absolutely necessary, could do some K-means clustering
        here by calling into scikit-learn.

    Args:
        indx (sequence): list of x indices
        indy (sequence): list of y indices
        x (sequence): x coordinate array
        y (sequence): y coordinate array
        multiple (bool): If False, average all points as a single
            cluster
        periodic (sequence): indicate whether that direction is
            periodic, and if so, whether the coordinate arrays are
            overlapped or not. Values can be True, False, or '+'. '+'
            indicates that x[0] and x[-1] are not colocated, so assume
            they're dx apart where dx = x[-1] - x[-2].
        diagonals (bool): if true, then diagonal points are considered
            neighbors

    Returns:
        ndarray: 2xN for N clusters
    """
    assert len(indx) == len(indy)
    inds = np.array([indx, indy]).T
    crds = [np.asarray(x), np.asarray(y)]
    thresh = 1.5 if diagonals else 1.1

    # setup some bookkeeping about periodicity
    pd = [False if pi == "0" else bool(pi) for pi in periodic]
    pdN = [len(_x) for _x in crds]
    pdL = np.array([_x[-1] - _x[0] for _x in crds])
    xh = np.array([_x[-1] for _x in crds])
    xl = np.array([_x[0] for _x in crds])
    for i, p in enumerate(periodic):
        if not pd[i]:
            pdN[i] = 0
            pdL[i] = 0.0
        elif str(p).strip() == '+':
            pdL[i] += crds[i][-1] - crds[i][-2]
    alt = []
    if pd[0]:
        alt += [(-pdN[0], 0), (pdN[0], 0)]
    if pd[1]:
        alt += [(0, -pdN[1]), (0, pdN[1])]
    if pd[0] and pd[1]:
        alt += [(-pdN[0], -pdN[1]), (pdN[0], pdN[1])]

    # cli == cluster indices, clx == cluster locations
    clx = []

    if multiple:
        clusters = []
        cli = []
        for i, (ix, iy) in enumerate(inds):
            ind = inds[i]
            dists, wraps = distance_to_clusters((ix, iy), cli, alt=alt)
            touching = dists < thresh
            ntouching = np.sum(touching)
            if ntouching > 1:
                # if this point touches > 1 cluster, then merge the other
                # clusters
                clusters.append([i])
                cli.append([[ix], [iy]])
                clx.append([[x[ix]], [y[iy]]])

                for k in reversed(np.flatnonzero(touching)):
                    clusters[-1] += clusters.pop(k)
                    icli = cli.pop(k)
                    iclx = clx.pop(k)
                    for j, N, L in zip(count(), pdN, pdL):
                        iclij = np.asarray(icli[j]) - wraps[k][j] * N
                        iclxj = np.asarray(iclx[j]) - wraps[k][j] * L
                        cli[-1][j] += iclij.tolist()
                        clx[-1][j] += iclxj.tolist()
            elif ntouching == 1:
                icluster = np.argmax(touching)
                wrap = wraps[icluster]
                clusters[icluster].append(i)
                for j, N, L in zip(count(), pdN, pdL):
                    cli[icluster][j].append(ind[j] + wrap[j] * N)
                    clx[icluster][j].append(crds[j][ind[j]] + wrap[j] * L)
            else:
                clusters.append([i])
                cli.append([[ix], [iy]])
                clx.append([[x[ix]], [y[iy]]])
    else:
        clx = [[x[inds[0, :]], y[inds[1, :]]]]

    pts = np.array([np.average(clxi, axis=1) for clxi in clx]).reshape((-1, 2))
    pts -= pdL * np.floor((pts - xl) / (xh - xl))
    return pts.T

def _main():
    # test clustering; x == theta && y == phi
    Nx, Lx = 16, 180.0
    Ny, Ly = 32, 360.0
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny, endpoint=True)
    ix = [0,      0, Nx - 1, Nx - 1, Nx - 4]  # pylint: disable=bad-whitespace
    iy = [0, Ny - 1,      0, Ny - 1, Ny - 4]  # pylint: disable=bad-whitespace
    pts = fin_clusters(ix, iy, x, y, multiple=True, periodic="01")
    print(pts.shape)
    print(pts)

if __name__ == "__main__":
    import sys
    sys.exit(_main())

##
## EOF
##
