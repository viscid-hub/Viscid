"""I don't know if this is worth keeping as its own module,
TOPOLOGY_* is copied here so that one can import this module
without needing to have built the cython module streamline.pyx
"""

from __future__ import print_function
from itertools import count

import numpy as np
import viscid

TOPOLOGY_MS_NONE = 0  # no translation needed
TOPOLOGY_MS_CLOSED = 1  # translated from 5, 6, 7(4|5|6)
TOPOLOGY_MS_OPEN_NORTH = 2  # translated from 13 (8|5)
TOPOLOGY_MS_OPEN_SOUTH = 4  # translated from 14 (8|6)
TOPOLOGY_MS_SW = 8  # no translation needed
# TOPOLOGY_MS_CYCLIC = 16  # no translation needed
TOPOLOGY_MS_INVALID = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
# TOPOLOGY_MS_OTHER = list(range(32, 512))  # >= 16

TOPOLOGY_G_NONE = 0

SEP_VAL = (TOPOLOGY_MS_CLOSED | TOPOLOGY_MS_SW |
           TOPOLOGY_MS_OPEN_NORTH | TOPOLOGY_MS_OPEN_SOUTH)

color_map_msphere = {TOPOLOGY_MS_CLOSED: (0.0, 0.8, 0.0),
                     TOPOLOGY_MS_OPEN_NORTH: (0.0, 0.0, 0.7),
                     TOPOLOGY_MS_OPEN_SOUTH: (0.7, 0.0, 0.0),
                     TOPOLOGY_MS_SW: (0.7, 0.7, 0.7)
                    }

color_map_generic = {}

# for legacy reasons, make some aliases
TOPOLOGY_NONE = TOPOLOGY_MS_NONE
TOPOLOGY_CLOSED = TOPOLOGY_MS_CLOSED
TOPOLOGY_OPEN_NORTH = TOPOLOGY_MS_OPEN_NORTH
TOPOLOGY_OPEN_SOUTH = TOPOLOGY_MS_OPEN_SOUTH
TOPOLOGY_SW = TOPOLOGY_MS_SW
# TOPOLOGY_CYCLIC = TOPOLOGY_MS_CYCLIC
TOPOLOGY_INVALID = TOPOLOGY_MS_INVALID
# TOPOLOGY_OTHER = TOPOLOGY_MS_OTHER
color_map = color_map_msphere


def topology2color(topology, topo_style="msphere", bad_color=None):
    """Determine RGB from topology value

    Parameters:
        topology (int, list, ndarray): some value in
            ``calculator.streamline.TOPOLOGY_*``
        topo_style (string): msphere, or a dict with its own
            mapping
        bad_color (tuple): rgb color for invalid topologies

    Returns:
        Nx3 array of rgb data or (R, G, B) tuple if topology is a
        single value
    """
    if isinstance(topo_style, dict):
        mapping = topo_style
    elif topo_style == "msphere":
        mapping = color_map_msphere
    else:
        mapping = color_map_generic

    if bad_color is None:
        bad_color = (0.0, 0.0, 0.0)

    ret = None
    try:
        if isinstance(topology, viscid.field.Field):
            topology = topology.flat_data
        ret = np.empty((len(topology), 3))
        for i, topo in enumerate(topology):
            try:
                ret[i, :] = mapping[topo]
            except KeyError:
                ret[i] = bad_color
    except TypeError:
        try:
            ret = mapping[int(topology)]
        except KeyError:
            ret = bad_color
    return ret

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

def cluster(indx, indy, x, y, multiple=True, periodic=(False, False),
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

    pts = np.array([np.average(clxi, axis=1) for clxi in clx])
    pts -= pdL * np.floor((pts - xl) / (xh - xl))
    return pts.T

def find_sep_points_cartesian(fld, min_iterations=1, max_iterations=10,
                              multiple=True, sep_val=SEP_VAL, plot=False,
                              mask_limit=0b1111, periodic="00", pt_bnds=()):
    """Find separator as intersection of all global topologies

    Neighbors are bitwise ORed until at least one value matches
    `sep_val` which is presumably (Close | Open N | Open S | SW).
    This happens between min_iterations and max_iterations times,
    where the resolution of each iteration is reduced by a factor
    of two, ie, worst case 2**(max_iterations).

    Args:
        fld (Field): Topology (bitmask) as a field
        min_iterations (int): Iterate at least this many times
        max_iterations (int): Iterate at most this many times
        multiple (bool): passed to :py:func:`cluster`
        sep_val (int): Value of bitmask that indicates a separator
        plot (bool): Make a 2D plot of Fld and the sep candidates
        mask_limit (int): if > 0, then bitmask fld with mask_limit,
            i.e., fld = fld & mask_limit (bitwise and)
        periodic (sequence): indicate whether that direction is
            periodic, and if so, whether the coordinate arrays are
            overlapped or not. Values can be True, False, or '+'. '+'
            indicates that x[0] and x[-1] are not colocated, so assume
            they're dx apart where dx = x[-1] - x[-2].
        pt_bnd (sequence): Boundaries that come to a point, i.e., all
            values along that boundary are neighbors such as the poles
            of a sphere. Specified like "0-" for lower boundary of
            dimension 0 or "1+" for the upper boundary of dimension 1.

    Returns:
        ndarray: 3xN for N clusters of separator points in the same
        coordinates as `fld`
    """
    pd = [False if pi == "0" else bool(pi) for pi in periodic]
    fld = fld.slice_reduce(":")
    if mask_limit:
        fld = np.bitwise_and(fld, mask_limit)
    a = fld.data
    x, y = fld.get_crds()

    for i in range(max_iterations):
        if pd[0]:
            a[(0, -1), :] |= a[(-1, 0), :]
        if pd[1]:
            a[:, (0, -1)] |= a[:, (-1, 0)]

        a = (a[ :-1,  :-1] | a[ :-1, 1:  ] |  # pylint: disable=bad-whitespace
             a[1:  ,  :-1] | a[1:  , 1:  ])   # pylint: disable=bad-whitespace
        x = 0.5 * (x[1:] + x[:-1])
        y = 0.5 * (y[1:] + y[:-1])

        # bitwise_or an entire bounary if all points are neighbors, like
        # at the poles of a sphere
        for bnd in pt_bnds:
            slc = [slice(None), slice(None)]
            slc[int(bnd[0])] = -1 if bnd[1] == "+" else 0
            a[slc] = np.bitwise_or.reduce(a[slc])

        indx, indy = np.where(a == sep_val)
        if i + 1 >= min_iterations and len(indx):
            break

    pts = cluster(indx, indy, x, y, multiple=multiple, periodic=periodic)

    if plot:
        from viscid.plot import mpl

        mpl.clf()
        mpl.subplot(121)
        mpl.plot(fld, title=True)

        mpl.subplot(122)
        or_fld = viscid.arrays2field(a, (x, y), name="OR")
        mpl.plot(or_fld, title=True)

        _x, _y = or_fld.get_crds()
        mpl.plt.plot(_x[indx], _y[indy], 'ko')
        # mpl.plt.show()

        mpl.plt.plot(pts[0], pts[1], 'y^')
        mpl.plt.show()

    return pts

def find_sep_points_spherical(fld, theta_phi=False, overlap=False, cap=False,
                              **kwargs):
    """Wrap :py:func:`find_sep_points_cartesian` for spheres and caps

    This is kind of a janky interface since data about
    theta_phi / overlap / cap could exist in the field
    """
    pd = "1" if overlap else "+"
    if theta_phi:
        kwargs['periodic'] = (False, pd)
        # kwargs['pt_bnds'] = ("0-",) if cap else ("0-", "0+")
    else:
        kwargs['periodic'] = (pd, False)
        # kwargs['pt_bnds'] = ("1-",) if cap else ("1-", "1+")
    return find_sep_points_cartesian(fld, **kwargs)

def _main():
    # test clustering; x == theta && y == phi
    Nx, Lx = 16, 180.0
    Ny, Ly = 32, 360.0
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny, endpoint=True)
    ix = [0,      0, Nx - 1, Nx - 1, Nx - 4]  # pylint: disable=bad-whitespace
    iy = [0, Ny - 1,      0, Ny - 1, Ny - 4]  # pylint: disable=bad-whitespace
    pts = cluster(ix, iy, x, y, multiple=True, periodic="01")
    print(pts.shape)
    print(pts)

if __name__ == "__main__":
    import sys
    sys.exit(_main())
