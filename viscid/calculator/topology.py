"""I don't know if this is worth keeping as its own module,
TOPOLOGY_* is copied here so that one can import this module
without needing to have built the cython module streamline.pyx
"""

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

def cluster(indx, indy, x, y, multiple=True):
    """Cluster and average groups of neighboring points

    TODO: If absolutely necessary, could do some K-means clustering
        here by calling into scikit-learn.

    Args:
        indx (list): list of x indices
        indy (list): list of y indices
        x (list): list of x locations (same size as indx)
        y (list): list of y locations (same size as indy)
        multiple (bool): If False, average all points as a single
            cluster

    Returns:
        ndarray: 3xN for N clusters
    """
    # find clusters of points
    clusters = []
    if multiple:
        for ix, iy in zip(indx, indy):
            clustered = False
            for cl in clusters:
                for clix, cliy in zip(cl[0], cl[1]):
                    if np.abs(ix - clix) <= 1 and np.abs(iy - cliy) <= 1:
                        clustered = True
                        cl[0].append(ix)
                        cl[1].append(iy)
                        break
                if clustered:
                    break
            if not clustered:
                clusters.append([[ix], [iy]])
    else:
        if len(indy) > 0:
            clusters = [[indx, indy]]

    pts_x = np.array([np.average(x[cl[0]]) for cl in clusters])
    pts_y = np.array([np.average(y[cl[1]]) for cl in clusters])

    return np.array([pts_x, pts_y])

def find_sep_points_cartesian(fld, min_iterations=1, max_iterations=10,
                              multiple=True, sep_val=SEP_VAL, plot=False,
                              mask_limit=0b1111):
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

    Returns:
        ndarray: 3xN for N clusters of separator points in the same
        coordinates as `fld`
    """
    fld = fld.slice_reduce(":")
    if mask_limit:
        fld = np.bitwise_and(fld, mask_limit)
    a = fld.data
    x, y = fld.get_crds()

    for i in range(max_iterations):
        a = (a[ :-1,  :-1] | a[ :-1, 1:  ] |  # pylint: disable=bad-whitespace
             a[1:  ,  :-1] | a[1:  , 1:  ])   # pylint: disable=bad-whitespace
        x = 0.5 * (x[1:] + x[:-1])
        y = 0.5 * (y[1:] + y[:-1])

        indx, indy = np.where(a == sep_val)
        if i + 1 >= min_iterations and len(indx):
            break

    pts = cluster(indx, indy, x, y, multiple=multiple)

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
