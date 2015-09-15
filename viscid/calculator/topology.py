"""I don't know if this is worth keeping as its own module,
TOPOLOGY_* is copied here so that one can import this module
without needing to have built the cython module streamline.pyx
"""

import numpy as np

TOPOLOGY_MS_NONE = 0  # no translation needed
TOPOLOGY_MS_CLOSED = 1  # translated from 5, 6, 7(4|5|6)
TOPOLOGY_MS_OPEN_NORTH = 2  # translated from 13 (8|5)
TOPOLOGY_MS_OPEN_SOUTH = 4  # translated from 14 (8|6)
TOPOLOGY_MS_SW = 8  # no translation needed
# TOPOLOGY_MS_CYCLIC = 16  # no translation needed
TOPOLOGY_MS_INVALID = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
# TOPOLOGY_MS_OTHER = list(range(32, 512))  # >= 16

TOPOLOGY_G_NONE = 0

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
