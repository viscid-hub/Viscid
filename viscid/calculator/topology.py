"""I don't know if this is worth keeping as its own module,
TOPOLOGY_* is copied here so that one can import this module
without needing to have built the cython module streamline.pyx
"""

from __future__ import print_function

import numpy as np
import viscid
from viscid.cython import streamline


__all__ = ['topology2color', 'color_map_msphere', 'color_map_generic',
           'color_map']

# TOPOLOGY_MS_INVALID = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
# TOPOLOGY_G_NONE = 0

color_map_msphere = {streamline.TOPOLOGY_MS_CLOSED: (0.0, 0.8, 0.0),
                     streamline.TOPOLOGY_MS_OPEN_NORTH: (0.0, 0.0, 0.7),
                     streamline.TOPOLOGY_MS_OPEN_SOUTH: (0.7, 0.0, 0.0),
                     streamline.TOPOLOGY_MS_SW: (0.7, 0.7, 0.7)
                    }
color_map_generic = {}

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

##
## EOF
##
