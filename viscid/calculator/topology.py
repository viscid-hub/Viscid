""" I don't know if this is worth keeping as its own module,
TOPOLOGY_* is copied here so that one can import this module
without needing to have built the cython module streamline.pyx """

TOPOLOGY_NONE = 0  # no translation needed
TOPOLOGY_CLOSED = 1  # translated from 5, 6, 7(4|5|6)
TOPOLOGY_OPEN_NORTH = 2  # translated from 13 (8|5)
TOPOLOGY_OPEN_SOUTH = 4  # translated from 14 (8|6)
TOPOLOGY_SW = 8  # no translation needed
# TOPOLOGY_CYCLIC = 16  # no translation needed
TOPOLOGY_INVALID = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
TOPOLOGY_OTHER = range(32, 512)  # >= 16

def color_from_topology(topology):
    """ topology is an integer described by calculator.streamline.TOPOLOGY_*
    returns a color tuple
    Note: to override this color scheme, just set
          topoloy.color_from_topology = some_function """
    if topology == TOPOLOGY_CLOSED:
        return (0.0, 0.8, 0.0)
    elif topology == TOPOLOGY_OPEN_NORTH:
        return (0.0, 0.0, 0.7)
    elif topology == TOPOLOGY_OPEN_SOUTH:
        return (0.7, 0.0, 0.0)
    elif topology == TOPOLOGY_SW:
        return (0.7, 0.7, 0.7)
    # elif topology == TOPOLOGY_CYCLIC:
    #     return (1.0, 1.0, 1.0)
    else:
        return (0.0, 0.0, 0.0)
