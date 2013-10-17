""" I don't know if this is worth keeping as its own module """

TOPOLOGY_NONE = [0]
TOPOLOGY_INVALID = [1, 2, 3, 4, 9, 10, 11, 12, 15]
TOPOLOGY_CLOSED = [7, 5, 6]  # 5 (both N), 6 (both S), 7(both hemispheres)
TOPOLOGY_SW = [8]
TOPOLOGY_OPEN = [13, 14]
TOPOLOGY_OPEN_NORTH = [13]
TOPOLOGY_OPEN_SOUTH = [14]
TOPOLOGY_OTHER = range(16, 512)  # >= 16

def color_from_topology(topology):
    """ topology is an integer described by calculator.streamline.TOPOLOGY_*
    returns a color tuple
    Note: to override this color scheme, just set
          topoloy.color_from_topology = some_function """
    if topology in TOPOLOGY_INVALID:
        return (0.0, 0.0, 0.0)
    elif topology in TOPOLOGY_CLOSED:
        return (0.0, 0.8, 0.0)
    elif topology in TOPOLOGY_SW:
        return (0.5, 0.5, 0.5)
    elif topology in TOPOLOGY_OPEN_NORTH:
        return (0.0, 0.0, 0.7)
    elif topology in TOPOLOGY_OPEN_SOUTH:
        return (0.7, 0.0, 0.0)
    else:
        return (0.0, 0.0, 0.0)
