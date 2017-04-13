#!/usr/bin/env python
# pylint: disable=unused-import,wildcard-import,unused-wildcard-import
from __future__ import print_function
import sys

print("Viscid says: from viscid import *")
from viscid import *
import viscid

print("Viscid says: importing numpy as np")
import numpy as np

if ('mpl' in sys.argv or 'pylab' in sys.argv or 'matplotlib' in sys.argv or
    'vpyplot' in sys.argv or 'vlt' in sys.argv):
    print("Viscid says: from viscid.plot import vpyplot as vlt")
    from viscid.plot import vpyplot as vlt
    print("Viscid says: import matplotlib")
    import matplotlib
    print("Viscid says: import matplotlib as mpl")
    import matplotlib as mpl
    print("Viscid says: from matplotlib import pyplot as plt")
    from matplotlib import pyplot as plt

if 'mvi' in sys.argv or 'mlab' in sys.argv or 'mayavi' in sys.argv:
    print("Viscid says: from viscid.plot import mvi")
    from viscid.plot import mvi
    print("Viscid says: from mayavi import mlab")
    from mayavi import mlab
