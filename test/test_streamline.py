#!/usr/bin/env python

from __future__ import print_function
import argparse

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import numexpr as ne

from viscid import field
from viscid import coordinate
from viscid import readers
from viscid.calculator import cycalc
# from viscid.plot import mpl

verb = 0

def get_dipole(m=[0.0, 0.0, 1.0]):
    dtype = 'float32'
    x = np.array(np.linspace(-10, 10, 100), dtype=dtype)
    y = np.array(np.linspace(-10, 10, 100), dtype=dtype)
    z = np.array(np.linspace(-10, 10, 100), dtype=dtype)
    crds = coordinate.wrap_crds("Rectilinear", (('z', z), ('y', y), ('x', x)))

    # half = np.array([0.5], dtype=dtype) #pylint: disable=W0612
    # two = np.array([2.0], dtype=dtype) #pylint: disable=W0612
    m = np.array(m, dtype=dtype)
    mx, my, mz = m

    Zcc, Ycc, Xcc = crds.get_cc(shaped=True) #pylint: disable=W0612

    rsq = ne.evaluate("Xcc**2 + Ycc**2 + Zcc**2")
    Bx = ne.evaluate("rsq**-1.5 * (3 * Xcc * (mx * Xcc + my * Ycc + mz * Zcc) - mx / rsq)")
    By = ne.evaluate("rsq**-1.5 * (3 * Ycc * (mx * Xcc + my * Ycc + mz * Zcc) - my / rsq)")
    Bz = ne.evaluate("rsq**-1.5 * (3 * Zcc * (mx * Xcc + my * Ycc + mz * Zcc) - mz / rsq)")

    fld = field.VectorField("v_cc", crds, [Bx, By, Bz],
                            force_layout=field.LAYOUT_INTERLACED,
                            center="Cell", forget_source=True)
    return fld

def main():
    parser = argparse.ArgumentParser(description="Load some data files")
    parser.add_argument('--show', '-s', action='store_true',
                        help='increase verbosity')
    parser.add_argument('-v', action='count', default=0,
                        help='increase verbosity')
    parser.add_argument('-q', action='count', default=0,
                        help='decrease verbosity')
    parser.add_argument('files', nargs="*", help='input files')
    args = parser.parse_args()
    verb = args.v - args.q

    B = get_dipole()
    lines = cycalc.streamlines(B, [[-5.0, -5.0, 0.0],
                                   [-5.0, 0.0, 0.0],
                                   [-5.0, 5.0, 0.0],
                                  ])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for line in lines:
        z, y, x = np.array(line)
        ax.plot(x, y, z)
    plt.show()


if __name__ == "__main__":
    main()

##
## EOF
##
