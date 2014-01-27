#!/usr/bin/env python
""" Test Fields' numpy compatability """

from __future__ import print_function
import sys
import os
from time import time
import logging
import argparse
# import resource

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm

_viscid_root = os.path.realpath(os.path.dirname(__file__) + '/../src/viscid/')
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)

from viscid import vutil
from viscid import field
from viscid import coordinate
from viscid.calculator import calc
from viscid.plot import mpl

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)
    dtype = 'float32'

    x = np.array(np.linspace(-1, 1, 2), dtype=dtype)
    y = np.array(np.linspace(-2, 2, 30), dtype=dtype)
    z = np.array(np.linspace(-5, 5, 90), dtype=dtype)

    crds = coordinate.wrap_crds("Rectilinear", (('z', z), ('y', y),
                                                ('x', x)))
    Z, Y, X = crds.get_crds_cc(shaped=True)

    vx = 0.5 * X**2 +       Y    + 0.0 * Z
    vy = 0.0 * X    + 0.5 * Y**2 + 0.0 * Z
    vz = 0.0 * X    + 0.0 * Y    + 0.5 * Z**2

    v = field.VectorField("v", crds, [vx, vy, vz],
                              center="Cell", forget_source=True,
                              deep_meta={"force_layout": field.LAYOUT_INTERLACED},
                             )
    vx, vy, vz = v.component_fields()
    mag = calc.magnitude(v)
    another = np.transpose(mag)

    plt.subplot(151)
    mpl.plot(vx)
    plt.subplot(152)
    mpl.plot(vy)
    plt.subplot(153)
    mpl.plot(vz)
    plt.subplot(154)
    mpl.plot(mag)
    plt.subplot(155)
    mpl.plot(another)

    if args.show:
        plt.show()

    return 0

if __name__ == "__main__":
    errcode = main()
    sys.exit(errcode)

##
## EOF
##
