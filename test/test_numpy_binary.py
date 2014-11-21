#!/usr/bin/env python
""" test loading a numpy npz binary file """

from __future__ import print_function
import sys
import os
import argparse

import numpy as np

_viscid_root = os.path.realpath(os.path.dirname(__file__) + '/../viscid/')
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)

from viscid import vutil
from viscid import readers
from viscid import coordinate
from viscid import field
from viscid.plot import mpl
from viscid.plot.mpl import plt

def main():
    parser = argparse.ArgumentParser(description="Test xdmf")
    parser.add_argument("--show", "--plot", action="store_true")
    parser.add_argument("--keep", action="store_true")
    args = vutil.common_argparse(parser)

    # setup a simple force free field
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2.5, 2.5, 25)
    z = np.linspace(-3, 3, 30)
    crds = coordinate.wrap_crds("nonuniform_cartesian", [('z', z), ('y', y), ('x', x)])

    psi = field.empty("Scalar", "psi", crds, center="Node")
    b = field.empty("Vector", "b", crds, nr_comps=3,
                    layout=field.LAYOUT_INTERLACED, center="Cell")

    Z, Y, X = crds.get_crds_nc("zyx", shaped=True)
    Zcc, Ycc, Xcc = crds.get_crds_cc("zyx", shaped=True)
    psi[:, :, :] = 0.5 * (X**2 + Y**2 - Z**2)
    b[:, :, :, 0] = Xcc
    b[:, :, :, 1] = Ycc
    b[:, :, :, 2] = -Zcc

    fname = _viscid_root + '/../sample/test.npz'
    readers.save_fields(fname, [psi, b])

    f = readers.load_file(fname)
    plt.subplot(131)
    mpl.plot(f['psi'], "y=0")
    plt.subplot(132)
    mpl.plot(f['b'].component_fields()[0], "y=0")
    plt.subplot(133)
    mpl.plot(f['b'].component_fields()[2], "y=0")

    if args.show:
        plt.show()

    if not args.keep:
        os.remove(fname)

if __name__ == "__main__":
    main()

##
## EOF
##
