#!/usr/bin/env python
""" test loading a numpy npz binary file """

from __future__ import print_function
import os
import argparse

from matplotlib import pyplot as plt
import numpy as np

from viscid_test_common import next_plot_fname

import viscid
from viscid import vutil
from viscid.plot import vpyplot as vlt


def main():
    parser = argparse.ArgumentParser(description="Test xdmf")
    parser.add_argument("--show", "--plot", action="store_true")
    parser.add_argument("--keep", action="store_true")
    args = vutil.common_argparse(parser)

    # setup a simple force free field
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2.5, 2.5, 25)
    z = np.linspace(-3, 3, 30)
    psi = viscid.empty([x, y, z], name='psi', center='node')
    b = viscid.empty([x, y, z], nr_comps=3, name='b', center='cell',
                     layout='interlaced')

    X, Y, Z = psi.get_crds_nc("xyz", shaped=True)
    Xcc, Ycc, Zcc = psi.get_crds_cc("xyz", shaped=True)
    psi[:, :, :] = 0.5 * (X**2 + Y**2 - Z**2)
    b['x'] = Xcc
    b['y'] = Ycc
    b['z'] = -Zcc

    fname = os.path.join(viscid.sample_dir, 'test.npz')
    viscid.save_fields(fname, [psi, b])

    f = viscid.load_file(fname)
    plt.subplot(131)
    vlt.plot(f['psi'], "y=0")
    plt.subplot(132)
    vlt.plot(f['b'].component_fields()[0], "y=0")
    plt.subplot(133)
    vlt.plot(f['b'].component_fields()[2], "y=0")

    plt.savefig(next_plot_fname(__file__))
    if args.show:
        plt.show()

    if not args.keep:
        os.remove(fname)

if __name__ == "__main__":
    main()

##
## EOF
##
