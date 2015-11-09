#!/usr/bin/env python
""" Test Fields' numpy compatability """

from __future__ import print_function
import sys
import argparse
# import resource

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm

from viscid_test_common import next_plot_fname

import viscid
from viscid import vutil
from viscid.plot import mpl

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)
    dtype = 'float32'

    x = np.array(np.linspace(-1, 1, 2), dtype=dtype)
    y = np.array(np.linspace(-2, 2, 30), dtype=dtype)
    z = np.array(np.linspace(-5, 5, 90), dtype=dtype)
    v = viscid.empty([x, y, z], nr_comps=3, name='V', center='cell',
                     layout='interlaced')
    X, Y, Z = v.get_crds_cc(shaped=True)

    v['x'] = (0.5 * X**2) + (      Y   ) + (0.0 * Z   )
    v['y'] = (0.0 * X   ) + (0.5 * Y**2) + (0.0 * Z   )
    v['z'] = (0.0 * X   ) + (0.0 * Y   ) + (0.5 * Z**2)

    mag = viscid.magnitude(v)
    mag2 = np.sqrt(np.sum(v * v, axis=v.nr_comp))
    another = np.transpose(mag)

    plt.subplot(151)
    mpl.plot(v['x'])
    plt.subplot(152)
    mpl.plot(v['y'])
    plt.subplot(153)
    mpl.plot(mag)
    plt.subplot(154)
    mpl.plot(mag2)
    plt.subplot(155)
    mpl.plot(another)

    mpl.plt.savefig(next_plot_fname(__file__))
    if args.show:
        plt.show()

    return 0

if __name__ == "__main__":
    errcode = main()
    sys.exit(errcode)

##
## EOF
##
