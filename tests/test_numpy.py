#!/usr/bin/env python
"""Test sending Fields through Numpy functions"""

from __future__ import print_function
import sys
import argparse
# import resource

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm

from viscid_test_common import assert_similar, assert_different, next_plot_fname

import viscid
from viscid import vutil
from viscid.plot import mpl


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)
    dtype = 'float32'

    ########################################################################
    # hard core test transpose (since this is used for mapfield transforms)
    x = np.array(np.linspace( 1, -1, 9), dtype=dtype)
    y = np.array(np.linspace(-1,  1, 9), dtype=dtype)
    z = np.array(np.linspace(-1,  1, 9), dtype=dtype)
    vI1 = viscid.empty([x, y, z], nr_comps=3, name='V', center='cell',
                       layout='interlaced')
    vI2 = viscid.empty([z, y, x], nr_comps=3, name='V', center='cell',
                       layout='interlaced', crd_names='zyx')
    vF1 = viscid.empty([x, y, z], nr_comps=3, name='V', center='cell',
                       layout='flat')
    vF2 = viscid.empty([z, y, x], nr_comps=3, name='V', center='cell',
                       layout='flat', crd_names='zyx')
    X1, Y1, Z1 = vI1.get_crds_cc(shaped=True)
    X2, Y2, Z2 = vI2.get_crds_cc(shaped=True)

    for v in (vI1, vF1):
        v['x'] = (0.5 * X1) + (0.0 * Y1) + (0.0 * Z1)
        v['y'] = (0.0 * X1) + (0.5 * Y1) + (0.5 * Z1)
        v['z'] = (0.0 * X1) + (0.5 * Y1) + (0.5 * Z1)
    for v in (vI2, vF2):
        v['z'] = (0.5 * X2) + (0.5 * Y2) + (0.0 * Z2)
        v['y'] = (0.5 * X2) + (0.5 * Y2) + (0.0 * Z2)
        v['x'] = (0.0 * X2) + (0.0 * Y2) + (0.5 * Z2)

    assert_different(vI1, vI2)

    # test some straight up transposes of both interlaced and flat fields
    assert_similar(vI1.spatial_transpose(), vI2)
    assert_similar(vI1.ST, vI2)
    assert_similar(vF1.spatial_transpose(), vF2)
    assert_similar(vI1.transpose(), vF2)
    assert_similar(vI1.T, vF2)
    assert_different(vI1.transpose(), vI2)
    assert_similar(vF1.transpose(), vI2)
    assert_different(vF1.transpose(), vF2)
    assert_similar(np.transpose(vI1), vF2)
    assert_similar(np.transpose(vF1), vI2)

    # now specify specific axes using all 3 interfaces
    assert_similar(vI1.spatial_transpose('x', 'z', 'y'), vI1)
    assert_similar(np.transpose(vI1, axes=[0, 2, 1, 3]), vI1)
    assert_similar(vI1.transpose(0, 2, 1, 3), vI1)
    assert_similar(vF1.spatial_transpose('x', 'z', 'y'), vF1)
    assert_similar(np.transpose(vF1, axes=(0, 1, 3, 2)), vF1)
    assert_similar(vF1.transpose(0, 1, 3, 2), vF1)

    # now test swapaxes since that uses
    assert_similar(vI1.swapaxes(1, 2), vI1)
    assert_similar(np.swapaxes(vI1, 1, 2), vI1)
    assert_similar(vI1.swap_crd_axes('y', 'z'), vI1)

    ##############################
    # test some other mathy stuff
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
    sys.exit(_main())

##
## EOF
##
