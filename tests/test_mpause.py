#!/usr/bin/env python
""" Try to convert a Field to a mayavi type and plot
streamlines or something """

from __future__ import print_function
import argparse
import sys

from viscid_test_common import sample_dir, xfail

import numpy as np
import viscid
from viscid import vutil


def main():
    parser = argparse.ArgumentParser(description="Test calc")
    _ = vutil.common_argparse(parser)

    f = viscid.load_file(sample_dir + '/sample_xdmf.3d.[0].xdmf')
    mp = viscid.get_mp_info(f['pp'], f['b'], f['j'], f['e_cc'], fit='mp_xloc',
                            slc="x=7f:12.0f, y=-6f:6f, z=-6f:6f",
                            cache=False)

    Y, Z = mp['pp_max_xloc'].meshgrid(prune=True)

    # # get normals from paraboloid surface
    if isinstance(mp['paraboloid'], viscid.DeferredImportError):
        xfail("Scipy not installed; paraboloid curve fitting not tested")
    else:
        parab_n = viscid.paraboloid_normal(Y, Z, *mp['paraboloid'][0])
        parab_n = parab_n.reshape(3, -1)

    # get normals from minvar
    minvar_y = Y.reshape(-1)
    minvar_z = Z.reshape(-1)
    minvar_n = np.zeros([3, len(minvar_y)])

    for i in range(minvar_n.shape[1]):
        p0 = [0.0, minvar_y[i], minvar_z[i]]
        p0[0] = mp['pp_max_xloc']['y={0[0]}f, z={0[1]}f'.format(p0)]
        lmn = viscid.find_minvar_lmn_around(f['b'], p0, l=2.0, n=64)
        minvar_n[:, i] = lmn[2, :]

    theta = (180 / np.pi) * np.arccos(np.sum(parab_n * minvar_n, axis=0))

    # make sure paraboloid normals and minvar normals are closeish
    # this is a poor check, but at least it's something
    assert np.min(theta) < 3.0
    assert np.average(theta) < 20.0
    assert np.median(theta) < 20.0
    assert np.max(theta) < 70.0

    return 0

if __name__ == "__main__":
    sys.exit(main())

##
## EOF
##
