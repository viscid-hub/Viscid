#!/usr/bin/env python
"""Test extracting magnetopause info from 3d OpenGGCM data"""

from __future__ import print_function
import argparse
import os
import sys
import warnings

from viscid_test_common import xfail

import numpy as np
import viscid
from viscid import sample_dir
from viscid import vutil

try:
    from scipy.optimize import OptimizeWarning
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# The magnetopause tools only work in GSE crds. Usually these flags
# should be set in your rc file. See the corresponding page in the
# tutorial for more information
viscid.readers.openggcm.GGCMFile.read_log_file = True
viscid.readers.openggcm.GGCMGrid.mhd_to_gse_on_read = "auto"


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    _ = vutil.common_argparse(parser)

    if _HAS_SCIPY:
        warnings.filterwarnings("ignore", category=OptimizeWarning)

    f = viscid.load_file(os.path.join(sample_dir, 'sample_xdmf.3d.[0].xdmf'))
    mp = viscid.get_mp_info(f['pp'], f['b'], f['j'], f['e_cc'], fit='mp_xloc',
                            slc="x=7j:12.0j, y=-6j:6j, z=-6j:6j",
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
    sys.exit(_main())

##
## EOF
##
