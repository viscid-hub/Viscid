#!/usr/bin/env python
"""Test transformations between map representations

Results are verrified as accurate by assert.
"""

from __future__ import print_function
import argparse
import sys
import os

from viscid_test_common import assert_similar

import viscid
from viscid import sample_dir
from viscid import vutil


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = vutil.common_argparse(parser)  # pylint: disable=unused-variable

    fiof = viscid.load_file(os.path.join(sample_dir, 'sample_xdmf.iof.xdmf'))

    fac = fiof['fac_tot']

    fac_mf = viscid.as_mapfield(fac)
    fac_sphere = viscid.as_spherefield(fac)
    if fac_sphere is not fac:
        raise RuntimeError("cenversion should have been a noop")

    fac_mf_rad = viscid.as_mapfield(fac, units='rad')
    fac_sphere_rad = viscid.as_spherefield(fac, units='rad')

    fac_mf_sphere = viscid.as_spherefield(fac_mf)
    assert_similar(fac_mf_sphere, fac)

    fac_mf_sphere_deg_rad = viscid.as_spherefield(fac_mf, units='rad')
    assert_similar(fac_mf_sphere_deg_rad, fac_sphere_rad)

    fac_mf_sphere_rad_deg = viscid.as_spherefield(fac_mf_rad, units='deg')
    assert_similar(fac_mf_sphere_rad_deg, fac)

    fac_sphere_mf_rad_deg = viscid.as_mapfield(fac_sphere_rad, units='deg')
    assert_similar(fac_sphere_mf_rad_deg, fac_mf)

    fac_sphere_mf_rad_rad = viscid.as_mapfield(fac_sphere_rad, units='rad')
    assert_similar(fac_sphere_mf_rad_rad, fac_mf_rad)

    fac_mf_T = viscid.as_mapfield(fac, order=('lat', 'lon'))
    assert_similar(fac_mf_T, fac_mf.T)

    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
