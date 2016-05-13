#!/usr/bin/env python
""" test a ggcm grid wrapper """

from __future__ import print_function
import argparse

from viscid_test_common import sample_dir, assert_similar

import viscid
from viscid import vutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = vutil.common_argparse(parser)  # pylint: disable=unused-variable

    fiof = viscid.load_file(sample_dir + '/sample_xdmf.iof.xdmf')

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

if __name__ == "__main__":
    main()

##
## EOF
##
