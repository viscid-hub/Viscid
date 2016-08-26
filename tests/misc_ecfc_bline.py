#!/usr/bin/env python
"""THIS TEST IS NOT RUN AUTOMATICALLY

This test depends on 3 runs:
      trillian.sr.unh.edu:~kmaynard/scratch/tests/tmedium{,2,3}

    Since these runs are non-negligable in size (a few hundred MB each),
    I'm opting not to include this test in the suite. That being said, this
    is the most comprehensive way to test cell / edge centered fields.
"""

from __future__ import division, print_function
import sys

import numpy as np

import viscid
from viscid.plot import mpl


def main():
    mhd_type = "C"
    make_plots = 1

    mhd_type = mhd_type.upper()
    if mhd_type.startswith("C"):
        if mhd_type in ("C",):
            f = viscid.load_file("$WORK/tmedium/*.3d.[-1].xdmf")
        elif mhd_type in ("C2", "C3"):
            f = viscid.load_file("$WORK/tmedium2/*.3d.[-1].xdmf")
        else:
            raise ValueError()
        catol = 1e-8
        rtol = 2e-6
    elif mhd_type in ("F", "FORTRAN"):
        f = viscid.load_file("$WORK/tmedium3/*.3df.[-1]")
        catol = 1e-8
        rtol = 7e-2
    else:
        raise ValueError()

    do_fill_dipole = True

    gslc = "x=-21.2f:12f, y=-11f:11f, z=-11f:11f"
    b = f['b_cc'][gslc]
    b1 = f['b_fc'][gslc]
    e_cc = f['e_cc'][gslc]
    e_ec = f['e_ec'][gslc]

    if do_fill_dipole:
        mask = viscid.make_spherical_mask(b, rmax=3.5)
        viscid.fill_dipole(b, mask=mask)

        mask = viscid.make_spherical_mask(b1, rmax=3.5)
        viscid.fill_dipole(b1, mask=mask)

        mask = None

    # seeds = viscid.SphericalCap(r=1.02, ntheta=64, nphi=32, angle0=17, angle=20,
    #                             philim=(100, 260), roll=-180.0)
    # seeds = viscid.SphericalCap(r=1.02, ntheta=64, nphi=32, angle0=17, angle=20,
    #                             philim=(0, 10), roll=0.0)
    seedsN = viscid.Sphere(r=1.02, ntheta=16, nphi=16, thetalim=(15, 25),
                           philim=(0, 300), crd_system=b)
    seedsS = viscid.Sphere(r=1.02, ntheta=16, nphi=16, thetalim=(155, 165),
                           philim=(0, 300), crd_system=b)

    bl_kwargs = dict(ibound=0.9, obound0=(-20, -10, -10), obound1=(11, 10, 10))

    # blines_cc, topo_cc = viscid.streamlines(b, seeds, **bl_kwargs)
    blinesN_fc, topoN_fc = viscid.streamlines(b1, seedsN, **bl_kwargs)
    _, topoS_fc = viscid.streamlines(b1, seedsS, output=viscid.OUTPUT_TOPOLOGY,
                                     **bl_kwargs)

    if True:
        from viscid.plot import mvi
        mesh = mvi.mesh_from_seeds(seedsN, scalars=topoN_fc)
        mesh.actor.property.backface_culling = True
        # mvi.plot_lines(blines_cc, scalars="#000000", tube_radius=0.03)
        mvi.plot_lines(blinesN_fc, scalars=viscid.topology2color(topoN_fc),
                       opacity=0.7)

        mvi.plot_blue_marble(r=1.0)
        mvi.plot_earth_3d(radius=1.01, crd_system=b.find_info('crd_system'),
                          night_only=True, opacity=0.5)
        mvi.show()

    if True:
        mpl.subplot(121, projection='polar')
        mpl.plot(topoN_fc)
        mpl.subplot(122, projection='polar')
        mpl.plot(topoS_fc)
        mpl.show()

    return 0

if __name__ == "__main__":
    sys.exit(main())

##
## EOF
##
