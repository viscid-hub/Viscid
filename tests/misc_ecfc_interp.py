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

import matplotlib.pyplot as plt
import numpy as np

import viscid
from viscid.plot import vpyplot as vlt


def compare_vectors(orig_fld, cmp_fld, catol=1e-8, rtol=2e-6,
                          trim_slc='x=2:-2, y=2:-2, z=2:-2', make_plots=False):
    # mag_shape = list(orig_fld.sshape) + [1]
    mag = viscid.magnitude(orig_fld)  # .data.reshape(mag_shape)
    reldiff = (cmp_fld - orig_fld) / mag
    reldiff = reldiff[trim_slc]
    reldiff.name = cmp_fld.name + " - " + orig_fld.name
    reldiff.pretty_name = cmp_fld.pretty_name + " - " + orig_fld.pretty_name
    comp_beyond_limit = [False] * 3

    for i, d in enumerate('xyz'):
        abs_max_rel_diff = np.nanmax(np.abs(reldiff[d]))
        max_crd_diff = np.max(orig_fld.get_crd(d) - cmp_fld.get_crd(d))
        print("{0}{1}, max absolute relative diff: {2:.3e} ({1} crds: {3:.1e})"
              "".format(cmp_fld.name, d, abs_max_rel_diff, max_crd_diff))
        if abs_max_rel_diff > rtol or abs(max_crd_diff) > catol:
            comp_beyond_limit[i] = True

        # plot differences?
        if make_plots:
            plt.clf()
            ax1 = plt.subplot(311)
            vlt.plot(orig_fld[d]['y=0f'], symmetric=True, earth=True)
            plt.subplot(312, sharex=ax1, sharey=ax1)
            vlt.plot(cmp_fld[d]['y=0f'], symmetric=True, earth=True)
            plt.subplot(313, sharex=ax1, sharey=ax1)
            vlt.plot(reldiff[d]['y=0f'], symmetric=True, earth=True)
            vlt.show()

    # if any(comp_beyond_limit):
    #     raise RuntimeError("Tolerance exceeded on ->CC accuracy")


def main():
    mhd_type = "C3"
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

    b = f['b_cc']
    b1 = f['b_fc']
    e_cc = f['e_cc']
    e_ec = f['e_ec']
    # divb =  f['divB']

    # viscid.interact()

    if True:
        bD = viscid.empty_like(b)
        bD.data = np.array(b.data)

        b1D = viscid.empty_like(b1)
        b1D.data = np.array(b1.data)

        mask5 = viscid.make_spherical_mask(bD, rmax=3.5)
        mask1_5 = viscid.make_spherical_mask(bD, rmax=1.5)
        viscid.fill_dipole(bD, mask=mask5)
        viscid.set_in_region(bD, bD, 0.0, 0.0, mask=mask1_5, out=bD)

        # compare_vectors(_b, bD, make_plots=True)
        mask5 = viscid.make_spherical_mask(b1D, rmax=3.5)
        mask1_5 = viscid.make_spherical_mask(b1D, rmax=1.5)
        viscid.fill_dipole(b1D, mask=mask5)
        viscid.set_in_region(b1D, b1D, 0.0, 0.0, mask=mask1_5, out=b1D)

        compare_vectors(bD["x=1:-1, y=1:-1, z=1:-1"], b1D.as_cell_centered(),
                        make_plots=True)

        # plt.clf()
        # dkwargs = dict(symmetric=True, earth=True, clim=(-1e2, 1e2))
        # ax1 = plt.subplot(311)
        # vlt.plot(viscid.div(b1)['y=0f'], **dkwargs)
        # plt.subplot(312, sharex=ax1, sharey=ax1)
        # vlt.plot(viscid.div(b)['y=0f'], **dkwargs)
        # plt.subplot(313, sharex=ax1, sharey=ax1)
        # vlt.plot(viscid.div(b1D)['y=0f'], **dkwargs)
        # vlt.show()

        bD = b1D = mask5 = mask1_5 = None

    # straight up interpolate b1 to cc crds and compare with b
    if True:
        b1_cc = viscid.interp_trilin(b1, b).as_flat()

        viscid.set_in_region(b, b, alpha=0.0, beta=0.0, out=b,
                             mask=viscid.make_spherical_mask(b, rmax=5.0))
        viscid.set_in_region(b1_cc, b1_cc, alpha=0.0, beta=0.0, out=b1_cc,
                             mask=viscid.make_spherical_mask(b1_cc, rmax=5.0))

        compare_vectors(b, b1_cc, make_plots=True)

    # make div?
    if True:
        # make seeds for 1.5x supersampling b1
        n = 128
        seeds = viscid.Volume((5.1, -0.02, -5.0), (12.0, 0.02, 5.0), (n, 3, n))
        # do interpolation onto new seeds
        b2 = viscid.interp_trilin(b1, seeds)

        div_b = viscid.div(b)
        div_b1 = viscid.div(b1)
        div_b2 = viscid.div(b2)

        viscid.set_in_region(div_b, div_b, alpha=0.0, beta=0.0, out=div_b,
                             mask=viscid.make_spherical_mask(div_b, rmax=5.0))
        viscid.set_in_region(div_b1, div_b1, alpha=0.0, beta=0.0, out=div_b1,
                             mask=viscid.make_spherical_mask(div_b1, rmax=5.0))
        viscid.set_in_region(div_b2, div_b2, alpha=0.0, beta=0.0, out=div_b2,
                             mask=viscid.make_spherical_mask(div_b2, rmax=5.0))
        viscid.set_in_region(divb, divb, alpha=0.0, beta=0.0, out=divb,
                             mask=viscid.make_spherical_mask(divb, rmax=5.0))

        plt.clf()
        ax1 = vlt.subplot(311)
        vlt.plot(div_b['y=0f'], symmetric=True, earth=True)
        vlt.subplot(312, sharex=ax1, sharey=ax1)
        # vlt.plot(div_b1['y=0f'], symmetric=True, earth=True)
        vlt.plot(div_b2['y=0f'], symmetric=True, earth=True)
        vlt.subplot(313, sharex=ax1, sharey=ax1)
        vlt.plot(divb['y=0f'], symmetric=True, earth=True)

        vlt.show()

    return 0

if __name__ == "__main__":
    sys.exit(main())

##
## EOF
##
