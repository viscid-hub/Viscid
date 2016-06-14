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


def compare_vectors(cc_fld, ecfc_fld, to_cc_fn, catol=1e-8, rtol=2e-6,
                    trim_slc='x=1:-1, y=1:-1, z=1:-1', make_plots=False):
    trimmed = cc_fld[trim_slc]
    cc = to_cc_fn(ecfc_fld)
    reldiff = (cc - trimmed) / viscid.magnitude(trimmed)
    reldiff = reldiff["x=1:-1, y=1:-1, z=1:-1"]
    reldiff.name = cc.name + " - " + trimmed.name
    reldiff.pretty_name = cc.pretty_name + " - " + trimmed.pretty_name
    comp_beyond_limit = [False] * 3

    for i, d in enumerate('xyz'):
        abs_max_rel_diff = np.nanmax(np.abs(reldiff[d]))
        max_crd_diff = np.max(trimmed.get_crd(d) - cc.get_crd(d))
        print("{0}{1}, max absolute relative diff: {2:.3e} ({1} crds: {3:.1e})"
              "".format(cc_fld.name, d, abs_max_rel_diff, max_crd_diff))
        if abs_max_rel_diff > rtol or abs(max_crd_diff) > catol:
            comp_beyond_limit[i] = True

        # plot differences?
        if make_plots:
            ax1 = mpl.plt.subplot(311)
            mpl.plot(cc_fld[d]['y=0f'], symmetric=True, earth=True)
            mpl.plt.subplot(312, sharex=ax1, sharey=ax1)
            mpl.plot(cc[d]['y=0f'], symmetric=True, earth=True)
            mpl.plt.subplot(313, sharex=ax1, sharey=ax1)
            mpl.plot(reldiff[d]['y=0f'], symmetric=True, earth=True)
            mpl.show()

    if any(comp_beyond_limit):
        raise RuntimeError("Tolerance exceeded on ->CC accuracy")

def main():
    mhd_type = "C"
    make_plots = 1
    test_fc = 1
    test_ec = 1
    test_div = 1

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

    ISLICE = slice(None)
    # ISLICE = 'y=0f:0.15f'

    # #################
    # # test out fc2cc
    if test_fc:
        b = f['b'][ISLICE]
        b1 = f['b1'][ISLICE]

        compare_vectors(b, b1, viscid.fc2cc, catol=catol, rtol=rtol,
                        make_plots=make_plots)

    #################
    # test out ec2cc
    if test_ec:
        e_cc = f['e_cc'][ISLICE]
        e_ec = f['e_ec'][ISLICE]

        if mhd_type not in ("F", "FORTRAN"):
            compare_vectors(e_cc, e_ec, viscid.ec2cc, catol=catol, rtol=rtol,
                            make_plots=make_plots)

    #################
    # test out divfc
    # Note: Relative error on Div B is meaningless b/c the coordinates
    #       are not the same up to order (dx/4) I think. You can see this
    #       since (fcdiv - divb_trimmed) is both noisy and stripy
    if test_div:
        if mhd_type not in ("F", "FORTRAN"):
            b1 = f['b1'][ISLICE]
            divb = f['divB'][ISLICE]
            trimmed = divb['x=1:-1, y=1:-1, z=1:-1']

            divb1 = viscid.div_fc(b1)

            viscid.set_in_region(trimmed, trimmed, alpha=0.0, beta=0.0, out=trimmed,
                                 mask=viscid.make_spherical_mask(trimmed, rmax=5.0))
            viscid.set_in_region(divb1, divb1, alpha=0.0, beta=0.0, out=divb1,
                                 mask=viscid.make_spherical_mask(divb1, rmax=5.0))

            reldiff = (divb1 - trimmed) / viscid.magnitude(viscid.fc2cc(b1))
            reldiff = reldiff["x=1:-1, y=1:-1, z=1:-1"]
            reldiff.name = divb1.name + " - " + trimmed.name
            reldiff.pretty_name = divb1.pretty_name + " - " + trimmed.pretty_name

            abs_max_rel_diff = np.nanmax(np.abs(reldiff))
            max_crd_diff = [0.0] * 3
            for i, d in enumerate('xyz'):
                max_crd_diff[i] = np.max(trimmed.get_crd(d) - divb1.get_crd(d))
            print("divB max absolute relative diff: {0:.3e} "
                  "(crds: X: {1[0]:.3e}, Y: {1[1]:.3e}, Z: {1[2]:.3e})"
                  "".format(abs_max_rel_diff, max_crd_diff))

            # plot differences?
            if make_plots:
                ax1 = mpl.plt.subplot(311)
                mpl.plot(divb['y=0f'], symmetric=True, earth=True)
                mpl.plt.subplot(312, sharex=ax1, sharey=ax1)
                mpl.plot(divb1['y=0f'], symmetric=True, earth=True)
                mpl.plt.subplot(313, sharex=ax1, sharey=ax1)
                mpl.plot(reldiff['y=0f'], symmetric=True, earth=True)
                mpl.show()

            # Since the coordinates will be different by order dx^2 (i think),
            # there is no way to compare the divB from simulation with the
            # one we get here. However, they should be the same up to a few %, and
            # down to noise level with stripes of enhanced noise. These stripes
            # are the errors in the coordinate values (since the output only
            # gives us weird nc = averaged cc locations)
            #
            # if abs_max_rel_diff > rtol or np.any(np.abs(max_crd_diff) > catol):
            #     raise RuntimeError("Tolerance exceeded on divB calculation")

    return 0

if __name__ == "__main__":
    sys.exit(main())

##
## EOF
##
