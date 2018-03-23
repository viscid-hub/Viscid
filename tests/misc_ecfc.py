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
from viscid.plot import vpyplot as vlt
import matplotlib.pyplot as plt


def compare_vectors(cc_fld, ecfc_fld, to_cc_fn, catol=1e-8, rtol=2.2e-6,
                    trim_slc=':', bnd=True, make_plots=False):
    trimmed = cc_fld[trim_slc]
    cc = to_cc_fn(ecfc_fld, bnd=bnd)
    reldiff = (cc - trimmed) / viscid.magnitude(trimmed)
    reldiff = reldiff["x=1:-1, y=1:-1, z=1:-1"]
    reldiff.name = cc.name + " - " + trimmed.name
    reldiff.pretty_name = cc.pretty_name + " - " + trimmed.pretty_name
    comp_beyond_limit = [False] * 3

    for i, d in enumerate('xyz'):
        abs_max_rel_diff = np.nanmax(np.abs(reldiff[d]))
        # trim first and last when diffing crds since they're extrapolated
        # in the ecfc field, so we already know they won't match up do dx
        max_crd_diff = np.max((trimmed.get_crd(d) - cc.get_crd(d))[1:-1])
        print("{0}{1}, max absolute relative diff: {2:.3e} ({1} crds: {3:.1e})"
              "".format(cc_fld.name, d, abs_max_rel_diff, max_crd_diff))
        if abs_max_rel_diff > rtol or abs(max_crd_diff) > catol:
            comp_beyond_limit[i] = True

        # plot differences?
        if make_plots:
            ax1 = plt.subplot(311)
            vlt.plot(cc_fld[d]['y=0f'], symmetric=True, earth=True)
            plt.subplot(312, sharex=ax1, sharey=ax1)
            vlt.plot(cc[d]['y=0f'], symmetric=True, earth=True)
            plt.subplot(313, sharex=ax1, sharey=ax1)
            vlt.plot(reldiff[d]['y=0f'], symmetric=True, earth=True)
            vlt.show()

    if any(comp_beyond_limit):
        raise RuntimeError("Tolerance exceeded on ->CC accuracy")

def main():
    mhd_type = "C"
    make_plots = 1
    test_fc = 1
    test_ec = 1
    test_div = 1
    test_interp = 1
    test_streamline = 1

    mhd_type = mhd_type.upper()
    if mhd_type.startswith("C"):
        if mhd_type in ("C",):
            f = viscid.load_file("$WORK/tmedium/*.3d.[-1].xdmf")
        elif mhd_type in ("C2", "C3"):
            f = viscid.load_file("$WORK/tmedium2/*.3d.[-1].xdmf")
        else:
            raise ValueError()
        catol = 1e-8
        rtol = 5e-6
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
        bnd = 0

        if mhd_type not in ("F", "FORTRAN"):
            b1 = f['b1'][ISLICE]
            divb = f['divB'][ISLICE]
            if bnd:
                trimmed = divb
            else:
                trimmed = divb['x=1:-1, y=1:-1, z=1:-1']
            b1mag = viscid.magnitude(viscid.fc2cc(b1, bnd=bnd))

            divb1 = viscid.div_fc(b1, bnd=bnd)

            viscid.set_in_region(trimmed, trimmed, alpha=0.0, beta=0.0, out=trimmed,
                                 mask=viscid.make_spherical_mask(trimmed, rmax=5.0))
            viscid.set_in_region(divb1, divb1, alpha=0.0, beta=0.0, out=divb1,
                                 mask=viscid.make_spherical_mask(divb1, rmax=5.0))

            reldiff = (divb1 - trimmed) / b1mag
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
                ax1 = plt.subplot(311)
                vlt.plot(divb['y=0f'], symmetric=True, earth=True)
                plt.subplot(312, sharex=ax1, sharey=ax1)
                vlt.plot(divb1['y=0f'], symmetric=True, earth=True)
                plt.subplot(313, sharex=ax1, sharey=ax1)
                vlt.plot(reldiff['y=0f'], symmetric=True, earth=True)
                vlt.show()

            # Since the coordinates will be different by order dx^2 (i think),
            # there is no way to compare the divB from simulation with the
            # one we get here. However, they should be the same up to a few %, and
            # down to noise level with stripes of enhanced noise. These stripes
            # are the errors in the coordinate values (since the output only
            # gives us weird nc = averaged cc locations)
            #
            # if abs_max_rel_diff > rtol or np.any(np.abs(max_crd_diff) > catol):
            #     raise RuntimeError("Tolerance exceeded on divB calculation")

    if test_streamline:
        b_cc = f['b_cc']['x=-40f:12f, y=-15f:15f, z=-15f:15f']
        b_fc = f['b_fc']['x=-40f:12f, y=-15f:15f, z=-15f:15f']

        cotr = viscid.cotr.Cotr()
        r_mask = 3.0
        # set b_cc to dipole inside some sphere
        isphere_mask = viscid.make_spherical_mask(b_cc, rmax=r_mask)
        moment = cotr.get_dipole_moment(crd_system=b_cc)
        viscid.fill_dipole(b_cc, m=moment, mask=isphere_mask)
        # set b_fc to dipole inside some sphere
        isphere_mask = viscid.make_spherical_mask(b_fc, rmax=r_mask)
        moment = cotr.get_dipole_moment(crd_system=b_fc)
        viscid.fill_dipole(b_fc, m=moment, mask=isphere_mask)

        seeds = viscid.Volume([-10, 0, -5], [10, 0, 5], (16, 1, 3))
        sl_kwargs = dict(ibound=1.0, method=viscid.EULER1A)
        lines_cc, topo_cc = viscid.calc_streamlines(b_cc, seeds, **sl_kwargs)
        lines_fc, topo_fc = viscid.calc_streamlines(b_fc, seeds, **sl_kwargs)

        if make_plots:
            plt.figure(figsize=(10, 6))

            ax0 = plt.subplot(211)
            topo_cc_colors = viscid.topology2color(topo_cc)
            vlt.plot(f['pp']['y=0f'], logscale=True, earth=True, cmap='plasma')
            vlt.plot2d_lines(lines_cc, topo_cc_colors, symdir='y')

            ax0 = plt.subplot(212, sharex=ax0, sharey=ax0)
            topo_fc_colors = viscid.topology2color(topo_fc)
            vlt.plot(f['pp']['y=0f'], logscale=True, earth=True, cmap='plasma')
            vlt.plot2d_lines(lines_fc, topo_fc_colors, symdir='y')

            plt.xlim(-20, 10)
            plt.ylim(-10, 10)
            vlt.auto_adjust_subplots()
            vlt.show()

    if test_interp:
        # test interpolation with E . B / B
        b_cc = f['b_cc']
        b_fc = f['b_fc']
        e_cc = f['e_cc']
        e_ec = f['e_ec']

        cotr = viscid.cotr.Cotr()
        r_mask = 3.0
        # set b_cc to dipole inside some sphere
        isphere_mask = viscid.make_spherical_mask(b_cc, rmax=r_mask)
        moment = cotr.get_dipole_moment(crd_system=b_cc)
        viscid.fill_dipole(b_cc, m=moment, mask=isphere_mask)
        # set b_fc to dipole inside some sphere
        isphere_mask = viscid.make_spherical_mask(b_fc, rmax=r_mask)
        moment = cotr.get_dipole_moment(crd_system=b_fc)
        viscid.fill_dipole(b_fc, m=moment, mask=isphere_mask)
        # zero out e_cc inside some sphere
        viscid.set_in_region(e_cc, e_cc, alpha=0.0, beta=0.0, out=e_cc,
                             mask=viscid.make_spherical_mask(e_cc, rmax=r_mask))
        # zero out e_ec inside some sphere
        viscid.set_in_region(e_ec, e_ec, alpha=0.0, beta=0.0, out=e_ec,
                             mask=viscid.make_spherical_mask(e_ec, rmax=r_mask))

        tmp = viscid.empty([np.linspace(-10, 10, 64), np.linspace(-10, 10, 64),
                            np.linspace(-10, 10, 64)], center="Cell")

        b_cc_interp = viscid.interp_linear(b_cc, tmp)
        b_fc_interp = viscid.interp_linear(b_fc, tmp)
        e_cc_interp = viscid.interp_linear(e_cc, tmp)
        e_ec_interp = viscid.interp_linear(e_ec, tmp)

        epar_cc = viscid.dot(e_cc_interp, b_cc_interp) / viscid.magnitude(b_cc_interp)
        epar_ecfc = viscid.dot(e_ec_interp, b_fc_interp) / viscid.magnitude(b_fc_interp)

        if make_plots:
            # plt.figure()
            # ax0 = plt.subplot(121)
            # vlt.plot(b_cc['x']['y=0f'], clim=(-40, 40))
            # plt.subplot(122, sharex=ax0, sharey=ax0)
            # vlt.plot(b_fc['x']['y=0f'], clim=(-40, 40))
            # vlt.show()

            plt.figure(figsize=(14, 5))
            ax0 = plt.subplot(131)
            vlt.plot(epar_cc['y=0f'], symmetric=True, cbarlabel="Epar CC")
            plt.subplot(132, sharex=ax0, sharey=ax0)
            vlt.plot(epar_ecfc['y=0f'], symmetric=True, cbarlabel="Epar ECFC")
            plt.subplot(133, sharex=ax0, sharey=ax0)
            vlt.plot(((epar_cc - epar_ecfc) / epar_cc)['y=0f'], clim=(-10, 10),
                     cbarlabel="Rel Diff")
            vlt.auto_adjust_subplots()
            vlt.show()

    return 0

if __name__ == "__main__":
    sys.exit(main())

##
## EOF
##
