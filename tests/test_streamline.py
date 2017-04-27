#!/usr/bin/env python
"""Test streamline calculation and plots"""

from __future__ import print_function
import argparse
from itertools import count, cycle
import sys

import numpy as np

from viscid_test_common import next_plot_fname

import viscid


_global_ns = dict()


def run_test(_fld, _seeds, plot2d=True, plot3d=True, title='', show=False,
             **kwargs):
    lines, topo = viscid.calc_streamlines(_fld, _seeds, **kwargs)
    topo_color = viscid.topology2color(topo)

    # downsample lines for plotting
    lines = [line[:, ::8] for line in lines]

    try:
        if not plot2d:
            raise ImportError
        from matplotlib import pyplot as plt
        from viscid.plot import vpyplot as vlt
        plt.clf()

        vlt.plot2d_lines(lines, scalars=topo_color, symdir='y', marker='^')
        if title:
            plt.title(title)

        plt.savefig(next_plot_fname(__file__, series='2d'))
        if show:
            plt.show()
    except ImportError:
        pass

    try:
        if not plot3d:
            raise ImportError
        from viscid.plot import vlab

        try:
            fig = _global_ns['figure']
            vlab.clf()
        except KeyError:
            fig = vlab.figure(size=[1200, 800], offscreen=not show,
                             bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
            _global_ns['figure'] = fig

        fld_mag = np.log(viscid.magnitude(_fld))
        try:
            # note: mayavi.mlab.mesh can't take color tuples as scalars
            #       so one can't use topo_color on a mesh surface. This
            #       is a limitation of mayavi. To actually plot a specific
            #       set of colors on a mesh, one must use a texture
            mesh = vlab.mesh_from_seeds(_seeds, scalars=topo, opacity=0.6)
            mesh.actor.property.backface_culling = True
        except RuntimeError:
            pass
        vlab.plot_lines(lines, scalars=fld_mag, tube_radius=0.01,
                       cmap='viridis')
        if title:
            vlab.title(title)

        vlab.savefig(next_plot_fname(__file__, series='3d'))
        if show:
            vlab.show()

    except ImportError:
        pass

def lines_and_lsps(B, seeds, cotr=None, **kwargs):
    """Return a list of streamlines and the l-shell,phi for each point"""
    tstats = dict()
    lines, _ = viscid.timeit(viscid.calc_streamlines, B, seeds,
                             timeit_quiet=True, timeit_stats=tstats, **kwargs)
    walltime = tstats['max']
    # lsrlps = [viscid.xyz2lsrlp(line, cotr=cotr, crd_system=B) for line in lines]
    lsrlps = [viscid.xyz2lsrlp(line, cotr=cotr, crd_system=B) for line in lines]
    lsps = [np.array(lsrlp[(0, 3), :]) for lsrlp in lsrlps]
    return lines, lsps, walltime

def format_data_range(dset):
    # x0 = np.percentile(dset, 0)
    x25 = np.percentile(dset, 25)
    x50 = np.percentile(dset, 50)
    x75 = np.percentile(dset, 75)
    # x100 = np.percentile(dset, 100)

    # return ("{0:.2e} <-- {1:.2e} -< {2:.2e} >- {3:.2e} --> {4:.2e}"
    #         "".format(x0, x25, x50, x75, x100))
    # return "{0:.2e} <- {1:.2e} -> {2:.2e}".format(x25, x50, x75)
    return (r"{0:.2e} $\leftarrow$ {1:.2e} $\rightarrow$ {2:.2e}"
            r"".format(x25, x50, x75))

def set_violin_colors(v):
    from viscid.plot import vpyplot as vlt

    cycler = cycle(vlt.get_current_colorcycle())
    colors = []
    for b in v['bodies']:
        c = next(cycler)
        b.set_color(c)
        colors.append(c)
    return colors


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notwo", dest='notwo', action="store_true")
    parser.add_argument("--nothree", dest='nothree', action="store_true")
    parser.add_argument("--show", "--plot", action="store_true")
    args = viscid.vutil.common_argparse(parser, default_verb=0)

    plot2d = not args.notwo
    plot3d = not args.nothree

    # #################################################
    # viscid.logger.info("Testing field lines on 2d field...")
    B = viscid.make_dipole(twod=True)
    line = viscid.seed.Line((0.2, 0.0, 0.0), (1.0, 0.0, 0.0), 10)
    obound0 = np.array([-4, -4, -4], dtype=B.data.dtype)
    obound1 = np.array([4, 4, 4], dtype=B.data.dtype)
    run_test(B, line, plot2d=plot2d, plot3d=plot3d, title='2D', show=args.show,
             ibound=0.07, obound0=obound0, obound1=obound1)

    #################################################
    viscid.logger.info("Testing field lines on 3d field...")
    B = viscid.make_dipole(m=[0.2, 0.3, -0.9])
    sphere = viscid.seed.Sphere((0.0, 0.0, 0.0), 2.0, ntheta=20, nphi=10)
    obound0 = np.array([-4, -4, -4], dtype=B.data.dtype)
    obound1 = np.array([4, 4, 4], dtype=B.data.dtype)
    run_test(B, sphere, plot2d=plot2d, plot3d=plot3d, title='3D', show=args.show,
             ibound=0.07, obound0=obound0, obound1=obound1, method=viscid.RK12)

    # The Remainder of this test makes sure higher order methods are indeed
    # more accurate than lower order methods... this could find a bug in
    # the integrators

    ##################################################
    # test accuracy of streamlines in an ideal dipole
    cotr = viscid.Cotr(dip_tilt=15.0, dip_gsm=21.0)  # pylint: disable=not-callable
    m = cotr.get_dipole_moment(crd_system='gse')
    seeds = viscid.seed.Sphere((0.0, 0.0, 0.0), 2.0, pole=-m, ntheta=25, nphi=25,
                               thetalim=(5, 90), philim=(5, 360), phi_endpoint=False)
    B = viscid.make_dipole(m=m, crd_system='gse', n=(256, 256, 256),
                           l=(-25, -25, -25), h=(25, 25, 25), dtype='f8')

    seeds_xyz = seeds.get_points()
    # seeds_lsp = viscid.xyz2lsrlp(seeds_xyz, cotr=cotr, crd_system=B)[(0, 3), :]
    seeds_lsp = viscid.xyz2lsrlp(seeds_xyz, cotr=cotr, crd_system=B)[(0, 3), :]

    e1_lines, e1_lsps, t_e1 = lines_and_lsps(B, seeds, method='euler1',
                                             ibound=1.0, cotr=cotr)
    rk2_lines, rk2_lsps, t_rk2 = lines_and_lsps(B, seeds, method='rk2',
                                                ibound=1.0, cotr=cotr)
    rk4_lines, rk4_lsps, t_rk4 = lines_and_lsps(B, seeds, method='rk4',
                                                ibound=1.0, cotr=cotr)
    e1a_lines, e1a_lsps, t_e1a = lines_and_lsps(B, seeds, method='euler1a',
                                                ibound=1.0, cotr=cotr)
    rk12_lines, rk12_lsps, t_rk12 = lines_and_lsps(B, seeds, method='rk12',
                                                   ibound=1.0, cotr=cotr)
    rk45_lines, rk45_lsps, t_rk45 = lines_and_lsps(B, seeds, method='rk45',
                                                   ibound=1.0, cotr=cotr)

    def _calc_rel_diff(_lsp, _ideal_lsp, _d):
        _diffs = []
        for _ilsp, _iideal in zip(_lsp, _ideal_lsp.T):
            _a = _ilsp[_d, :]
            _b = _iideal[_d]
            _diffs.append((_a - _b) / _b)
        return _diffs

    lshell_diff_e1 = _calc_rel_diff(e1_lsps, seeds_lsp, 0)
    phi_diff_e1 = _calc_rel_diff(e1_lsps, seeds_lsp, 1)

    lshell_diff_rk2 = _calc_rel_diff(rk2_lsps, seeds_lsp, 0)
    phi_diff_rk2 = _calc_rel_diff(rk2_lsps, seeds_lsp, 1)

    lshell_diff_rk4 = _calc_rel_diff(rk4_lsps, seeds_lsp, 0)
    phi_diff_rk4 = _calc_rel_diff(rk4_lsps, seeds_lsp, 1)

    lshell_diff_e1a = _calc_rel_diff(e1a_lsps, seeds_lsp, 0)
    phi_diff_e1a = _calc_rel_diff(e1a_lsps, seeds_lsp, 1)

    lshell_diff_rk12 = _calc_rel_diff(rk12_lsps, seeds_lsp, 0)
    phi_diff_rk12 = _calc_rel_diff(rk12_lsps, seeds_lsp, 1)

    lshell_diff_rk45 = _calc_rel_diff(rk45_lsps, seeds_lsp, 0)
    phi_diff_rk45 = _calc_rel_diff(rk45_lsps, seeds_lsp, 1)

    methods = ['Euler 1', 'Runge Kutta 2', 'Runge Kutta 4',
               'Euler 1 Adaptive Step', 'Runge Kutta 12 Adaptive Step',
               'Runge Kutta 45 Adaptive Step']
    wall_ts = [t_e1, t_rk2, t_rk4, t_e1a, t_rk12, t_rk45]
    all_lines = [e1_lines, rk2_lines, rk4_lines, e1a_lines, rk12_lines,
                 rk45_lines]
    all_lshell_diffs = [lshell_diff_e1, lshell_diff_rk2, lshell_diff_rk4,
                        lshell_diff_e1a, lshell_diff_rk12, lshell_diff_rk45]
    lshell_diffs = [np.abs(np.concatenate(lshell_diff_e1, axis=0)),
                    np.abs(np.concatenate(lshell_diff_rk2, axis=0)),
                    np.abs(np.concatenate(lshell_diff_rk4, axis=0)),
                    np.abs(np.concatenate(lshell_diff_e1a, axis=0)),
                    np.abs(np.concatenate(lshell_diff_rk12, axis=0)),
                    np.abs(np.concatenate(lshell_diff_rk45, axis=0))]
    phi_diffs = [np.abs(np.concatenate(phi_diff_e1, axis=0)),
                 np.abs(np.concatenate(phi_diff_rk2, axis=0)),
                 np.abs(np.concatenate(phi_diff_rk4, axis=0)),
                 np.abs(np.concatenate(phi_diff_e1a, axis=0)),
                 np.abs(np.concatenate(phi_diff_rk12, axis=0)),
                 np.abs(np.concatenate(phi_diff_rk45, axis=0))]
    npts = [len(lsd) for lsd in lshell_diffs]
    lshell_75 = [np.percentile(lsdiff, 75) for lsdiff in lshell_diffs]

    # # 3D DEBUG PLOT:: for really getting under the covers
    # vlab.clf()
    # earth1 = viscid.seed.Sphere((0.0, 0.0, 0.0), 1.0, pole=-m, ntheta=60, nphi=120,
    #                             thetalim=(15, 165), philim=(0, 360))
    # ls1 = viscid.xyz2lsrlp(earth1.get_points(), cotr=cotr, crd_system='gse')[0, :]
    # earth2 = viscid.seed.Sphere((0.0, 0.0, 0.0), 2.0, pole=-m, ntheta=60, nphi=120,
    #                             thetalim=(15, 165), philim=(0, 360))
    # ls2 = viscid.xyz2lsrlp(earth2.get_points(), cotr=cotr, crd_system='gse')[0, :]
    # earth4 = viscid.seed.Sphere((0.0, 0.0, 0.0), 4.0, pole=-m, ntheta=60, nphi=120,
    #                             thetalim=(15, 165), philim=(0, 360))
    # ls4 = viscid.xyz2lsrlp(earth4.get_points(), cotr=cotr, crd_system='gse')[0, :]
    # clim = [2.0, 6.0]
    # vlab.mesh_from_seeds(earth1, scalars=ls1, clim=clim, logscale=True)
    # vlab.mesh_from_seeds(earth2, scalars=ls2, clim=clim, logscale=True, opacity=0.5)
    # vlab.mesh_from_seeds(earth4, scalars=ls2, clim=clim, logscale=True, opacity=0.25)
    # vlab.plot3d_lines(e1_lines, scalars=[_e1_lsp[0, :] for _e1_lsp in e1_lsps],
    #                  clim=clim, logscale=True)
    # vlab.colorbar(title="L-Shell")
    # vlab.show()

    assert lshell_75[1] < lshell_75[0], "RK2 should have less error than Euler"
    assert lshell_75[2] < lshell_75[1], "RK4 should have less error than RK2"
    assert lshell_75[3] < lshell_75[0], "Euler 1a should have less error than Euler 1"
    assert lshell_75[4] < lshell_75[0], "RK 12 should have less error than Euler 1"
    assert lshell_75[5] < lshell_75[1], "RK 45 should have less error than RK2"

    try:
        if not plot2d:
            raise ImportError
        from matplotlib import pyplot as plt
        from viscid.plot import vpyplot as vlt

        # stats on error for all points on all lines
        _ = plt.figure(figsize=(15, 8))
        ax1 = vlt.subplot(121)
        v = plt.violinplot(lshell_diffs, showextrema=False, showmedians=False,
                               vert=False)
        colors = set_violin_colors(v)
        xl, xh = plt.gca().get_xlim()
        for i, txt, c in zip(count(), methods, colors):
            t_txt = ", took {0:.2e} seconds".format(wall_ts[i])
            stat_txt = format_data_range(lshell_diffs[i])
            plt.text(xl + 0.35 * (xh - xl), i + 1.15, txt + t_txt, color=c)
            plt.text(xl + 0.35 * (xh - xl), i + 0.85, stat_txt, color=c)
        ax1.get_yaxis().set_visible(False)
        plt.title('L-Shell')
        plt.xlabel('Relative Difference from Ideal (as fraction)')

        ax2 = vlt.subplot(122)
        v = plt.violinplot(phi_diffs, showextrema=False, showmedians=False,
                               vert=False)
        colors = set_violin_colors(v)
        xl, xh = plt.gca().get_xlim()
        for i, txt, c in zip(count(), methods, colors):
            t_txt = ", took {0:.2e} seconds".format(wall_ts[i])
            stat_txt = format_data_range(phi_diffs[i])
            plt.text(xl + 0.35 * (xh - xl), i + 1.15, txt + t_txt, color=c)
            plt.text(xl + 0.35 * (xh - xl), i + 0.85, stat_txt, color=c)
        ax2.get_yaxis().set_visible(False)
        plt.title('Longitude')
        plt.xlabel('Relative Difference from Ideal (as fraction)')

        vlt.auto_adjust_subplots()

        vlt.savefig(next_plot_fname(__file__, series='q2'))
        if args.show:
            vlt.show()

        # stats for ds for all points on all lines
        _ = plt.figure(figsize=(10, 8))
        ax1 = vlt.subplot(111)

        ds = [np.concatenate([np.linalg.norm(_l[:, 1:] - _l[:, :-1], axis=0)
                              for _l in lines]) for lines in all_lines]
        v = plt.violinplot(ds, showextrema=False, showmedians=False,
                               vert=False)
        colors = set_violin_colors(v)
        xl, xh = plt.gca().get_xlim()
        for i, txt, c in zip(count(), methods, colors):
            stat_txt = format_data_range(ds[i])
            plt.text(xl + 0.01 * (xh - xl), i + 1.15, txt, color=c)
            plt.text(xl + 0.01 * (xh - xl), i + 0.85, stat_txt, color=c)
        ax1.get_yaxis().set_visible(False)
        plt.xscale('log')
        plt.title('Step Size')
        plt.xlabel('Absolute Step Size')
        vlt.savefig(next_plot_fname(__file__, series='q2'))
        if args.show:
            vlt.show()


        # random other information
        _ = plt.figure(figsize=(13, 10))

        ## wall time for each method
        vlt.subplot(221)
        plt.scatter(range(len(methods)), wall_ts, color=colors,
                        s=150, marker='s', edgecolors='none')
        for i, meth in enumerate(methods):
            meth = meth.replace(" Adaptive Step", "\nAdaptive Step")
            plt.annotate(meth, (i, wall_ts[i]), xytext=(0, 15.0),
                             color=colors[i], horizontalalignment='center',
                             verticalalignment='bottom',
                             textcoords='offset points')
        plt.ylabel("Wall Time (s)")
        x_padding = 0.5
        plt.xlim(-x_padding, len(methods) - x_padding)
        yl, yh = np.min(wall_ts), np.max(wall_ts)
        y_padding = 0.4 * (yh - yl)
        plt.ylim(yl - y_padding, yh + y_padding)
        plt.gca().get_xaxis().set_visible(False)
        for _which in ('right', 'top'):
            plt.gca().spines[_which].set_color('none')

        ## number of points calculated for each method
        vlt.subplot(222)
        plt.scatter(range(len(methods)), npts, color=colors,
                        s=150, marker='s', edgecolors='none')
        for i, meth in enumerate(methods):
            meth = meth.replace(" Adaptive Step", "\nAdaptive Step")
            plt.annotate(meth, (i, npts[i]), xytext=(0, 15.0),
                             color=colors[i], horizontalalignment='center',
                             verticalalignment='bottom',
                             textcoords='offset points')
        plt.ylabel("Number of Streamline Points Calculated")
        x_padding = 0.5
        plt.xlim(-x_padding, len(methods) - x_padding)
        yl, yh = np.min(npts), np.max(npts)
        y_padding = 0.4 * (yh - yl)
        plt.ylim(yl - y_padding, yh + y_padding)
        plt.gca().get_xaxis().set_visible(False)
        for _which in ('right', 'top'):
            plt.gca().spines[_which].set_color('none')

        ## Wall time per segment, this should show the overhead of the method
        vlt.subplot(223)
        wall_t_per_seg = np.asarray(wall_ts) / np.asarray(npts)
        plt.scatter(range(len(methods)), wall_t_per_seg, color=colors,
                        s=150, marker='s', edgecolors='none')
        for i, meth in enumerate(methods):
            meth = meth.replace(" Adaptive Step", "\nAdaptive Step")
            plt.annotate(meth, (i, wall_t_per_seg[i]), xytext=(0, 15.0),
                             color=colors[i], horizontalalignment='center',
                             verticalalignment='bottom',
                             textcoords='offset points')
        plt.ylabel("Wall Time Per Line Segment")
        x_padding = 0.5
        plt.xlim(-x_padding, len(methods) - x_padding)
        yl, yh = np.min(wall_t_per_seg), np.max(wall_t_per_seg)
        y_padding = 0.4 * (yh - yl)
        plt.ylim(yl - y_padding, yh + y_padding)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().xaxis.set_major_formatter(viscid.plot.mpl_extra.steve_axfmt)
        for _which in ('right', 'top'):
            plt.gca().spines[_which].set_color('none')

        ## 75th percentile of l-shell error for each method
        vlt.subplot(224)
        plt.scatter(range(len(methods)), lshell_75, color=colors,
                        s=150, marker='s', edgecolors='none')
        plt.yscale('log')

        for i, meth in enumerate(methods):
            meth = meth.replace(" Adaptive Step", "\nAdaptive Step")
            plt.annotate(meth, (i, lshell_75[i]), xytext=(0, 15.0),
                             color=colors[i], horizontalalignment='center',
                             verticalalignment='bottom',
                             textcoords='offset points')
        plt.ylabel("75th Percentile of Relative L-Shell Error")
        x_padding = 0.5
        plt.xlim(-x_padding, len(methods) - x_padding)
        ymin, ymax = np.min(lshell_75), np.max(lshell_75)
        plt.ylim(0.75 * ymin, 2.5 * ymax)
        plt.gca().get_xaxis().set_visible(False)
        for _which in ('right', 'top'):
            plt.gca().spines[_which].set_color('none')

        vlt.auto_adjust_subplots(subplot_params=dict(wspace=0.25, hspace=0.15))

        vlt.savefig(next_plot_fname(__file__, series='q2'))
        if args.show:
            vlt.show()

    except ImportError:
        pass

    try:
        if not plot3d:
            raise ImportError
        from viscid.plot import vlab

        try:
            fig = _global_ns['figure']
            vlab.clf()
        except KeyError:
            fig = vlab.figure(size=[1200, 800], offscreen=not args.show,
                              bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
            _global_ns['figure'] = fig

        for i, method in zip(count(), methods):
            # if i in (3, 4):
            #     next_plot_fname(__file__, series='q3')
            #     print(i, "::", [line.shape[1] for line in all_lines[i]])
            #     # continue
            vlab.clf()
            _lshell_diff = [np.abs(s) for s in all_lshell_diffs[i]]
            vlab.plot3d_lines(all_lines[i], scalars=_lshell_diff)
            vlab.colorbar(title="Relative L-Shell Error (as fraction)")
            vlab.title(method, size=0.5)
            vlab.orientation_axes()
            vlab.view(azimuth=40, elevation=140, distance=80.0,
                      focalpoint=[0, 0, 0])
            vlab.savefig(next_plot_fname(__file__, series='q3'))
            if args.show:
                vlab.show()
    except ImportError:
        pass


    # prevent weird xorg bad-instructions on tear down
    if 'figure' in _global_ns and _global_ns['figure'] is not None:
        from viscid.plot import vlab
        vlab.mlab.close(_global_ns['figure'])

    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
