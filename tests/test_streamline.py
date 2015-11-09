#!/usr/bin/env python
""" test making and plotting streamlines """

from __future__ import print_function
import argparse

import numpy as np

from viscid_test_common import next_plot_fname

import viscid


def run_test(_fld, _seeds, plot2d=True, plot3d=True, title='', show=False,
             **kwargs):
    lines, topo = viscid.calc_streamlines(_fld, _seeds, **kwargs)
    topo_fld = _seeds.wrap_field(topo)
    topo_color = viscid.topology2color(topo)

    # downsample lines for plotting
    lines = [line[:, ::8] for line in lines]

    try:
        if not plot2d:
            raise ImportError
        from viscid.plot import mpl
        mpl.plt.clf()

        mpl.plot2d_lines(lines, scalars=topo_color, symdir='y', marker='^')
        if title:
            mpl.plt.title(title)

        mpl.plt.savefig(next_plot_fname(__file__, series='2d'))
        if show:
            mpl.plt.show()
    except ImportError:
        pass

    try:
        if not plot3d:
            raise ImportError
        from viscid.plot import mvi
        mvi.clf()

        fld_mag = np.log(viscid.magnitude(_fld))
        try:
            # note: mayavi.mlab.mesh can't take color tuples as scalars
            #       so one can't use topo_color on a mesh surface. This
            #       is a limitation of mayavi. To actually plot a specific
            #       set of colors on a mesh, one must use a texture
            vertices, scalars = _seeds.wrap_mesh(topo_fld.data)
            mesh = mvi.mlab.mesh(vertices[0], vertices[1], vertices[2],
                                 scalars=scalars, opacity=0.5)
            mesh.actor.property.backface_culling = True
        except RuntimeError:
            pass
        mvi.plot_lines(lines, scalars=fld_mag, tube_radius=0.01)
        if title:
            mvi.mlab.title(title)

        mvi.mlab.savefig(next_plot_fname(__file__, series='3d'))
        if show:
            mvi.show()

    except ImportError:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notwo", dest='notwo', action="store_true")
    parser.add_argument("--nothree", dest='nothree', action="store_true")
    parser.add_argument("--show", "--plot", action="store_true")
    args = viscid.vutil.common_argparse(parser, default_verb=0)

    plot2d = not args.notwo
    plot3d = not args.nothree

    viscid.logger.info("Testing field lines on 2d field...")
    B = viscid.vlab.get_dipole(twod=True)
    line = viscid.seed.Line((0.2, 0.0, 0.0), (1.0, 0.0, 0.0), 10)
    obound0 = np.array([-4, -4, -4], dtype=B.data.dtype)
    obound1 = np.array([4, 4, 4], dtype=B.data.dtype)
    run_test(B, line, plot2d=plot2d, plot3d=plot3d, title='2D', show=args.show,
             ibound=0.07, obound0=obound0, obound1=obound1)

    viscid.logger.info("Testing field lines on 3d field...")
    B = viscid.vlab.get_dipole(m=[0.2, 0.3, -0.9])
    sphere = viscid.seed.Sphere((0.0, 0.0, 0.0), 2.0, ntheta=20, nphi=10)
    obound0 = np.array([-4, -4, -4], dtype=B.data.dtype)
    obound1 = np.array([4, 4, 4], dtype=B.data.dtype)
    run_test(B, sphere, plot2d=plot2d, plot3d=plot3d, title='3D', show=args.show,
             ibound=0.07, obound0=obound0, obound1=obound1, method=viscid.RK12)

if __name__ == "__main__":
    main()

##
## EOF
##
