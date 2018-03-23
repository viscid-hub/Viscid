#!/usr/bin/env python
"""test matplotlib + mayavi prepare_lines scalar processing

this test is not run by default since it makes a lot of plots
"""

from __future__ import print_function
import argparse
import sys
import textwrap

import numpy as np

from viscid_test_common import next_plot_fname

import viscid


_global_ns = dict()


def do_test(lines, scalars, show=False, txt=""):
    viscid.logger.info('--> ' + txt)
    title = txt + '\n' + "\n".join(textwrap.wrap("scalars = {0}".format(scalars),
                                                 width=50))

    try:
        from viscid.plot import vpyplot as vlt
        from matplotlib import pyplot as plt

        vlt.clf()
        vlt.plot_lines(lines, scalars=scalars)
        plt.title(title)
        vlt.savefig(next_plot_fname(__file__, series='q2'))
        if show:
            vlt.show()
    except ImportError:
        pass

    try:
        from mayavi import mlab
        from viscid.plot import vlab

        try:
            fig = _global_ns['figure']
            vlab.clf()
        except KeyError:
            fig = vlab.figure(size=[1200, 800], offscreen=not show,
                              bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
            _global_ns['figure'] = fig

        vlab.clf()
        vlab.plot_lines3d(lines, scalars=scalars)
        vlab.fancy_axes()
        mlab.text(0.05, 0.05, title)
        vlab.savefig(next_plot_fname(__file__, series='q3'))
        if show:
            vlab.show(stop=True)
    except ImportError:
        pass

def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", "--plot", action="store_true")
    args = viscid.vutil.common_argparse(parser, default_verb=0)

    viscid.logger.setLevel(viscid.logging.DEBUG)
    args.show = False

    cotr = viscid.Cotr(dip_tilt=20.0, dip_gsm=15.0)  # pylint: disable=not-callable
    b = viscid.make_dipole(m=cotr.get_dipole_moment(), n=(32, 32, 32))

    seeds = viscid.Circle(n=5, r=1.5, pole=[0, 0, 1])
    lines, topo = viscid.calc_streamlines(b, seeds, ibound=1.4, method='rk45')

    for i in range(2):
        # make sure this works for lines with 0, 1, 2, 3 vertices
        if i == 1:
            lines[1] = lines[2][:, :0]
            lines[2] = lines[2][:, :1]
            lines[3] = lines[3][:, :2]
            lines[4] = lines[4][:, :3]

        viscid.logger.debug('---')
        viscid.logger.debug('{0}'.format(len(lines)))
        for line in lines:
            viscid.logger.debug('line shape: {0}'.format(line.shape))
        viscid.logger.debug('---')

        do_test(lines,
                scalars=None,
                txt='given None',
                show=args.show)

        do_test(lines,
                scalars='#ff0000',
                txt='given a single 24bit rgb hex color',
                show=args.show)

        do_test(lines,
                scalars='#ff000066',
                txt='given a single 32bit rgba hex color',
                show=args.show)

        do_test(lines,
                scalars='#f00',
                txt='given a single 12bit rgb hex color',
                show=args.show)

        do_test(lines,
                scalars='#f006',
                txt='given a single 16bit rgba hex color',
                show=args.show)

        do_test(lines,
                scalars=['#ff0000', '#cc0000', '#aa0000', '#880000', '#660000'],
                txt='given a list of Nlines 24bit rgb hex colors',
                show=args.show)

        do_test(lines,
                scalars=['#ff000066', '#cc000066', '#aa000066', '#88000066',
                         '#66000066'],
                txt='given a list of Nlines 32bit rgba hex colors',
                show=args.show)

        do_test(lines,
                scalars=['#f00', '#c00', '#a00', '#800', '#600'],
                txt='given a list of Nlines 12bit rgb hex colors',
                show=args.show)

        do_test(lines,
                scalars=['#f00a', '#c009', '#a008', '#8007', '#6006'],
                txt='given a list of Nlines 16bit rgba hex colors',
                show=args.show)

        do_test(lines,
                scalars=[0.8, 0.0, 0.2],
                txt='given a single rgb [0..1] color',
                show=args.show)

        do_test(lines,
                scalars=[0.8, 0.0, 0.2, 0.8],
                txt='given a single rgba [0..1] color',
                show=args.show)

        do_test(lines,
                scalars=[(0.8, 0.0, 0.2), (0.7, 0.0, 0.3), (0.6, 0.0, 0.4),
                         (0.5, 0.0, 0.5), (0.4, 0.0, 0.6)],
                txt='given a list of Nlines rgb [0..1] tuples',
                show=args.show)

        do_test(lines,
                scalars=[(0.8, 0.0, 0.2, 1.0), (0.7, 0.0, 0.3, 0.9),
                         (0.6, 0.0, 0.4, 0.8), (0.5, 0.0, 0.5, 0.7),
                         (0.4, 0.0, 0.6, 0.6)],
                txt='given a list of Nlines rgba [0..1] tuples',
                show=args.show)

        do_test(lines,
                scalars=[250, 0, 250],
                txt='given a single rgb [0..255] color',
                show=args.show)

        do_test(lines,
                scalars=[250, 0, 250, 190],
                txt='given a single rgba [0..255] color',
                show=args.show)

        do_test(lines,
                scalars=[(204, 0, 51), (179, 0, 77), (153, 0, 102),
                         (127, 0, 127), (0.4, 0, 102)],
                txt='given a list of Nlines rgb [0..255] tuples',
                show=args.show)

        do_test(lines,
                scalars=[(204, 0, 51, 255), (179, 0, 77, 230),
                         (153, 0, 102, 204), (127, 0, 127, 179),
                         (102, 0, 102, 153)],
                txt='given a list of Nlines rgba [0..255] tuples',
                show=args.show)

        do_test(lines,
                scalars=['#ff000088', 'blue', 'lavenderblush', 'c', '#4f4'],
                txt='given a mix of color hex/html color names',
                show=args.show)

        do_test(lines,
                scalars=topo,
                txt='scalars == topo value',
                show=args.show)

        do_test(lines,
                scalars=viscid.topology2color(topo),
                txt='scalars == topo2color value',
                show=args.show)

        do_test(lines,
                scalars=np.log(viscid.magnitude(b)),
                txt='given bmag',
                show=args.show)

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
