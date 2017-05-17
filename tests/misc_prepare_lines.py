#!/usr/bin/env python
"""test matplotlib + mayavi prepare_lines scalar processing

this test is not run by default since it makes a lot of plots
"""

from __future__ import print_function
import argparse
from itertools import count, cycle
import sys

import numpy as np

from viscid_test_common import next_plot_fname

import viscid


_global_ns = dict()


def do_test(lines, scalars, show=False):
    try:
        from matplotlib import pyplot as plt
        from viscid.plot import vpyplot as vlt

        vlt.clf()
        vlt.plot_lines(lines, scalars=scalars)
        vlt.savefig(next_plot_fname(__file__, series='q2'))
        if show:
            vlt.show()
    except ImportError:
        pass

    try:
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

    # make sure this works for lines with 0, 1, 2, 3 vertices
    # lines[1] = lines[2][:, :0]
    # lines[2] = lines[2][:, :1]
    # lines[3] = lines[3][:, :2]
    # lines[4] = lines[4][:, :3]

    viscid.logger.debug('---')
    viscid.logger.debug('{0}'.format(len(lines)))
    for line in lines:
        viscid.logger.debug('line shape: {0}'.format(line.shape))
    viscid.logger.debug('---')

    viscid.logger.info("--> given a single hex color")
    scalars0 = '#ff0000'
    do_test(lines, scalars0, show=args.show)

    viscid.logger.info("--> given a list of Nlines hex colors")
    scalars1 = ['#ff0000', '#cc0000', '#aa0000', '#880000', '#660000']
    do_test(lines, scalars1, show=args.show)

    viscid.logger.info("--> given a list of 5 rgb [0..1] tuples")
    scalars2 = [(0.8, 0.0, 0.2), (0.7, 0.0, 0.3), (0.6, 0.0, 0.4),
               (0.5, 0.0, 0.5), (0.4, 0.0, 0.6)]
    do_test(lines, scalars2, show=args.show)

    viscid.logger.info("--> given a list of 5 rgba [0..1] tuples")
    scalars3 = [(0.8, 0.0, 0.2, 1.0), (0.7, 0.0, 0.3, 0.9), (0.6, 0.0, 0.4, 0.8),
               (0.5, 0.0, 0.5, 0.7), (0.4, 0.0, 0.6, 0.6)]
    do_test(lines, scalars3, show=args.show)

    viscid.logger.info("--> given a single rgb [0..1] color")
    scalars4 = [0.8, 0.0, 0.2]
    do_test(lines, scalars4, show=args.show)

    viscid.logger.info("--> given a single rgba [0..1] color")
    scalars5 = [0.8, 0.0, 0.2, 0.8]
    do_test(lines, scalars5, show=args.show)

    viscid.logger.info("--> given a list of 5 rgb [0..255] tuples")
    scalars6 = [(204, 0, 51), (179, 0, 77), (153, 0, 102),
               (127, 0, 127), (0.4, 0, 102)]
    do_test(lines, scalars6, show=args.show)

    viscid.logger.info("--> given a list of 5 rgba [0..255] tuples")
    scalars7 = [(204, 0, 51, 255), (179, 0, 77, 230), (153, 0, 102, 204),
               (127, 0, 127, 179), (102, 0, 102, 153)]
    do_test(lines, scalars7, show=args.show)

    viscid.logger.info("--> given a single rgb [0..255] color")
    scalars8 = [250, 0, 250]
    do_test(lines, scalars8, show=args.show)

    viscid.logger.info("--> given a single rgba [0..255] color")
    scalars9 = [250, 0, 250, 190]
    do_test(lines, scalars9, show=args.show)

    viscid.logger.info('--> scalars == topo value')
    do_test(lines, topo, show=args.show)

    viscid.logger.info('--> scalars == topo2color value')
    do_test(lines, viscid.topology2color(topo), show=args.show)

    viscid.logger.info('--> given bmag')
    scalars = np.log(viscid.magnitude(b))
    do_test(lines, scalars, show=args.show)

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
