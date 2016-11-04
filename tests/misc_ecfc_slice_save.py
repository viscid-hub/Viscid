#!/usr/bin/env python

from __future__ import division, print_function
import os
import sys

import viscid
from viscid.plot import mpl

def main():
    f = viscid.load_file("~/dev/work/tmedium/*.3d.[-1].xdmf")
    grid = f.get_grid()

    gslc = "x=-26f:12.5f, y=-15f:15f, z=-15f:15f"
    # gslc = "x=-12.5f:26f, y=-15f:15f, z=-15f:15f"

    b_cc = f['b_cc'][gslc]
    b_cc.name = "b_cc"
    b_fc = f['b_fc'][gslc]
    b_fc.name = "b_fc"

    e_cc = f['e_cc'][gslc]
    e_cc.name = "e_cc"
    e_ec = f['e_ec'][gslc]
    e_ec.name = "e_ec"

    pp = f['pp'][gslc]
    pp.name = 'pp'

    pargs = dict(logscale=True, earth=True)

    # mpl.clf()
    # ax1 = mpl.subplot(211)
    # mpl.plot(f['pp']['y=0f'], **pargs)
    # # mpl.plot(viscid.magnitude(f['b_cc']['y=0f']), **pargs)
    # # mpl.show()
    # mpl.subplot(212, sharex=ax1, sharey=ax1)
    # mpl.plot(viscid.magnitude(viscid.fc2cc(f['b_fc'])['y=0f']), **pargs)
    # mpl.show()

    basename = './tmediumR.3d.{0:06d}'.format(int(grid.time))
    viscid.save_fields(basename + '.h5', [b_cc, b_fc, e_cc, e_ec, pp])

    f2 = viscid.load_file(basename + ".xdmf")

    pargs = dict(logscale=True, earth=True)

    mpl.clf()
    ax1 = mpl.subplot(211)
    mpl.plot(f2['pp']['y=0f'], style='contour', levels=5, colorbar=None,
             colors='k', **pargs)
    mpl.plot(viscid.magnitude(f2['b_cc']['y=0f']), **pargs)
    mpl.subplot(212, sharex=ax1, sharey=ax1)
    mpl.plot(viscid.magnitude(viscid.fc2cc(f2['b_fc'])['y=0f']), **pargs)
    mpl.show()

    os.remove(basename + '.h5')
    os.remove(basename + '.xdmf')

    return 0

if __name__ == "__main__":
    sys.exit(main())

##
## EOF
##
