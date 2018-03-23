#!/usr/bin/env python

from __future__ import division, print_function
import sys

import numpy as np
import viscid


def run_test(x, y, profile=False):
    z = np.array([0.0])
    fld_nc = viscid.empty([x, y, z], center='node', name="MyField",
                          pretty_name="My Field [Units]")

    fld_cc = fld_nc.as_centered('cell')
    x2, y2 = fld_cc.get_crds_nc('xy')

    dx = x2[:-1] - x
    dy = y2[:-1] - y
    assert np.all(dx < np.zeros_like(dx)), 'cells are jumbled in x'
    assert np.all(dy < np.zeros_like(dy)), 'cells are jumbled in y'

    dx = x - x2[1:]
    dy = y - y2[1:]
    assert np.all(dx < np.zeros_like(dx)), 'cells are jumbled in x'
    assert np.all(dy < np.zeros_like(dy)), 'cells are jumbled in y'

    dx = x2[1:] - x2[:-1]
    dy = y2[1:] - y2[:-1]
    assert np.all(dx > np.zeros_like(dx)), 'new x nc not monotonic'
    assert np.all(dy > np.zeros_like(dy)), 'new y nc not monotonic'

def _main():
    # test non uniform crds
    x = np.array([-3, -2, 0, 1, 4], dtype='f')
    y = np.array([-3, -2, 0, 4, 8], dtype='f')
    run_test(x, y)

    x = np.linspace(-1, 1, 1024)**3
    y = np.linspace(-1, 1, 1024)**3
    run_test(x, y)

    # test uniform crds
    x = np.linspace(-1, 1, 1024)
    y = np.linspace(-2, 2, 1024)
    run_test(x, y)

    # test uniform crds
    x = viscid.linspace_datetime64('2008-01-01T12:00:00', '2008-01-01T15:00:00',
                                   1024)
    y = np.linspace(-1, 1, 1024)
    run_test(x, y)

    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
