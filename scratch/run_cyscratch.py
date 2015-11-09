
from __future__ import print_function
from timeit import default_timer as time
import sys

import numpy as np
import numexpr as ne
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()})

import cyscratch
from viscid import field
from viscid import coordinate

def get_dipole(m=None, twod=False):
    dtype = 'float64'
    n = 256
    x = np.array(np.linspace(-5, 5, n), dtype=dtype)
    y = np.array(np.linspace(-5, 5, n), dtype=dtype)
    z = np.array(np.linspace(-5, 5, n), dtype=dtype)
    if twod:
        y = np.array(np.linspace(-0.1, 0.1, 2), dtype=dtype)
    crds = coordinate.wrap_crds("nonuniform_cartesian", (('z', z), ('y', y), ('x', x)))

    one = np.array([1.0], dtype=dtype)  # pylint: disable=W0612
    three = np.array([3.0], dtype=dtype)  # pylint: disable=W0612
    if not m:
        m = [0.0, 0.0, -1.0]
    m = np.array(m, dtype=dtype)
    mx, my, mz = m  # pylint: disable=W0612

    Zcc, Ycc, Xcc = crds.get_crds_cc(shaped=True)  # pylint: disable=W0612

    rsq = ne.evaluate("Xcc**2 + Ycc**2 + Zcc**2")  # pylint: disable=W0612
    mdotr = ne.evaluate("mx * Xcc + my * Ycc + mz * Zcc")  # pylint: disable=W0612
    Bx = ne.evaluate("((three * Xcc * mdotr / rsq) - mx) / rsq**1.5")
    By = ne.evaluate("((three * Ycc * mdotr / rsq) - my) / rsq**1.5")
    Bz = ne.evaluate("((three * Zcc * mdotr / rsq) - mz) / rsq**1.5")

    fld = field.VectorField("B_cc", crds, [Bx, By, Bz],
                            center="Cell", forget_source=True,
                            _force_layout=field.LAYOUT_INTERLACED,
                           )
    # fld_rsq = field.ScalarField("r", crds, hmm,
    #                             center="Cell", forget_source=True)
    return fld  # , fld_rsq

def main():
    B = get_dipole(twod=True)
    bmag = cyscratch.magnitude(B)

if __name__ == "__main__":
    main()

##
## EOF
##

