#!/usr/bin/env python

from __future__ import print_function
from timeit import default_timer as time
import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
import cProfile
import pstats

from viscid import vutil
from viscid import field
from viscid import coordinate
from viscid import readers
from viscid.calculator import calc
from viscid.calculator import cycalc
from viscid.calculator import streamline
from viscid.calculator import seed
from viscid.plot import vpyplot as vlt

def make_dipole(m=None, twod=False):
    dtype = 'float64'
    n = 64
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

if __name__ == "__main__":
    B = make_dipole(m=[0.2, 0.3, -0.9])
    t0 = time()
    # sphere = seed.Sphere((0.0, 0.0, 0.0), 2.0, 500, 500)
    # cProfile.runctx("""interp_vals = cycalc.interp_trilin(B, sphere)""",
    #                 globals(), locals(), "interp.prof")
    # plane = seed.Plane((1., 1., 1.), (1., 1., 1.), (1., 0., 0.),
    #                    1.0, 1.0, 500, 500)
    vol = B.crds
    # print(plane.get_points())
    cProfile.runctx("""interp_vals = cycalc.interp_trilin(B, vol)""",
                    globals(), locals(), "interp.prof")
    t1 = time()
    print("Total time: {0:.3e}".format(t1 - t0))
    s = pstats.Stats("interp.prof")
    s.strip_dirs().sort_stats("cumtime").print_stats()
    # print([line.shape for line in lines])
    # vlt.scatter_3d(vol.get_points(), interp_vals[:, 2], show=True)

    interp_field = field.wrap_field(interp_vals, vol.as_coordinates(),  # pylint: disable=undefined-variable
                                    name="interp", fldtype="vector",
                                    center="cell")
    interp_y1 = interp_field["y=1"]
    exact_y1 = B["y=1"]
    bxi, byi, bzi = interp_y1.component_fields()
    bxe, bye, bze = exact_y1.component_fields()
    for interp, exact in zip([bxi, byi, bzi], [bxe, bye, bze]):
        plt.clf()
        plt.subplot(211)
        vlt.plot(exact, show=False)
        plt.subplot(212)
        vlt.plot(interp, show=True)

##
## EOF
##
