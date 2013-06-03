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
from viscid.plot import mpl

def get_dipole(m=None, twod=False):
    dtype = 'float64'
    n = 256
    x = np.array(np.linspace(-5, 5, n), dtype=dtype)
    y = np.array(np.linspace(-5, 5, n), dtype=dtype)
    z = np.array(np.linspace(-5, 5, n), dtype=dtype)
    if twod:
        y = np.array(np.linspace(-0.1, 0.1, 2), dtype=dtype)
    crds = coordinate.wrap_crds("Rectilinear", (('z', z), ('y', y), ('x', x)))

    one = np.array([1.0], dtype=dtype) #pylint: disable=W0612
    three = np.array([3.0], dtype=dtype) #pylint: disable=W0612
    if not m:
        m = [0.0, 0.0, -1.0]
    m = np.array(m, dtype=dtype)
    mx, my, mz = m #pylint: disable=W0612

    Zcc, Ycc, Xcc = crds.get_crd(shaped=True, center="Cell") #pylint: disable=W0612

    rsq = ne.evaluate("Xcc**2 + Ycc**2 + Zcc**2") #pylint: disable=W0612
    mdotr = ne.evaluate("mx * Xcc + my * Ycc + mz * Zcc") #pylint: disable=W0612
    Bx = ne.evaluate("((three * Xcc * mdotr / rsq) - mx) / rsq**1.5")
    By = ne.evaluate("((three * Ycc * mdotr / rsq) - my) / rsq**1.5")
    Bz = ne.evaluate("((three * Zcc * mdotr / rsq) - mz) / rsq**1.5")

    fld = field.VectorField("B_cc", crds, [Bx, By, Bz],
                            center="Cell", forget_source=True,
                            info={"force_layout": field.LAYOUT_INTERLACED},
                           )
    # fld_rsq = field.ScalarField("r", crds, hmm,
    #                             center="Cell", forget_source=True)
    return fld  # , fld_rsq

if __name__=="__main__":
    B = get_dipole(m=[0.2, 0.3, -0.9])
    t0 = time()
    # sphere = seed.Sphere((0.0, 0.0, 0.0), 2.0, 500, 500)
    # cProfile.runctx("""interp_vals = cycalc.trilin_interp(B, sphere)""",
    #                 globals(), locals(), "interp.prof")    
    plane = seed.Plane((1., 1., 1.), (1., 1., 1.), (1., 0., 0.), 
                       1.0, 1.0, 500, 500)
    # print(plane.points())
    cProfile.runctx("""interp_vals = cycalc.trilin_interp(B, plane)""",
                    globals(), locals(), "interp.prof")
    t1 = time()
    print("Total time: {0:.3e}".format(t1 - t0))
    s = pstats.Stats("interp.prof")
    s.strip_dirs().sort_stats("cumtime").print_stats()
    # print([line.shape for line in lines])
    mpl.scatter_3d(plane.points(), interp_vals[:, 2], show=True)

    interp_field = field.wrap_field("Vector", "interp", plane.as_coordinates(),
                                    interp_vals)
    bx, by, bz = interp_field.component_fields()
    mpl.plot(bx, show=True)
    mpl.plot(by, show=True)
    mpl.plot(bz, show=True)

##
## EOF
##
