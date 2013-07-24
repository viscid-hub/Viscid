#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
import cProfile
import pstats
from timeit import default_timer as time

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
    cProfile.runctx("""lines, topo = streamline.streamlines(B,
                               seed.Sphere((0.0, 0.0, 0.0),
                                           2.0, 20, 10),
                               ds0=0.01, ibound=0.05, maxit=10000,
                               method=streamline.EULER1,
                               tol_lo=1e-3, tol_hi=1e-2)""",
                    globals(), locals(), "stream.prof")
    t1 = time()
    print("timeit: {0:.4} seconds".format(t1 - t0))
    s = pstats.Stats("stream.prof")
    s.strip_dirs().sort_stats("cumtime").print_stats()
    # print([line.shape for line in lines])
    mpl.plot_streamlines(lines, show=True)

##
## EOF
##
