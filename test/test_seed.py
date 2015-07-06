#!/usr/bin/env python
""" test using seeds to calc streamlines / interpolation """

from __future__ import print_function
import argparse

import viscid
from viscid.calculator import interp_trilin
from viscid.plot import mpl
from viscid.vlab import get_dipole

def run_test(_fld, _seeds, plot_2d=True, plot_3d=True, selection=None,
             show=False):
    arr = interp_trilin(_fld, _seeds)
    ifld = _seeds.wrap_field(arr)
    pts = _seeds.points()
    if plot_2d:
        mpl.plot(ifld, selection=selection, show=show)
    if plot_3d:
        try:
            from viscid.plot import mvi
            mvi.mlab.points3d(pts[2], pts[1], pts[0], arr, scale_mode='none')
            if show:
                mvi.mlab.show()
        except ImportError:
            mpl.scatter_3d(pts, arr)
            mpl.plt.gca().set_aspect('equal')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--three", action="store_true")
    parser.add_argument("--show", "--plot", action="store_true")
    args = viscid.vutil.common_argparse(parser)
    args = parser.parse_args()

    B = get_dipole(l=[-10] * 3, h=[10] * 3, n=[128] * 3, m=[0.0, 0.0, -1.0])
    bmag = viscid.calculator.calc.magnitude(B)

    ### PLANE
    p0 = [0.5, 0.6, 0.7][::-1]
    N = [1., 2., 3.][::-1]
    L = [-3., -2., -1.][::-1]
    plane = viscid.seed.Plane(p0, N, L, 2., 2., 15, 20)
    run_test(bmag, plane, True, args.three, show=args.show)

    ### Sphere
    p0 = [1.0, 1.0, 1.0][::-1]
    sphere = viscid.seed.Sphere(p0, 1.0, 15, 20)
    run_test(bmag, sphere, True, args.three, show=args.show)

    ### Sphere Cap
    p0 = [0.5, 0.6, 0.7][::-1]
    p1 = [1.7, 1.6, 1.5][::-1]
    sphere_cap = viscid.seed.SphericalCap(p0, p1, 120.0, 15, 20)
    run_test(bmag, sphere_cap, True, args.three, show=args.show)

    ### Circle
    p0 = [0.5, 0.6, 0.7][::-1]
    p1 = [0.7, 0.6, 0.5][::-1]
    sphere_cap = viscid.seed.Circle(p0, p1, 20, r=0.8)
    run_test(bmag, sphere_cap, True, args.three, show=args.show)

if __name__ == "__main__":
    main()

##
## EOF
##
