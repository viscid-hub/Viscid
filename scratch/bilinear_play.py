#!/usr/bin/env python

from __future__ import print_function
from itertools import count

import numpy as np
from matplotlib import pyplot as plt
from mayavi import mlab

from viscid import field
from viscid import coordinate
from viscid.plot import mpl
from viscid.plot import mvi
from viscid.calculator import cycalc

np.seterr(divide='ignore')

def find_roots_face(a1, b1, c1, d1, a2, b2, c2, d2):
    """
    f1(x, y) = a1 + b1 * x + c1 * y + d1 * x * y
    f2(x, y) = a1 + b1 * x + c1 * y + d1 * x * y
    where {x, y} are in the range [0.0, 1.0]

    returns ndarray(x), ndarray(y) such that
    f1(xi, yi) = f2(xi, yi) = 0 and
    0 <= xi <= 1.0 && 0 <= yi <= 1.0
    """
    # F(x) = a * x**2 + b * x + c = 0 has the same roots xi
    # as f1(x, ?) = f2(x, ?) = 0, so solve that, then
    # substitute back into f1(xi, yi) = 0 to find yi
    a = b1 * d2 - d1 * b2
    b = (a1 * d2 - d1 * a2) + (b1 * c2 - c1 * b2)
    c = a1 * c2 - c1 * a2

    if a == 0.0:
        if b != 0.0:
            roots_x = np.array([-c / b])
            # print("::", roots_x)
        else:
            roots_x = np.array([])
            # print("::", roots_x)
    else:
        desc = b**2 - (4.0 * a * c)
        if desc > 0.0:
            # il y a deux solutions
            rootx1 = (-b + np.sqrt(desc)) / (2.0 * a)
            rootx2 = (-b - np.sqrt(desc)) / (2.0 * a)
            roots_x = np.array([rootx1, rootx2])
            # print("::", roots_x)
        elif desc == 0.0:
            # il y a seulment une solution
            roots_x = np.array([-b / (2.0 * a)])
            # print("::", roots_x)
        else:
            roots_x = np.array([])
            # print("::", roots_x)

    roots_y = - (a1 + b1 * roots_x) / (c1 + d1 * roots_x)

    # remove roots that are outside the box
    keep = ((roots_x >= 0.0) & (roots_x <= 1.0) &
            (roots_y >= 0.0) & (roots_y <= 1.0))
    roots_x = roots_x[keep]
    roots_y = roots_y[keep]

    return roots_x, roots_y

def main():
    xl, xh, nx = -1.0, 1.0, 41
    yl, yh, ny = -1.5, 1.5, 41
    zl, zh, nz = -2.0, 2.0, 41
    x = np.linspace(xl, xh, nx)
    y = np.linspace(yl, yh, ny)
    z = np.linspace(zl, zh, nz)
    crds = coordinate.wrap_crds("nonuniform_cartesian",
                                [('z', z), ('y', y), ('x', x)])
    fld1 = field.empty("Scalar", "f1", crds, center="Node")
    fld2 = field.empty("Scalar", "f2", crds, center="Node")
    fld3 = field.empty("Scalar", "f3", crds, center="Node")
    fld = field.empty("Vector", "f", crds, 3, center="Node",
                      layout="interlaced")
    Z, Y, X = crds.get_crds(shaped=True)

    x01, y01, z01 = 0.5, 0.5, 0.5
    x02, y02, z02 = 0.5, 0.5, 0.5
    x03, y03, z03 = 0.5, 0.5, 0.5

    fld1[:] = 0.0 + 1.0 * (X - x01) + 1.0 * (Y - y01) + 1.0 * (Z - z01) + \
              1.0 * (X - x01) * (Y - y01) + 1.0 * (Y - y01) * (Z - z01) + \
              1.0 * (X - x01) * (Y - y01) * (Z - z01)
    fld2[:] = 0.0 + 1.0 * (X - x02) + 1.0 * (Y - y02) + 1.0 * (Z - z02) + \
              1.0 * (X - x02) * (Y - y02) + 1.0 * (Y - y02) * (Z - z02) - \
              1.0 * (X - x02) * (Y - y02) * (Z - z02)
    fld3[:] = 0.0 + 1.0 * (X - x03) + 1.0 * (Y - y03) + 1.0 * (Z - z03) + \
              1.0 * (X - x03) * (Y - y03) - 1.0 * (Y - y03) * (Z - z03) + \
              1.0 * (X - x03) * (Y - y03) * (Z - z03)
    fld[..., 0] = fld1
    fld[..., 1] = fld2
    fld[..., 2] = fld3

    f1_src = mvi.field_to_source(fld1)
    f2_src = mvi.field_to_source(fld2)
    f3_src = mvi.field_to_source(fld3)
    e = mlab.get_engine()
    e.add_source(f1_src)
    e.add_source(f2_src)
    e.add_source(f3_src)
    mlab.pipeline.iso_surface(f1_src, contours=[0.0],
                              opacity=1.0, color=(1.0, 0.0, 0.0))
    mlab.pipeline.iso_surface(f2_src, contours=[0.0],
                              opacity=1.0, color=(0.0, 1.0, 0.0))
    mlab.pipeline.iso_surface(f3_src, contours=[0.0],
                              opacity=1.0, color=(0.0, 0.0, 1.0))
    mlab.show()

    nullpt = cycalc.interp_trilin(fld, [(0.5, 0.5, 0.5)])
    print("f(0.5, 0.5, 0.5):", nullpt)

    ax1 = plt.subplot2grid((4, 3), (0, 0))
    positive_roots = []
    for di, d in enumerate([0, -1]):
        #### XY face
        a1 = fld1[d, 0, 0]
        b1 = fld1[d, 0, -1] - fld1[d, 0, 0]
        c1 = fld1[d, -1, 0] - fld1[d, 0, 0]
        d1 = fld1[d, -1, -1] - fld1[d, 0, -1] - fld1[d, -1, 0] + fld1[d, 0, 0]

        a2 = fld2[d, 0, 0]
        b2 = fld2[d, 0, -1] - fld2[d, 0, 0]
        c2 = fld2[d, -1, 0] - fld2[d, 0, 0]
        d2 = fld2[d, -1, -1] - fld2[d, 0, -1] - fld2[d, -1, 0] + fld2[d, 0, 0]

        a3 = fld3[d, 0, 0]
        b3 = fld3[d, 0, -1] - fld3[d, 0, 0]
        c3 = fld3[d, -1, 0] - fld3[d, 0, 0]
        d3 = fld3[d, -1, -1] - fld3[d, 0, -1] - fld3[d, -1, 0] + fld3[d, 0, 0]

        roots_x, roots_y = find_roots_face(a1, b1, c1, d1, a2, b2, c2, d2)

        for xrt, yrt in zip(roots_x, roots_y):
            print("=")
            print("fx", a1 + b1 * xrt + c1 * yrt + d1 * xrt * yrt)
            print("fy", a2 + b2 * xrt + c2 * yrt + d2 * xrt * yrt)
            print("=")

        # find f3 at the root points
        f3 = np.empty_like(roots_x)
        markers = [None] * len(f3)
        for i, rtx, rty in zip(count(), roots_x, roots_y):
            f3[i] = a3 + b3 * rtx + c3 * rty + d3 * rtx * rty
            if f3[i] >= 0.0:
                markers[i] = 'k^'
                positive_roots.append((rtx, rty, d))
            else:
                markers[i] = 'g^'

        # rescale the roots to the original domain
        roots_x = (xh - xl) * roots_x + xl
        roots_y = (yh - yl) * roots_y + yl

        xp = np.linspace(0.0, 1.0, nx)

        # plt.subplot(121)
        plt.subplot2grid((4, 3), (0 + 2 * di, 0), sharex=ax1, sharey=ax1)
        mpl.plot(fld['x'], "z={0}i".format(d),
                 plot_opts="x={0}_{1},y={2}_{3},lin_-10_10".format(xl, xh, yl, yh))
        y1 = - (a1 + b1 * xp) / (c1 + d1 * xp)
        plt.plot(x, (yh - yl) * y1 + yl, 'k')
        for i, xrt, yrt in zip(count(), roots_x, roots_y):
            plt.plot(xrt, yrt, markers[i])

        # plt.subplot(122)
        plt.subplot2grid((4, 3), (1 + 2 * di, 0), sharex=ax1, sharey=ax1)
        mpl.plot(fld['y'], "z={0}i".format(d),
                 plot_opts="x={0}_{1},y={2}_{3},lin_-10_10".format(xl, xh, yl, yh))
        y2 = - (a2 + b2 * xp) / (c2 + d2 * xp)
        plt.plot(x, (yh - yl) * y2 + yl, 'k')
        for xrt, yrt in zip(roots_x, roots_y):
            plt.plot(xrt, yrt, 'k^')

        #### XZ face
        a1 = fld1[0, d, 0]
        b1 = fld1[0, d, -1] - fld1[0, d, 0]
        c1 = fld1[-1, d, 0] - fld1[0, d, 0]
        d1 = fld1[-1, d, -1] - fld1[0, d, -1] - fld1[-1, d, 0] + fld1[0, d, 0]

        a2 = fld2[0, d, 0]
        b2 = fld2[0, d, -1] - fld2[0, d, 0]
        c2 = fld2[-1, d, 0] - fld2[0, d, 0]
        d2 = fld2[-1, d, -1] - fld2[0, d, -1] - fld2[-1, d, 0] + fld2[0, d, 0]

        a3 = fld3[0, d, 0]
        b3 = fld3[0, d, -1] - fld3[0, d, 0]
        c3 = fld3[-1, d, 0] - fld3[0, d, 0]
        d3 = fld3[-1, d, -1] - fld3[0, d, -1] - fld3[-1, d, 0] + fld3[0, d, 0]

        # roots_x, roots_z = find_roots_face(a1, b1, c1, d1, a3, b3, c3, d3)
        roots_x, roots_z = find_roots_face(a1, b1, c1, d1, a2, b2, c2, d2)

        for xrt, zrt in zip(roots_x, roots_z):
            print("=")
            print("fx", a1 + b1 * xrt + c1 * zrt + d1 * xrt * zrt)
            print("fy", a2 + b2 * xrt + c2 * zrt + d2 * xrt * zrt)
            print("=")

        # find f3 at the root points
        f3 = np.empty_like(roots_x)
        markers = [None] * len(f3)
        for i, rtx, rtz in zip(count(), roots_x, roots_z):
            f3[i] = a3 + b3 * rtx + c3 * rtz + d3 * rtx * rtz
            if f3[i] >= 0.0:
                markers[i] = 'k^'
                positive_roots.append((rtx, rtz, d))
            else:
                markers[i] = 'g^'

        # rescale the roots to the original domain
        roots_x = (xh - xl) * roots_x + xl
        roots_z = (zh - zl) * roots_z + zl

        xp = np.linspace(0.0, 1.0, nx)

        # plt.subplot(121)
        plt.subplot2grid((4, 3), (0 + 2 * di, 1), sharex=ax1, sharey=ax1)
        mpl.plot(fld['x'], "y={0}i".format(d),
                 plot_opts="x={0}_{1},y={2}_{3},lin_-10_10".format(xl, xh, zl, zh))
        z1 = - (a1 + b1 * xp) / (c1 + d1 * xp)
        plt.plot(x, (zh - zl) * z1 + zl, 'k')
        for i, xrt, zrt in zip(count(), roots_x, roots_z):
            plt.plot(xrt, zrt, markers[i])

        # plt.subplot(122)
        plt.subplot2grid((4, 3), (1 + 2 * di, 1), sharex=ax1, sharey=ax1)
        mpl.plot(fld['y'], "y={0}i".format(d),
                 plot_opts="x={0}_{1},y={2}_{3},lin_-10_10".format(xl, xh, zl, zh))
        z2 = - (a2 + b2 * xp) / (c2 + d2 * xp)
        plt.plot(x, (zh - zl) * z2 + zl, 'k')
        for i, xrt, zrt in zip(count(), roots_x, roots_z):
            plt.plot(xrt, zrt, markers[i])

        #### YZ face
        a1 = fld1[0, 0, d]
        b1 = fld1[0, -1, d] - fld1[0, 0, d]
        c1 = fld1[-1, 0, d] - fld1[0, 0, d]
        d1 = fld1[-1, -1, d] - fld1[0, -1, d] - fld1[-1, 0, d] + fld1[0, 0, d]

        a2 = fld2[0, 0, d]
        b2 = fld2[0, -1, d] - fld2[0, 0, d]
        c2 = fld2[-1, 0, d] - fld2[0, 0, d]
        d2 = fld2[-1, -1, d] - fld2[0, -1, d] - fld2[-1, 0, d] + fld2[0, 0, d]

        a3 = fld3[0, 0, d]
        b3 = fld3[0, -1, d] - fld3[0, 0, d]
        c3 = fld3[-1, 0, d] - fld3[0, 0, d]
        d3 = fld3[-1, -1, d] - fld3[0, -1, d] - fld3[-1, 0, d] + fld3[0, 0, d]

        # roots_y, roots_z = find_roots_face(a2, b2, c2, d2, a3, b3, c3, d3)
        roots_y, roots_z = find_roots_face(a1, b1, c1, d1, a2, b2, c2, d2)

        for yrt, zrt in zip(roots_y, roots_z):
            print("=")
            print("fx", a1 + b1 * yrt + c1 * zrt + d1 * yrt * zrt)
            print("fy", a2 + b2 * yrt + c2 * zrt + d2 * yrt * zrt)
            print("=")

        # find f1 at the root points
        f3 = np.empty_like(roots_y)
        markers = [None] * len(f3)
        for i, rty, rtz in zip(count(), roots_y, roots_z):
            f3[i] = a3 + b3 * rty + c3 * rtz + d3 * rty * rtz
            if f3[i] >= 0.0:
                markers[i] = 'k^'
                positive_roots.append((rty, rtz, d))
            else:
                markers[i] = 'g^'

        # rescale the roots to the original domain
        roots_y = (yh - yl) * roots_y + yl
        roots_z = (zh - zl) * roots_z + zl

        yp = np.linspace(0.0, 1.0, ny)

        # plt.subplot(121)
        plt.subplot2grid((4, 3), (0 + 2 * di, 2), sharex=ax1, sharey=ax1)
        mpl.plot(fld['x'], "x={0}i".format(d),
                 plot_opts="x={0}_{1},y={2}_{3},lin_-10_10".format(yl, yh, zl, zh))
        z1 = - (a1 + b1 * yp) / (c1 + d1 * yp)
        plt.plot(y, (zh - zl) * z1 + zl, 'k')
        for i, yrt, zrt in zip(count(), roots_y, roots_z):
            plt.plot(yrt, zrt, markers[i])

        # plt.subplot(122)
        plt.subplot2grid((4, 3), (1 + 2 * di, 2), sharex=ax1, sharey=ax1)
        mpl.plot(fld['y'], "x={0}i".format(d),
                 plot_opts="x={0}_{1},y={2}_{3},lin_-10_10".format(yl, yh, zl, zh))
        z2 = - (a2 + b2 * yp) / (c2 + d2 * yp)
        plt.plot(y, (zh - zl) * z2 + zl, 'k')
        for i, yrt, zrt in zip(count(), roots_y, roots_z):
            plt.plot(yrt, zrt, markers[i])

    print("Null Point?", len(positive_roots) % 2 == 1)

    plt.show()


main()

##
## EOF
##
