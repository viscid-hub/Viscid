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
    """ This function just finds roots of a quadratic equation
    that solves the intersection of f1 = f2 = 0.
    f1(x, y) = a1 + b1 * x + c1 * y + d1 * x * y
    f2(x, y) = a1 + b1 * x + c1 * y + d1 * x * y
    where {x, y} are in the range [0.0, 1.0]

    Returns ndarray(x), ndarray(y) such that
    f1(xi, yi) = f2(xi, yi) = 0 and
    0 <= xi <= 1.0 && 0 < yi < 1.0

    Note: the way this prunes edges is different for x and y,
    this is to keep from double counting edges of a cell
    when the faces are visited xy yz zx
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
            (roots_y > 0.0) & (roots_y < 1.0))
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
    bx = field.empty("Scalar", "$B_x$", crds, center="Node")
    by = field.empty("Scalar", "$B_y$", crds, center="Node")
    bz = field.empty("Scalar", "$B_z$", crds, center="Node")
    fld = field.empty("Vector", "B", crds, 3, center="Node",
                      layout="interlaced")
    Z, Y, X = crds.get_crds(shaped=True)

    x01, y01, z01 = 0.5, 0.5, 0.5
    x02, y02, z02 = 0.5, 0.5, 0.5
    x03, y03, z03 = 0.5, 0.5, 0.5

    bx[:] = 0.0 + 1.0 * (X - x01) + 1.0 * (Y - y01) + 1.0 * (Z - z01) + \
              1.0 * (X - x01) * (Y - y01) + 1.0 * (Y - y01) * (Z - z01) + \
              1.0 * (X - x01) * (Y - y01) * (Z - z01)
    by[:] = 0.0 + 1.0 * (X - x02) - 1.0 * (Y - y02) + 1.0 * (Z - z02) + \
              1.0 * (X - x02) * (Y - y02) + 1.0 * (Y - y02) * (Z - z02) - \
              1.0 * (X - x02) * (Y - y02) * (Z - z02)
    bz[:] = 0.0 + 1.0 * (X - x03) + 1.0 * (Y - y03) - 1.0 * (Z - z03) + \
              1.0 * (X - x03) * (Y - y03) + 1.0 * (Y - y03) * (Z - z03) + \
              1.0 * (X - x03) * (Y - y03) * (Z - z03)
    fld[..., 0] = bx
    fld[..., 1] = by
    fld[..., 2] = bz

    fig = mlab.figure(size=(1150, 850),
                      bgcolor=(1.0, 1.0, 1.0),
                      fgcolor=(0.0, 0.0, 0.0))
    f1_src = mvi.field_to_source(bx)
    f2_src = mvi.field_to_source(by)
    f3_src = mvi.field_to_source(bz)
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
    mlab.axes()
    mlab.show()

    nullpt = cycalc.interp_trilin(fld, [(0.5, 0.5, 0.5)])
    print("f(0.5, 0.5, 0.5):", nullpt)

    ax1 = plt.subplot2grid((4, 3), (0, 0))
    all_roots = []
    positive_roots = []
    ix = iy = iz = 0

    for di, d in enumerate([0, -1]):
        #### XY face
        a1 = bx[iz + d, iy, ix]
        b1 = bx[iz + d, iy, ix - 1] - a1
        c1 = bx[iz + d, iy - 1, ix] - a1
        d1 = bx[iz + d, iy - 1, ix - 1] - c1 - b1 - a1

        a2 = by[iz + d, iy, ix]
        b2 = by[iz + d, iy, ix - 1] - a2
        c2 = by[iz + d, iy - 1, ix] - a2
        d2 = by[iz + d, iy - 1, ix - 1] - c2 - b2 - a2

        a3 = bz[iz + d, iy, ix]
        b3 = bz[iz + d, iy, ix - 1] - a3
        c3 = bz[iz + d, iy - 1, ix] - a3
        d3 = bz[iz + d, iy - 1, ix - 1] - c3 - b3 - a3

        roots1, roots2 = find_roots_face(a1, b1, c1, d1, a2, b2, c2, d2)

        # for rt1, rt2 in zip(roots1, roots2):
        #     print("=")
        #     print("fx", a1 + b1 * rt1 + c1 * rt2 + d1 * rt1 * rt2)
        #     print("fy", a2 + b2 * rt1 + c2 * rt2 + d2 * rt1 * rt2)
        #     print("=")

        # find f3 at the root points
        f3 = np.empty_like(roots1)
        markers = [None] * len(f3)
        for i, rt1, rt2 in zip(count(), roots1, roots2):
            f3[i] = a3 + b3 * rt1 + c3 * rt2 + d3 * rt1 * rt2
            all_roots.append((rt1, rt2, d))  # switch order here
            if f3[i] >= 0.0:
                markers[i] = 'k^'
                positive_roots.append((rt1, rt2, d))  # switch order here
            else:
                markers[i] = 'w^'

        # rescale the roots to the original domain
        roots1 = (xh - xl) * roots1 + xl
        roots2 = (yh - yl) * roots2 + yl

        xp = np.linspace(0.0, 1.0, nx)

        # plt.subplot(121)
        plt.subplot2grid((4, 3), (0 + 2 * di, 0), sharex=ax1, sharey=ax1)
        mpl.plot(fld['x'], "z={0}i".format(d),
                 plot_opts="x={0}_{1},y={2}_{3},lin_-10_10".format(xl, xh, yl, yh))
        y1 = - (a1 + b1 * xp) / (c1 + d1 * xp)
        plt.plot(x, (yh - yl) * y1 + yl, 'k')
        for i, xrt, yrt in zip(count(), roots1, roots2):
            plt.plot(xrt, yrt, markers[i])

        # plt.subplot(122)
        plt.subplot2grid((4, 3), (1 + 2 * di, 0), sharex=ax1, sharey=ax1)
        mpl.plot(fld['y'], "z={0}i".format(d),
                 plot_opts="x={0}_{1},y={2}_{3},lin_-10_10".format(xl, xh, yl, yh))
        y2 = - (a2 + b2 * xp) / (c2 + d2 * xp)
        plt.plot(x, (yh - yl) * y2 + yl, 'k')
        for xrt, yrt in zip(roots1, roots2):
            plt.plot(xrt, yrt, markers[i])

        #### YZ face
        a1 = bx[iz, iy, ix + d]
        b1 = bx[iz, iy - 1, ix + d] - a1
        c1 = bx[iz - 1, iy, ix + d] - a1
        d1 = bx[iz - 1, iy - 1, ix + d] - c1 - b1 - a1

        a2 = by[iz, iy, ix + d]
        b2 = by[iz, iy - 1, ix + d] - a2
        c2 = by[iz - 1, iy, ix + d] - a2
        d2 = by[iz - 1, iy - 1, ix + d] - c2 - b2 - a2

        a3 = bz[iz, iy, ix + d]
        b3 = bz[iz, iy - 1, ix + d] - a3
        c3 = bz[iz - 1, iy, ix + d] - a3
        d3 = bz[iz - 1, iy - 1, ix + d] - c3 - b3 - a3

        roots1, roots2 = find_roots_face(a1, b1, c1, d1, a2, b2, c2, d2)

        # for rt1, rt2 in zip(roots1, roots2):
        #     print("=")
        #     print("fx", a1 + b1 * rt1 + c1 * rt2 + d1 * rt1 * rt2)
        #     print("fy", a2 + b2 * rt1 + c2 * rt2 + d2 * rt1 * rt2)
        #     print("=")

        # find f3 at the root points
        f3 = np.empty_like(roots1)
        markers = [None] * len(f3)
        for i, rt1, rt2 in zip(count(), roots1, roots2):
            f3[i] = a3 + b3 * rt1 + c3 * rt2 + d3 * rt1 * rt2
            all_roots.append((d, rt1, rt2))  # switch order here
            if f3[i] >= 0.0:
                markers[i] = 'k^'
                positive_roots.append((d, rt1, rt2))  # switch order here
            else:
                markers[i] = 'w^'

        # rescale the roots to the original domain
        roots1 = (yh - yl) * roots1 + yl
        roots2 = (zh - zl) * roots2 + zl

        yp = np.linspace(0.0, 1.0, ny)

        # plt.subplot(121)
        plt.subplot2grid((4, 3), (0 + 2 * di, 1), sharex=ax1, sharey=ax1)
        mpl.plot(fld['x'], "x={0}i".format(d),
                 plot_opts="x={0}_{1},y={2}_{3},lin_-10_10".format(yl, yh, zl, zh))
        z1 = - (a1 + b1 * yp) / (c1 + d1 * yp)
        plt.plot(y, (zh - zl) * z1 + zl, 'k')
        for i, yrt, zrt in zip(count(), roots1, roots2):
            plt.plot(yrt, zrt, markers[i])

        # plt.subplot(122)
        plt.subplot2grid((4, 3), (1 + 2 * di, 1), sharex=ax1, sharey=ax1)
        mpl.plot(fld['y'], "x={0}i".format(d),
                 plot_opts="x={0}_{1},y={2}_{3},lin_-10_10".format(yl, yh, zl, zh))
        z1 = - (a2 + b2 * yp) / (c2 + d2 * yp)
        plt.plot(y, (zh - zl) * z1 + zl, 'k')
        for i, yrt, zrt in zip(count(), roots1, roots2):
            plt.plot(yrt, zrt, markers[i])

        #### ZX face
        a1 = bx[iz, iy + d, ix]
        b1 = bx[iz - 1, iy + d, ix] - a1
        c1 = bx[iz, iy + d, ix - 1] - a1
        d1 = bx[iz - 1, iy + d, ix - 1] - c1 - b1 - a1

        a2 = by[iz, iy + d, ix]
        b2 = by[iz - 1, iy + d, ix] - a2
        c2 = by[iz, iy + d, ix - 1] - a2
        d2 = by[iz - 1, iy + d, ix - 1] - c2 - b2 - a2

        a3 = bz[iz, iy + d, ix]
        b3 = bz[iz - 1, iy + d, ix] - a3
        c3 = bz[iz, iy + d, ix - 1] - a3
        d3 = bz[iz - 1, iy + d, ix - 1] - c3 - b3 - a3

        roots1, roots2 = find_roots_face(a1, b1, c1, d1, a2, b2, c2, d2)

        # for rt1, rt2 in zip(roots1, roots2):
        #     print("=")
        #     print("fx", a1 + b1 * rt1 + c1 * rt2 + d1 * rt1 * rt2)
        #     print("fy", a2 + b2 * rt1 + c2 * rt2 + d2 * rt1 * rt2)
        #     print("=")

        # find f3 at the root points
        f3 = np.empty_like(roots1)
        markers = [None] * len(f3)
        for i, rt1, rt2 in zip(count(), roots1, roots2):
            f3[i] = a3 + b3 * rt1 + c3 * rt2 + d3 * rt1 * rt2
            all_roots.append((rt2, d, rt1))  # switch order here
            if f3[i] >= 0.0:
                markers[i] = 'k^'
                positive_roots.append((rt2, d, rt1))  # switch order here
            else:
                markers[i] = 'w^'

        # rescale the roots to the original domain
        roots1 = (zh - zl) * roots1 + zl
        roots2 = (xh - xl) * roots2 + xl

        zp = np.linspace(0.0, 1.0, nz)

        # plt.subplot(121)
        plt.subplot2grid((4, 3), (0 + 2 * di, 2), sharex=ax1, sharey=ax1)
        mpl.plot(fld['x'], "y={0}i".format(d),
                 plot_opts="x={0}_{1},y={2}_{3},lin_-10_10".format(xl, xh, zl, zh))
        x1 = - (a1 + b1 * zp) / (c1 + d1 * zp)
        plt.plot(z, (xh - xl) * x1 + xl, 'k')
        for i, zrt, xrt in zip(count(), roots1, roots2):
            plt.plot(xrt, zrt, markers[i])

        # plt.subplot(121)
        plt.subplot2grid((4, 3), (1 + 2 * di, 2), sharex=ax1, sharey=ax1)
        mpl.plot(fld['y'], "y={0}i".format(d),
                 plot_opts="x={0}_{1},y={2}_{3},lin_-10_10".format(xl, xh, zl, zh))
        x1 = - (a2 + b2 * zp) / (c2 + d2 * zp)
        plt.plot(z, (xh - xl) * x1 + xl, 'k')
        for i, zrt, xrt in zip(count(), roots1, roots2):
            plt.plot(xrt, zrt, markers[i])

    print("all:", len(all_roots), "positive:", len(positive_roots))
    if len(all_roots) % 2 == 1:
        print("something is fishy, there are an odd number of root points "
              "on the surface of your cube, there is probably a degenerate "
              "line or surface of nulls")
    print("Null Point?", (len(positive_roots) % 2 == 1))

    plt.show()


main()

##
## EOF
##
