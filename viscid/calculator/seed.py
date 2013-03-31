#!/usr/bin/env python
""" Helpers to generate seed points for fieldlines """

from __future__ import print_function

import numpy as np


class SeedGen(object):
    _points = None

    def __init__(self):
        pass

    @property
    def points(self):
        """ n x 3 ndarray of n seed points """
        return self._points

class PointSeedGen(SeedGen):
    def __init__(self, points):
        """ points should be an n x 3 array of zyx points """
        super(PointSeedGen, self).__init__()
        self.setup(points)

    def setup(self, points):
        self.points = np.array(points)

class LineSeedGen(SeedGen):
    def __init__(self, p1, p2, res=20):
        """ p1 & p2 are (z, y, x) points as list, tuple, or ndarray
        res is the number of need points to generate """
        super(LineSeedGen, self).__init__()
        self.setup(p1, p2, res)

    def setup(self, p1, p2, res):
        z = np.linspace(p1[0], p2[0], res)
        y = np.linspace(p1[1], p2[1], res)
        x = np.linspace(p1[2], p2[2], res)
        self._points = np.array([z, y, x]).T

class PlaneSeedGen(SeedGen):
    def __init__(self, p0, Ndir, Ldir, len_l, len_m, res_l=20, res_m=20):
        """ plane is specified in L,M,N coords where N is normal to the plane,
        and L is projected into the plane. len_l and len_m is the extent
        of the plane in the l and m directions. res_l and res_m are the
        resolution of points in teh two directions """
        super(PlaneSeedGen, self).__init__()
        self.setup(p0, Ndir, Ldir, len_l, len_m, res_l, res_m)

    def setup(self, p0, Ndir, Ldir, len_l, len_m, res_l=20, res_m=20):
        # turn N and L into flat ndarrays
        Ndir = np.array(Ndir).reshape(-1)
        Ldir = np.array(Ldir).reshape(-1)

        if (Ndir == Ldir).all():
            raise RuntimeError("Ndir == Ldir, this does not define a plane")

        # normalize normal direction
        Ndir = Ndir / np.sqrt(np.dot(Ndir, Ndir))
        # project Ldir into the plane perpendicular to Ndir
        # (N already normalized)
        Ldir = Ldir - np.dot(Ldir, Ndir) * Ndir
        # normalize Ldir
        Ldir = Ldir / np.sqrt(np.dot(Ldir, Ldir))
        # compute Mdir
        Mdir = np.cross(Ldir, Ndir)  # zyx => left handed cross product

        # create matrix to go LNM -> zyx
        trans = np.matrix([Ldir, Mdir, Ndir]).T

        l = np.linspace(-0.5 * len_l, 0.5 * len_l, res_l)
        m = np.linspace(-0.5 * len_m, 0.5 * len_m, res_m)
        x, y, z = [], [], []

        for li in l:
            for mj in m:
                pt = (trans * np.array([[li, mj, 0.0]]).T)
                x.append(p0[2] + pt[2, 0])
                y.append(p0[1] + pt[1, 0])
                z.append(p0[0] + pt[0, 0])

        self._points = np.array([z, y, x]).T

class SphereSeedGen(SeedGen):
    def __init__(self, p0, r, restheta=20, resphi=20):
        super(SphereSeedGen, self).__init__()
        self.setup(p0, r, restheta, resphi)

    def setup(self, p0, r, restheta, resphi):
        theta = np.linspace(0, np.pi, restheta + 1, endpoint=True)
        theta = 0.5 * (theta[1:] + theta[:-1])
        phi = np.linspace(0, 2.0 * np.pi, resphi, endpoint=False)
        T, P = np.ix_(theta, phi)

        x = p0[2] + r * np.sin(T) * np.cos(P)
        y = p0[1] + r * np.sin(T) * np.sin(P)
        z = p0[0] + r * np.cos(T) + 0.0 * P
        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)

        self._points = np.array([z, y, x]).T

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D #pylint: disable=W0611

    ax = plt.gca(projection='3d')

    l = LineSeedGen((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)).points
    ax.plot(l[:, 2], l[:, 1], l[:, 0], 'g.')

    s = SphereSeedGen((0.0, 0.0, 0.0), 2.0, 10, 20).points
    ax.plot(s[:, 2], s[:, 1], s[:, 0], 'b')

    p = PlaneSeedGen((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0),
                     2.0, 3.0, 10, 20).points
    ax.plot(p[:, 2], p[:, 1], p[:, 0], 'r')

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.show()

##
## EOF
##
