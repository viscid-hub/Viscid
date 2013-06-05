#!/usr/bin/env python
""" Helpers to generate seed points for fieldlines """

from __future__ import print_function
import itertools

import numpy as np

from .. import coordinate

class SeedGen(object):
    _cache = None
    _points = None
    dtype = None
    params = {}

    def __init__(self, cache=False, dtype=None):
        self._cache = cache
        self.dtype = dtype
        self.params = {}

    def points(self):
        """ 3 x n ndarray of n seed points """
        if self._points is not None:
            return self._points
        pts = self.gen_points()
        if self._cache:
            self._points = pts
        return pts

    def iter_points(self, **kwargs):
        """ always returns an iterator, this can be overridden in a subclass
        if it's more efficient to iterate than a call to gen_points,
        calling iter_points should make an effort to be the fastest way to
        iterate through the points, ignorant of caching. In cython, it seems to
        take twice the time to iterate over a generator, than to use a for loop
        over indices of an array, but only if the array already exists """
        pts = self.points()
        return itertools.izip(pts[0, :], pts[1, :], pts[2, :])

    def n_points(self, **kwargs):
        raise NotImplementedError()

    def as_coordinates(self):
        """ basically, wrap_crds on something meaningful """
        raise NotImplementedError()

    def gen_points(self):
        """ this should return an iterable of (z, y, x) points """
        raise NotImplementedError()

class Point(SeedGen):
    def __init__(self, points, cache=False, dtype=None):
        """ points should be an n x 3 array of zyx points """
        super(Point, self).__init__(cache=cache, dtype=dtype)
        self.params["points"] = points

    def n_points(self, **kwargs):
        return self.points().shape[0]
        
    def gen_points(self):
        pts_arr = np.array(self.params["points"], dtype=self.dtype).T
        pts_arr = pts_arr.reshape((3, -1))
        return pts_arr


class Line(SeedGen):
    def __init__(self, p0, p1, res=20, cache=False, dtype=None):
        """ p0 & p1 are (z, y, x) points as list, tuple, or ndarray
        res is the number of need points to generate """
        super(Line, self).__init__(cache=cache, dtype=dtype)
        self.params["p0"] = p0
        self.params["p1"] = p1
        self.params["res"] = res

    def n_points(self, **kwargs):
        return self.params["res"]

    def gen_points(self):
        p0 = self.params["p0"]
        p1 = self.params["p1"]
        res = self.params["res"]
        z = np.linspace(p0[0], p1[0], res).astype(self.dtype)
        y = np.linspace(p0[1], p1[1], res).astype(self.dtype)
        x = np.linspace(p0[2], p1[2], res).astype(self.dtype)
        return np.array([z, y, x])

    def as_coordinates(self):
        p0 = self.params["p0"]
        p1 = self.params["p1"]
        res = self.params["res"]

        dp = p1 - p0
        dist = np.sqrt(np.dot(dp, dp))
        x = np.linspace(0.0, dist, res)
        crd = coordinate.wrap_crds("Rectilinear", (('x', x),))
        return crd


class Plane(SeedGen):
    def __init__(self, p0, Ndir, Ldir, len_l, len_m, res_l=20, res_m=20,
                 cache=False):
        """ plane is specified in L,M,N coords where N is normal to the plane,
        and L is projected into the plane. len_l and len_m is the extent
        of the plane in the l and m directions. res_l and res_m are the
        resolution of points in the two directions. Note that p0 is the center
        of the plane, so the plane extends from (-len_l/2, len_l/2) around
        p0, and similarly in the m direction. """
        super(Plane, self).__init__(cache=cache)
        self.params["p0"] = p0
        self.params["Ndir"] = Ndir
        self.params["Ldir"] = Ldir
        self.params["len_l"] = len_l
        self.params["len_m"] = len_m
        self.params["res_l"] = res_l
        self.params["res_m"] = res_m

    def n_points(self, **kwargs):
        return self.params["res_l"] * self.params["res_m"]

    # def _make_arrays(self):
    #     pass

    def gen_points(self):
        # turn N and L into flat ndarrays
        Ndir = np.array(self.params["Ndir"], dtype=self.dtype).reshape(-1)
        Ldir = np.array(self.params["Ldir"], dtype=self.dtype).reshape(-1)
        len_l = self.params["len_l"]
        len_m = self.params["len_m"]
        res_l = self.params["res_l"]
        res_m = self.params["res_m"]        
        p0 = np.array(self.params["p0"]).reshape((3, 1))

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

        arr = np.empty((3, res_l * res_m), dtype=self.dtype)
        l = np.linspace(-0.5 * len_l, 0.5 * len_l,
                        res_l).astype(self.dtype)
        m = np.linspace(-0.5 * len_m, 0.5 * len_m,
                        res_m).astype(self.dtype)
        arr[0, :] = np.repeat(m, res_l)
        arr[1, :] = np.tile(l, res_m)
        arr[2, :] = 0.0

        # create matrix to go LNM -> zyx
        trans = np.array([Mdir, Ldir, Ndir]).T

        arr_transformed = p0 + np.dot(trans, arr)

        return arr_transformed

    def as_coordinates(self):
        len_l = self.params["len_l"]
        len_m = self.params["len_m"]
        res_l = self.params["res_l"]
        res_m = self.params["res_m"]

        l = np.linspace(-0.5 * len_l, 0.5 * len_l, res_l)
        m = np.linspace(-0.5 * len_m, 0.5 * len_m, res_m)

        crds = coordinate.wrap_crds("Rectilinear", (('y', m), ('x', l)))
        return crds


class Volume(SeedGen):
    def __init__(self, p0, p1, res=(20, 20, 20), cache=False):
        """ p1 & p2 are (z, y, x) points as list, tuple, or ndarray
        res is the number of need points to generate (nz, ny, nx) """
        super(Volume, self).__init__(cache=cache)
        self.params["p0"] = p0
        self.params["p1"] = p1
        self.params["res"] = res

    def n_points(self, **kwargs):
        return self.params["res"]

    def gen_points(self):
        return np.array(list(self.iter_points()))

    def _make_arrays(self):
        p0 = self.params["p0"]
        p1 = self.params["p1"]
        res = self.params["res"]
        z = np.linspace(p0[0], p1[0], res[0]).astype(self.dtype)
        y = np.linspace(p0[1], p1[1], res[1]).astype(self.dtype)
        x = np.linspace(p0[2], p1[2], res[2]).astype(self.dtype)
        return z, y, x

    def iter_points(self):
        z, y, x = self._make_arrays()
        return itertools.product(z, y, x)

    def as_coordinates(self):
        z, y, x = self._make_arrays()
        crd = coordinate.wrap_crds("Rectilinear", 
                                   (('z', z), ('y', y), ('x', x)))
        return crd
        

class Sphere(SeedGen):
    def __init__(self, p0, r, restheta=20, resphi=20, cache=False):
        super(Sphere, self).__init__(cache=cache)
        self.params["p0"] = p0
        self.params["r"] = r
        self.params["restheta"] = restheta
        self.params["resphi"] = resphi

    def n_points(self, **kwargs):
        return self.params["restheta"] * self.params["resphi"]

    def gen_points(self):
        p0 = self.params["p0"]
        r = self.params["r"]
        restheta = self.params["restheta"]
        resphi = self.params["resphi"]

        theta = np.linspace(0, np.pi, restheta + 1, 
                            endpoint=True).astype(self.dtype)
        theta = 0.5 * (theta[1:] + theta[:-1])
        phi = np.linspace(0, 2.0 * np.pi, resphi,
                          endpoint=False).astype(self.dtype)
        T, P = np.ix_(theta, phi)

        x = p0[2] + r * np.sin(T) * np.cos(P)
        y = p0[1] + r * np.sin(T) * np.sin(P)
        z = p0[0] + r * np.cos(T) + 0.0 * P
        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)

        return np.array([z, y, x])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D #pylint: disable=W0611

    ax = plt.gca(projection='3d')

    l = Line((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)).points()
    ax.plot(l[:, 2], l[:, 1], l[:, 0], 'g.')

    s = Sphere((0.0, 0.0, 0.0), 2.0, 10, 20).points()
    ax.plot(s[:, 2], s[:, 1], s[:, 0], 'b')

    p = Plane((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0),
                     2.0, 3.0, 10, 20).points()
    ax.plot(p[:, 2], p[:, 1], p[:, 0], 'r')

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.show()

##
## EOF
##
