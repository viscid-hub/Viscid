#!/usr/bin/env python
"""Helpers to generate seed points for streamlines"""

from __future__ import print_function
import itertools
try:
    from itertools import izip
except ImportError:
    izip = zip

import numpy as np

from viscid import field
from viscid import coordinate

class SeedGen(object):
    """All about seeds

    These objects are good for defining the root points of streamlines
    for :meth:`vicid.calculator.streamline.streamlines`, or
    interpolation points a la
    :meth:`vicid.calculator.cycalc.interp_trilin`.
    """
    _cache = None
    _points = None
    dtype = None
    params = {}

    def __init__(self, cache=False, dtype=None):
        self._cache = cache
        self.dtype = dtype
        self.params = {}

    def points(self):
        """Get points an ndarray

        Returns:
            3 x n ndarray of n seed point
        """
        if self._points is not None:
            return self._points
        pts = self.genr_points()
        if self._cache:
            self._points = pts
        return pts

    def iter_points(self, **kwargs):
        """always returns an iterator, this can be overridden in a
        subclass if it's more efficient to iterate than a call to
        :meth:`genr_points`, calling iter_points should make an effort
        to be the fastest way to iterate through the points, ignorant
        of caching. In cython, it seems to take twice the time to
        iterate over a generator, than to use a for loop over indices
        of an array, but only if the array already exists
        """
        pts = self.points()
        return izip(pts[0, :], pts[1, :], pts[2, :])

    def nr_points(self, **kwargs):
        raise NotImplementedError()

    def as_coordinates(self):
        """basically, wrap_crds on something meaningful"""
        raise NotImplementedError()

    def genr_points(self):
        """this should return an iterable of (z, y, x) points"""
        raise NotImplementedError()

    def wrap_field(self, fld_typ, name, data, **kwargs):
        """fld_type is 'Scalar' or 'Vector' or something like that"""
        crds = self.as_coordinates()
        return field.wrap_field(fld_typ, name, crds, data, **kwargs)


class Point(SeedGen):
    """Collection of points

    This is useful for wrapping up an ndarray of points to go
    into :meth:`vicid.calculator.streamline.streamlines`.
    """
    def __init__(self, points, cache=False, dtype=None):
        """points should be an n x 3 array of zyx points"""
        super(Point, self).__init__(cache=cache, dtype=dtype)
        self.params["points"] = points

    def nr_points(self, **kwargs):
        return self.points().shape[-1]

    def genr_points(self):
        pts = self.params["points"]
        if isinstance(pts, np.ndarray):
            if pts.shape[0] == 3:
                return pts.reshape((3, -1))
            elif pts.shape[-1] == 3:
                return np.array(pts.T).reshape((3, -1))
            else:
                raise ValueError("Malformed points")
        else:
            pts_arr = np.array(pts, dtype=self.dtype, copy=False)
            if pts_arr.shape[0] == 3:
                return pts_arr.reshape((3, -1))
            elif pts_arr.shape[-1] == 3:
                return np.array(pts_arr.T).reshape((3, -1))
            else:
                raise ValueError("Malformed points")

    def as_coordinates(self):
        raise NotImplementedError("not yet implemented")


class Line(SeedGen):
    """A line of seed points

    Defined by to endpoints
    """
    def __init__(self, p0, p1, res=20, cache=False, dtype=None):
        """ p0 & p1 are (z, y, x) points as list, tuple, or ndarray
        res is the number of need points to generate """
        super(Line, self).__init__(cache=cache, dtype=dtype)
        self.params["p0"] = p0
        self.params["p1"] = p1
        self.params["res"] = res

    def nr_points(self, **kwargs):
        return self.params["res"]

    def genr_points(self):
        p0 = self.params["p0"]
        p1 = self.params["p1"]
        res = self.params["res"]
        z = np.linspace(p0[0], p1[0], res).astype(self.dtype)
        y = np.linspace(p0[1], p1[1], res).astype(self.dtype)
        x = np.linspace(p0[2], p1[2], res).astype(self.dtype)
        return np.array([z, y, x])

    def as_coordinates(self):
        p0 = np.array(self.params["p0"])
        p1 = np.array(self.params["p1"])
        res = self.params["res"]

        dp = p1 - p0
        dist = np.sqrt(np.dot(dp, dp))
        x = np.linspace(0.0, dist, res)
        crd = coordinate.wrap_crds("nonuniform_cartesian", (('x', x),))
        return crd


class Plane(SeedGen):
    """A plane of seed points

    Defined by a point, a normal direction, a direction in the plane,
    and the length and width of the plane
    """
    def __init__(self, p0, Ndir, Ldir, len_l, len_m, res_l=20, res_m=20,
                 cache=False):
        """ plane is specified in L,M,N coords where N is normal to the plane,
        and L is projected into the plane. len_l and len_m is the extent
        of the plane in the l and m directions. res_l and res_m are the
        resolution of points in the two directions. Note that p0 is the center
        of the plane, so the plane extends from (-len_l/2, len_l/2) around
        p0, and similarly in the m direction.

        Also, if len_l or len_m is a 2-tuple (or list), that specifies the exact
        limits of the plane as opposed to going (-len/2, len/2) """
        super(Plane, self).__init__(cache=cache)
        if not isinstance(len_l, (list, tuple)):
            len_l = [-0.5 * len_l, 0.5 * len_l]
        if not isinstance(len_m, (list, tuple)):
            len_m = [-0.5 * len_m, 0.5 * len_m]

        self.params["p0"] = p0
        self.params["Ndir"] = Ndir
        self.params["Ldir"] = Ldir
        self.params["len_l"] = len_l
        self.params["len_m"] = len_m
        self.params["res_l"] = res_l
        self.params["res_m"] = res_m

    def nr_points(self, **kwargs):
        return self.params["res_l"] * self.params["res_m"]

    # def _make_arrays(self):
    #     pass

    def genr_points(self):
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
        l = np.linspace(len_l[0], len_l[1], res_l).astype(self.dtype)
        m = np.linspace(len_m[0], len_m[1], res_m).astype(self.dtype)
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

        l = np.linspace(len_l[0], len_l[1], res_l)
        m = np.linspace(len_m[0], len_m[1], res_m)

        crds = coordinate.wrap_crds("nonuniform_cartesian", (('y', m), ('x', l)))
        return crds


class Volume(SeedGen):
    """A volume of seed points

    Defined by two opposite corners of a box in 3d
    """
    def __init__(self, p0, p1, res=(20, 20, 20), cache=False):
        """ p1 & p2 are (z, y, x) points as list, tuple, or ndarray
        res is the number of need points to generate (nz, ny, nx) """
        super(Volume, self).__init__(cache=cache)
        self.params["p0"] = p0
        self.params["p1"] = p1
        self.params["res"] = res

    def nr_points(self, **kwargs):
        return np.prod(self.params["res"])

    def genr_points(self):
        crds = self._make_arrays()
        shape = [len(c) for c in crds]
        arr = np.empty([len(shape)] + [np.prod(shape)])
        for i, c in enumerate(crds):
            arr[i, :] = np.tile(np.repeat(c, np.prod(shape[:i])),
                                np.prod(shape[i + 1:]))
        return arr

    def _make_arrays(self):
        p0 = self.params["p0"]
        p1 = self.params["p1"]
        res = self.params["res"]
        z = np.linspace(p0[0], p1[0], res[0]).astype(self.dtype)
        y = np.linspace(p0[1], p1[1], res[1]).astype(self.dtype)
        x = np.linspace(p0[2], p1[2], res[2]).astype(self.dtype)
        return z, y, x

    def iter_points(self, **kwargs):
        z, y, x = self._make_arrays()
        return itertools.product(z, y, x)

    def as_coordinates(self):
        z, y, x = self._make_arrays()
        crd = coordinate.wrap_crds("nonuniform_cartesian",
                                   (('z', z), ('y', y), ('x', x)))
        return crd


class Sphere(SeedGen):
    """A sphere of points

    Defined by a center, and a radius
    """
    def __init__(self, p0, r, restheta=20, resphi=20, cache=False):
        super(Sphere, self).__init__(cache=cache)
        self.params["p0"] = np.array(p0, copy=False)
        self.params["r"] = r
        self.params["restheta"] = restheta
        self.params["resphi"] = resphi

    def nr_points(self, **kwargs):
        return self.params["restheta"] * self.params["resphi"]

    def genr_points(self):
        theta, phi = self._get_all_theta_phi()
        return self.spherical_to_zyx(theta, phi)

    def spherical_to_zyx(self, theta, phi):
        p0 = self.params["p0"]
        return p0.reshape((-1, 1)) + self._local_zyx(theta, phi)

    def as_coordinates(self):
        theta, phi = self._get_all_theta_phi()
        crds = coordinate.wrap_crds("nonuniform_cartesian",
                                    (('y', theta), ('x', phi)))
        return crds

    def _get_all_theta_phi(self):
        restheta = self.params["restheta"]
        resphi = self.params["resphi"]
        theta = np.linspace(0, np.pi, restheta + 1,
                            endpoint=True).astype(self.dtype)
        theta = 0.5 * (theta[1:] + theta[:-1])
        phi = np.linspace(0, 2.0 * np.pi, resphi,
                          endpoint=False).astype(self.dtype)
        return theta, phi

    def _local_zyx(self, theta, phi):
        r = self.params["r"]
        phi = np.asarray(phi).reshape(-1)
        theta = np.asarray(theta).reshape(-1)
        T, P = np.ix_(theta, phi)

        a = np.empty((3, len(theta) * len(phi)), dtype=self.dtype)
        # 2 == x, 1 == y, 0 == z
        a[2, :] = (r * np.sin(T) * np.cos(P)).reshape(-1)
        a[1, :] = (r * np.sin(T) * np.sin(P)).reshape(-1)
        a[0, :] = (r * np.cos(T) + 0.0 * P).reshape(-1)
        return a

class SphericalCap(Sphere):
    """A spherical cone or cap of seeds

    Defined by a center, and a point indicating the direction of the
    cone, and the half angle of the cone.
    """
    _euler_rot = None  # rotation matrix for 2 euler rotations

    def __init__(self, p0, p1, angle,
                 restheta=20, resphi=20,
                 r=None, cache=False):
        """ angle is the opening angle of the cone
        if r is given, p1 is moved to a distance r away from p0 """
        p0 = np.array(p0)
        p1 = np.array(p1)
        if r is None:
            r = np.sqrt(np.sum((p1 - p0)**2))
            print("calculated r:", r)
            # d = np.sqrt(np.sum((p1 - p0)**2))
            # p1 = p0 + (r / d) * (p1 - p0)
        super(SphericalCap, self).__init__(p0, r, restheta, resphi, cache=cache)
        self.params["p1"] = p1
        self.params["angle"] = angle * (np.pi / 180.0)

    @property
    def euler_rot(self):
        if self._euler_rot is None:
            p0 = self.params["p0"]
            p1 = self.params["p1"]
            script_r = p1 - p0
            script_rmag = np.sqrt(np.sum(script_r**2))

            theta0 = np.arccos(script_r[0] / script_rmag)
            phi0 = np.arctan2(script_r[1], script_r[2])

            # first 2 euler angle rotations, the 3rd one would be
            # the rotation around the cone's axis, so no need for that one
            # the convention used here is y-z-x, as in theta0 around y,
            # phi0 around z, and psi0 around x, but without the last one
            sint0 = np.sin(theta0)
            cost0 = np.cos(theta0)
            sinp0 = np.sin(phi0)
            cosp0 = np.cos(phi0)
            mat = np.array([[cost0,            0.0, -sint0],          # pylint: disable=C0326
                            [sinp0 * sint0,  cosp0,  sinp0 * cost0],  # pylint: disable=C0326
                            [cosp0 * sint0, -sinp0,  cosp0 * cost0],  # pylint: disable=C0326
                           ], dtype=self.dtype)
            self._euler_rot = mat
        return self._euler_rot

    def spherical_to_zyx(self, theta, phi):
        p0 = self.params["p0"]
        a = self._local_zyx(theta, phi)
        return p0.reshape((-1, 1)) + np.dot(self.euler_rot, a)

    def _get_all_theta_phi(self):
        angle = self.params["angle"]
        restheta = self.params["restheta"]
        resphi = self.params["resphi"]

        theta = np.linspace(angle, 0.0, restheta, endpoint=False)
        theta = np.array(theta[::-1], dtype=self.dtype)
        phi = np.linspace(0, 2.0 * np.pi, resphi,
                          endpoint=False).astype(self.dtype)
        return theta, phi


class Circle(SphericalCap):
    """A circle of seeds

    Defined by a center and a point normal to the plane of the circle
    """
    def __init__(self, p0, p1, res=20, r=None,
                 angle=90.0, cache=False):
        super(Circle, self).__init__(p0, p1, angle, restheta=1,
                                     resphi=res, r=r, cache=cache)

# class SphericalCap2(SphericalCap):
#     def _get_all_theta_phi(self):
#         angle = self.params["angle"]
#         restheta = self.params["restheta"]
#         resphi = self.params["resphi"]

#         # theta = np.linspace(angle, 0.0, restheta, endpoint=False)
#         # theta = np.array(theta[::-1], dtype=self.dtype)
#         # phi = np.linspace(0, 2.0 * np.pi, resphi,
#         #                   endpoint=False).astype(self.dtype)
#         theta = np.linspace(-angle, angle, restheta, endpoint=True).astype(self.dtype)
#         phi = np.linspace(0.0, np.pi, resphi, endpoint=False).astype(self.dtype)
#         return theta, phi


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D #pylint: disable=W0611
    # ax = plt.gca(projection='3d')
    # lin = Line((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)).points()
    # ax.plot(lin[:, 2], lin[:, 1], lin[:, 0], 'g.')
    # s = Sphere((0.0, 0.0, 0.0), 2.0, 10, 20).points()
    # ax.plot(s[:, 2], s[:, 1], s[:, 0], 'b')
    # p = Plane((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0),
    #                  2.0, 3.0, 10, 20).points()
    # ax.plot(p[:, 2], p[:, 1], p[:, 0], 'r')
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.show()

    from timeit import default_timer as time
    from mayavi import mlab
    from viscid import vlab
    from viscid.calculator import calc
    from viscid.calculator import cycalc

    B = vlab.get_dipole(m=[0.0, 0.0, -1.0])
    bmag = calc.magnitude(B)
    t0 = time()
    _p0 = [1.0, 1.0, 1.0]
    _p1 = [0.0, 0.0, 0.0]
    cap = SphericalCap(_p0, _p1, 120.0, r=1.0, cache=True)
    interp_vals = cycalc.interp_trilin(bmag, cap)
    t1 = time()
    # logger.info("interp took {0:.3e}s to compute.".format(t1 - t0))
    _pts = cap.points()
    mlab.points3d(_p0[2], _p0[1], _p0[0],
                  scale_factor=0.07, color=(0.0, 0.0, 0.0))
    mlab.points3d(_p1[2], _p1[1], _p1[0],
                  scale_factor=0.07, color=(1.0, 1.0, 1.0))
    mlab.points3d(_pts[2], _pts[1], _pts[0], interp_vals,
                  scale_factor=0.07, scale_mode="none")
    mlab.axes(nb_labels=5)
    mlab.show()

##
## EOF
##
