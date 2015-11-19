#!/usr/bin/env python
"""Helpers to generate seed points for streamlines"""

from __future__ import print_function
import itertools

import numpy as np

import viscid
from viscid.compat import izip

__all__ = ['make_rotation_matrix', 'to_seeds', 'SeedGen', 'Point',
           'MeshPoints', 'Line', 'Plane', 'Volume', 'Sphere', 'SphericalCap',
           'Circle', 'SphericalPatch', 'PolarIonosphere']

def to_seeds(pts):
    """Try to turn anything into a set of seeds

    Args:
        points (ndarray or list): should look like something that
            np.array(points) can turn to an Nx3 array of xyz points.
            This can be 3xN so long as N != 3.
    """
    if (hasattr(pts, "nr_points") and hasattr(pts, "iter_points") and
        hasattr(pts, "points")):  # pylint: disable=bad-continuation
        return pts
    else:
        return Point(pts)

def make_rotation_matrix(origin, p1, p2, roll=0.0):
    """Make a matrix that rotates origin-p1 to origin-p2

    Note:
        If you want `p1` and `p2` to be vectors, just set `origin` to
        `[0, 0, 0]`.

    Args:
        origin (ndarray): Origin of the rotation in 3D
        p1 (ndarray): Some point in 3D
        p2 (ndarray): A point in 3D such that p1 will be rotated
            around the origin so that it lies along the origin-p2
            line
        roll (float): Angle (in degrees) of roll around the
            origin-p2 vector

    Returns:
        ndarray: 3x3 orthonormal rotation matrix
    """
    origin = np.asarray(origin).reshape(-1)
    p1 = np.asarray(p1).reshape(-1)
    p2 = np.asarray(p2).reshape(-1)

    # normalize the origin-p1 and origin-p2 vectors as `a` and `b`
    a = (p1 - origin) / np.linalg.norm(p1 - origin).reshape(-1)
    b = (p2 - origin) / np.linalg.norm(p2 - origin).reshape(-1)

    # Use Rodrigues' Rotation Formula to rotate theta degrees around the
    # cross prodect of a and b. Here, K is the skew-symmetric cross-product
    # matrix
    theta = np.arccos(np.sum(a * b))
    k = np.cross(a, b)
    if np.allclose(k, [0, 0, 0]):
        k = np.cross(a, [1, 0, 0])
        if np.allclose(k, [0, 0, 0]):
            k = np.cross(a, [0, 1, 0])
    k = k / np.linalg.norm(k)
    K = np.array([[0    , -k[2],  k[1]],  # pylint: disable=bad-whitespace
                  [k[2] ,     0, -k[0]],  # pylint: disable=bad-whitespace
                  [-k[1],  k[0],     0]])  # pylint: disable=bad-whitespace
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    # now use the same formula to roll around the the origin-p2 axis
    phi = (np.pi / 180.0) * roll
    K = np.array([[0    , -b[2],  b[1]],  # pylint: disable=bad-whitespace
                  [b[2] ,     0, -b[0]],  # pylint: disable=bad-whitespace
                  [-b[1],  b[0],     0]])  # pylint: disable=bad-whitespace
    Rroll = np.eye(3) + np.sin(phi) * K + (1 - np.cos(phi)) * np.dot(K, K)

    return np.dot(Rroll, R)


class SeedGen(object):
    """All about seeds

    These objects are good for defining the root points of streamlines
    or locations for interpolation.

    - For Developers
        - Mandatory
            Subclasses **must** override

            - :py:meth:`get_nr_points`: get total number of seeds
            - :py:meth:`get_local_shape`: get the shape of the ndarray
              of local coordinates
            - :py:meth:`to_3d`: transform an array in local
              coordinates to 3D space. Should return a 3xN ndarray.
            - :py:meth:`to_local`: transform a 3xN ndarray to an
              ndarray of coordinates in local space. It's ok to raise
              NotImplementedError.
            - :py:meth:`_make_local_points`: Make an ndarray that is
              used in :py:meth:`_make_3d_points`
            - :py:meth:`_make_local_axes`: Make a tuple of arrays with
              lengths that match :py:attr:`local_shape`. It's ok to
              raise NotImplementedError if the local representation has
              no axes.

        - Optional
            Subclasses *may* override

            - :py:meth:`_make_3d_points`: By default, this is a
              combination of :py:meth:`_make_local_points` and
              :py:meth:`to_3d`. It can be overridden if there is a more
              efficient way te get points in 3d.
            - :py:meth:`iter_points`:  If a seed generator can be made
              lazy, then this should return an iterator that yields
              (x, y, z) points.
            - :py:meth:`get_rotation`: This should return an
              orthonormal rotation matrix if there is some rotation
              required to go from local coordinates to 3D space.
            - :py:meth:`as_mesh`
            - :py:meth:`as_coordinates`
            - :py:meth:`wrap_field`

        - Off Limits
            Don't override unless you know what you're doing

            - :py:meth:`get_points`

    - For Users
        - 3D
            These attributes give the seed points in 3D (xyz) space

            - :py:attr:`nr_points`
            - :py:meth:`get_points`
            - :py:meth:`iter_points`
            - :py:meth:`to_3d`

        - Generator Specific
            There is also the concept of local points, which are points
            that are meaningful for the specific SeedGen subclass. For
            example, for the Sphere subclass, local points are labled
            with (theta, phi) on the surface of the sphere. As such,
            :py:meth:`wrap_field` can be used to facilitate wrapping
            the result of :py:func:`trilin_interp` for easy plotting.

            - :py:attr:`local_shape`
            - :py:meth:`to_local`
            - :py:meth:`as_mesh`
            - :py:meth:`as_coordinates`
            - :py:meth:`wrap_field`
    """
    _cache = None
    _points = None
    dtype = None

    def __init__(self, cache=False, dtype=None):
        self._cache = cache
        self.dtype = dtype

    @property
    def nr_points(self):
        return self.get_nr_points()

    @property
    def local_shape(self):
        """Shape of local representation of seeds"""
        return self.get_local_shape()

    def get_nr_points(self, **kwargs):
        raise NotImplementedError()

    def get_local_shape(self, **kwargs):
        raise NotImplementedError()

    def to_3d(self, pts_local):
        """Transform points from the seed coordinate space to 3d

        Args:
            pts_local (ndarray): An array of points in local crds

        Returns:
            3xN ndarray of N xyz points
        """
        raise NotImplementedError()

    def to_local(self, pts_3d):
        """Transform points from 3d to the seed coordinate space

        Args:
            pts_local (ndarray): A 3xN array of N points in 3d

        Returns:
            ndarray similar in shape to self.local_shape
        """
        raise NotImplementedError()

    def _make_local_points(self):
        """Make a set of points in the local representation"""
        raise NotImplementedError()

    def _make_local_axes(self):
        """Make a tuple of arrays that match :py:attr:`local_shape`"""
        raise NotImplementedError()

    def _make_3d_points(self):
        """Make a 3xN ndarray of N xyz points"""
        return self.to_3d(self._make_local_points())

    def points(self, **kwargs):
        """Alias for :py:meth:`get_points`"""
        return self.get_points(**kwargs)

    def get_points(self, **kwargs):  # pylint: disable=unused-argument
        """Get a 3xN ndarray of N xyz points"""
        if self._points is not None:
            return self._points
        pts = self._make_3d_points()
        if self._cache:
            self._points = pts
        return pts

    def iter_points(self, **kwargs):  # pylint: disable=unused-argument
        """Make an iterator that yields `(x, y, z)` points

        This can be overridden in a subclass if it's more efficient to
        iterate than a call to :meth:`_make_points`. Calling iter_points
        should make an effort to be the fastest way to iterate through
        the points, regardless of caching.

        Note:
            In Cython, it seems to take twice the time to
            iterate over a generator than to use a for loop over
            indices of an array, but only if the array already exists.
        """
        pts = self.get_points()
        return izip(pts[0, :], pts[1, :], pts[2, :])

    def get_rotation(self):
        """Make a rotation matrix to go local -> real 3D space"""
        raise RuntimeError("{0} does not define get_rotation"
                           "".format(type(self).__name__))

    def as_coordinates(self):
        """Make :py:class:`Coordinates` for the local representation"""
        raise RuntimeError("{0} does not define as_coordinates"
                           "".format(type(self).__name__))

    def wrap_field(self, data, name="NoName", fldtype="scalar", **kwargs):
        """Wrap an ndarray into a field in the local representation"""
        crds = self.as_coordinates()
        return viscid.wrap_field(data, crds, name=name, fldtype=fldtype,
                                 **kwargs)

    def as_mesh(self):
        """Make a 3xUxV array that describes a 3D mesh surface"""
        raise RuntimeError("{0} does not define as_mesh"
                           "".format(type(self).__name__))

    def arr2mesh(self, arr):  # pylint: disable=unused-argument
        raise RuntimeError("{0} does not define arr2mesh"
                           "".format(type(self).__name__))

    def wrap_mesh(self, *arrs):
        arrs = [a.data if isinstance(a, viscid.field.Field) else a for a in arrs]
        vertices = self.as_mesh()
        arrs = [self.arr2mesh(arr).astype(arr.dtype) for arr in arrs]
        return [vertices] + arrs


class Point(SeedGen):
    """Collection of points"""
    def __init__(self, pts, cache=False, dtype=None):
        """Seed with an explicit set of points

        Args:
            pts (ndarray or list): should look like something that
                np.array(pts) can turn to an 3xN array of xyz
                points. This can be Nx3 as long as N != 3.
        """
        super(Point, self).__init__(cache=cache, dtype=dtype)
        self.pts = pts

    def get_nr_points(self, **kwargs):
        return self.get_points().shape[-1]

    def get_local_shape(self, **kwargs):
        return (3, self.nr_points)

    def to_3d(self, pts_local):
        return pts_local

    def to_local(self, pts_3d):
        return pts_3d

    def _make_local_points(self):
        if isinstance(self.pts, np.ndarray):
            if self.pts.shape[0] == 3:
                self.pts = self.pts.reshape((3, -1))
            elif self.pts.shape[-1] == 3:
                self.pts = self.pts.T.reshape((3, -1))
            else:
                raise ValueError("Malformed points")
        else:
            pts_arr = np.array(self.pts, dtype=self.dtype, copy=False)
            if pts_arr.shape[0] == 3:
                self.pts = pts_arr.reshape((3, -1))
            elif pts_arr.shape[-1] == 3:
                self.pts = pts_arr.T.reshape((3, -1))
            else:
                raise ValueError("Malformed points")

        return self.pts

    def _make_local_axes(self):
        return np.arange(self.nr_points)

    def as_coordinates(self):
        x = self._make_local_axes()
        crds = viscid.wrap_crds("nonuniform_cartesian", (('x', x), ))
        return crds


class MeshPoints(SeedGen):
    """Generic points with 2d mesh information


    """
    def __init__(self, pts, cache=False, dtype=None):
        """Generic set of 2d points

        Args:
            pts (ndarray): must have shape 3xNUxNV for the uv mesh
                directions
        """
        super(MeshPoints, self).__init__(cache=cache, dtype=dtype)
        if pts.shape[0] != 3:
            raise ValueError("First index of pts must be xyz")
        self.pts = pts
        self.nu = pts.shape[1]
        self.nv = pts.shape[2]

    def get_nr_points(self, **kwargs):
        return self.nu * self.nv

    def get_local_shape(self):
        return (self.nu, self.nv)

    def to_3d(self, pts_local):
        return pts_local.reshape(3, -1)

    def to_local(self, pts_3d):
        raise NotImplementedError()

    def _make_local_points(self):
        return self.pts

    def _make_local_axes(self):
        return np.arange(self.nu), np.arange(self.nv)

    def as_coordinates(self):
        l, m = self._make_local_axes()
        crds = viscid.wrap_crds("nonuniform_cartesian", (('x', l), ('y', m)))
        return crds

    def as_mesh(self):
        return self.get_points().reshape(3, self.nu, self.nv)

    def arr2mesh(self, arr):
        return arr.reshape(self.local_shape)


class Line(SeedGen):
    """A line of seed points"""
    def __init__(self, p0=(0, 0, 0), p1=(1, 0, 0), n=20,
                 cache=False, dtype=None):
        """p0 & p1 are `(x, y, z)` points as

        Args:
            p0 (list, tuple, or ndarray): starting point `(x, y, z)`
            p1 (list, tuple, or ndarray): ending point `(x, y, z)`
            n (int): number of points on the line
        """
        super(Line, self).__init__(cache=cache, dtype=dtype)
        self.p0 = np.asarray(p0, dtype=self.dtype)
        self.p1 = np.asarray(p1, dtype=self.dtype)
        self.n = n

    def get_nr_points(self, **kwargs):
        return self.n

    def get_local_shape(self, **kwargs):
        return (self.nr_points, )

    def to_3d(self, pts_local):
        ds = self.p1 - self.p0
        x = (self.p0[0] + ds[0] * pts_local).astype(self.dtype)
        y = (self.p0[1] + ds[1] * pts_local).astype(self.dtype)
        z = (self.p0[2] + ds[2] * pts_local).astype(self.dtype)
        return np.vstack([x, y, z])

    def to_local(self, pts_3d):
        raise NotImplementedError()

    def _make_local_points(self):
        return self._make_local_axes()

    def _make_local_axes(self):
        return np.linspace(0.0, 1.0, self.n)

    def as_coordinates(self):
        dp = self.p1 - self.p0
        dist = np.sqrt(np.dot(dp, dp))
        x = np.linspace(0.0, dist, self.n)
        crd = viscid.wrap_crds("nonuniform_cartesian", (('x', x),))
        return crd

    def as_mesh(self):
        return self.get_points().reshape(3, self.n, 1)


class Plane(SeedGen):
    """A plane of seed points"""

    def __init__(self, p0=(0, 0, 0), pN=(0, 0, 1), pL=(1, 0, 0),
                 len_l=2, len_m=2, nl=20, nm=20, NL_are_vectors=True,
                 cache=False, dtype=None):
        """Make a plane in L,M,N coordinates.

        Args:
            p0 (list, tuple, or ndarray): Origin of plane as (x, y, z)
            pN (list, tuple, or ndarray): Point as (x, y, z) such that
                a vector from p0 to pN is normal to the plane
            pL (list, tuple, or ndarray): Point as (x, y, z) such
                that a vector from p0 to pL is the `L` (`x`) axis of
                the plane. The `M` (`y`) axis is `N` cross `L`. `pL`
                is projected into the plane if needed.
            len_l (float, list): Length of the plane along the `L`
                axis. If given as a float, the plane extents from
                (-len_l / 2, len_l / 2). If given as a list, one can
                specify the axis limits explicitly.
            len_m (float, list): Length of the plane along the `M` axis
                with the same semantics as `len_l`.
            nl (int): number of points in the `L` direction
            nm (int): number of points in the `M` direction
            NL_are_vectors (bool): If True, then pN and pL are
                interpreted as vectors such that the basis vectors
                are `pN - p0` and `pL - p0`, otherwise they are
                interpreted as basis vectors themselves.

        Raises:
            RuntimeError: if Ndir == Ldir, can't determine a plane
        """

        super(Plane, self).__init__(cache=cache, dtype=dtype)

        p0 = np.array(p0, dtype=self.dtype).reshape(-1)
        pN = np.array(pN, dtype=self.dtype).reshape(-1)
        pL = np.array(pL, dtype=self.dtype).reshape(-1)

        if NL_are_vectors:
            Ndir = pN
            Ldir = pL
        else:
            Ndir = pN - p0
            Ldir = pL - p0

        # normalize normal direction
        Ndir = Ndir / np.sqrt(np.dot(Ndir, Ndir))
        # project Ldir into the plane perpendicular to Ndir
        # (N already normalized)
        Ldir = Ldir - np.dot(Ldir, Ndir) * Ndir
        # normalize Ldir
        Ldir = Ldir / np.sqrt(np.dot(Ldir, Ldir))
        # compute Mdir
        Mdir = np.cross(Ndir, Ldir)

        if np.allclose(Mdir, [0, 0, 0]):
            raise RuntimeError("N and L can't be parallel")

        if not isinstance(len_l, (list, tuple)):
            len_l = [-0.5 * len_l, 0.5 * len_l]
        if not isinstance(len_m, (list, tuple)):
            len_m = [-0.5 * len_m, 0.5 * len_m]

        self.p0 = p0
        self.Ndir = Ndir
        self.Ldir = Ldir
        self.Mdir = Mdir
        self.len_l = len_l
        self.len_m = len_m
        self.nl = nl
        self.nm = nm

    def get_nr_points(self, **kwargs):
        return self.nl * self.nm

    def get_local_shape(self):
        return (self.nl, self.nm)

    def to_3d(self, pts_local):
        lmn_to_xyz = self.get_rotation()
        rot_pts = np.einsum("ij,jk->ik", lmn_to_xyz, pts_local)
        return self.p0.reshape((3, 1)) + rot_pts

    def to_local(self, pts_3d):
        xyz_to_lmn = self.get_rotation().T
        return np.einsum("ij,jk->ik",
                         xyz_to_lmn, pts_3d - self.p0.reshape((3, 1)))

    def _make_local_points(self):
        plane_lmn = np.empty((3, self.nl * self.nm), dtype=self.dtype)
        l, m = self._make_local_axes()
        plane_lmn[0, :] = np.repeat(l, self.nm)
        plane_lmn[1, :] = np.tile(m, self.nl)
        plane_lmn[2, :] = 0.0
        return plane_lmn

    def _make_local_axes(self):
        len_l, len_m = self.len_l, self.len_m
        l = np.linspace(len_l[0], len_l[1], self.nl).astype(self.dtype)
        m = np.linspace(len_m[0], len_m[1], self.nm).astype(self.dtype)
        return l, m

    def get_rotation(self):
        """Get rotation from lmn -> xyz

        To transform xyz -> lmn, transpose the result since it's
        an orthogonal matrix.

        Note:
            Remember that the p0 offset (the center of the plane) is
            not done with the rotation.

        Returns:
            3x3 ndarray

        Example:
            >>> # Transform a set of points from lmn coordinates to xyz
            >>> import numpy as np
            >>> import viscid
            >>>
            >>> # p0 is the origin of the plane
            >>> p0 = np.array([0, 0, 0]).reshape(-1, 1)
            >>> plane = viscid.Plane(p0=p0, pN=(1, 1, 0),
            >>>                      pL=(1, -1, 0),
            >>>                      len_l=2.0, len_m=2.0)
            >>> lmn_to_xyz = plane.get_rotation()
            >>>
            >>> # make 10 random points in lmn coorinates
            >>> lmn = 2 * np.random.rand(3, 10) - 1
            >>> lmn[2] = 0.0
            >>> xyz = p0 + np.dot(lmn_to_xyz, lmn)
            >>>
            >>> from viscid.plot import mvi
            >>> verts = plane.as_mesh()
            >>> mvi.mlab.mesh(verts[0], verts[1], verts[2])
            >>> mvi.mlab.points3d(xyz[0], xyz[1], xyz[2])
            >>> mvi.show()

            >>> # Transform vector compenents from xyz to lmn
            >>> import numpy as np
            >>> import viscid
            >>>
            >>> x = np.linspace(-1, 1, 32)
            >>> y = np.linspace(-1, 1, 32)
            >>> z = np.linspace(-1, 1, 32)
            >>> Bxyz = viscid.zeros([x, y, z], center='node',
            >>>                     nr_comps=3, layout="interlaced")
            >>> Bxyz['x'] = 1.0
            >>>
            >>> plane = viscid.Plane(p0=[0, 0, 0], pN=[1, 1, 0],
            >>>                      pL=[1, -1, 1],
            >>>                      len_l=0.5, len_m=0.5)
            >>> vals = viscid.interp_trilin(Bxyz, plane)
            >>> B_interp = plane.wrap_field(vals, fldtype='vector',
            >>>                             layout='interlaced')
            >>> xyz_to_lmn = plane.get_rotation().T
            >>> Blmn = np.einsum("ij,lmj->lmi", xyz_to_lmn, B_interp)
            >>> Blmn = B_interp.wrap(Blmn)
            >>>
            >>> # use xyz to show in 3d via mayavi
            >>> from viscid.plot import mvi
            >>> verts, s = plane.wrap_mesh(Blmn['z'].data)
            >>> mvi.mlab.mesh(verts[0], verts[1], verts[2], scalars=s)
            >>> verts, vx, vy, vz = plane.wrap_mesh(B_interp['x'].data,
            >>>                                     B_interp['y'].data,
            >>>                                     B_interp['z'].data)
            >>> mvi.mlab.quiver3d(verts[0], verts[1], verts[2],
            >>>                   vx, vy, vz)
            >>> mvi.show()
            >>>
            >>> # use lmn to show in-plane / out-of-plane
            >>> from viscid.plot import mpl
            >>> mpl.plot(Blmn['z'])  # z means n here
            >>> mpl.plot2d_quiver(Blmn)
            >>> mpl.plt.show()
        """
        return np.array([self.Ldir, self.Mdir, self.Ndir]).T

    def as_coordinates(self):
        l, m = self._make_local_axes()
        crds = viscid.wrap_crds("nonuniform_cartesian", (('x', l), ('y', m)))
        return crds

    def as_mesh(self):
        return self.get_points().reshape(3, self.nl, self.nm)

    def arr2mesh(self, arr):
        return arr.reshape(self.local_shape)


class Volume(SeedGen):
    """A volume of seed points

    Defined by two opposite corners of a box in 3D
    """
    def __init__(self, xl=(-1, -1, -1), xh=(1, 1, 1), n=(20, 20, 20),
                 cache=False, dtype=None):
        """Make a volume

        Args:
            xl (list, tuple, or ndarray): Lower corner as (x, y, z)
            xh (list, tuple, or ndarray): Upper corner as (x, y, z)
            n (list, tuple, or ndarray): number of points (nx, ny, nz)
                defaults to (20, 20, 20)
        """
        super(Volume, self).__init__(cache=cache, dtype=dtype)

        self.xl = np.asarray(xl, dtype=self.dtype)
        self.xh = np.asarray(xh, dtype=self.dtype)
        self.n = np.empty_like(self.xl, dtype='i')
        self.n[:] = n

    def get_nr_points(self, **kwargs):
        return np.prod(self.n)

    def get_local_shape(self, **kwargs):
        return tuple(self.n)

    def to_3d(self, pts_local):
        return pts_local

    def to_local(self, pts_3d):
        return pts_3d

    def _make_local_points(self):
        crds = self._make_local_axes()
        shape = self.local_shape
        arr = np.empty([len(shape)] + [np.prod(shape)])
        for i, c in enumerate(crds):
            arr[i, :] = np.repeat(np.tile(c, np.prod(shape[:i])),
                                  np.prod(shape[i + 1:]))
        return arr

    def _make_local_axes(self):
        x = np.linspace(self.xl[0], self.xh[0], self.n[0]).astype(self.dtype)
        y = np.linspace(self.xl[1], self.xh[1], self.n[1]).astype(self.dtype)
        z = np.linspace(self.xl[2], self.xh[2], self.n[2]).astype(self.dtype)
        return x, y, z

    def iter_points(self, **kwargs):
        x, y, z = self._make_local_axes()
        return itertools.product(x, y, z)

    def as_coordinates(self):
        x, y, z = self._make_local_axes()
        crd = viscid.wrap_crds("nonuniform_cartesian",
                               (('x', x), ('y', y), ('z', z)))
        return crd

    def as_mesh(self):
        n = self.n.tolist()
        try:
            ind = n.index(1)
            pts = self.get_points()
            n.pop(ind)
            return pts.reshape([3] + n)
        except ValueError:
            raise RuntimeError("Can't make a 2d surface from a 3d volume")

    def arr2mesh(self, arr):
        vertices = self.as_mesh()
        return arr.reshape(vertices.shape[1:])


class Sphere(SeedGen):
    """Make seeds on the surface of a sphere"""

    def __init__(self, p0=(0, 0, 0), r=0.0, pole=(0, 0, 1), ntheta=20, nphi=20,
                 thetalim=(0, 180.0), philim=(0, 360.0), phi_endpoint=False,
                 pole_is_vector=True, theta_phi=False, roll=0.0,
                 cache=False, dtype=None):
        """Make seeds on the surface of a sphere

        Args:
            p0 (list, tuple, or ndarray): Origin of sphere as (x, y, z)
            r (float): Radius of sphere; or calculated from pole if 0
            pole (list, tuple, or ndarray): Vector pointing
                in the direction of the north pole of the sphere.
                Defaults to (0, 0, 1).
            ntheta (int): Number of points in theta
            nphi (int): Number of points in phi
            thetalim (list): min and max theta (in degrees)
            philim (list): min and max phi (in degrees)
            phi_endpoint (bool): if true, then let phi inclue upper
                value. This is false by default since 0 == 2pi.
            pole_is_vector (bool): Whether pole is a vector or a
                vector
            theta_phi (bool): If True, reult can be reshaped as
                (theta, phi), otherwise it's (phi, theta)
        """
        super(Sphere, self).__init__(cache=cache, dtype=dtype)

        self.p0 = np.asarray(p0, dtype=self.dtype)

        if pole_is_vector:
            self.pole = np.asarray(pole, dtype=self.dtype)
        else:
            if pole is None:
                pole = p0 + np.asarray([0, 0, 1], dtype=self.dtype)
            else:
                pole = np.asarray(pole, dtype=self.dtype)

            self.pole = pole - p0

        if not r:
            r = np.linalg.norm(self.pole)
        self.pole = self.pole / np.linalg.norm(self.pole)

        if not (len(thetalim) == len(philim) == 2):
            raise ValueError("thetalim and philim must have both min and max")

        self.r = r
        self.ntheta = ntheta
        self.nphi = nphi
        self.thetalim = np.deg2rad(thetalim)
        self.philim = np.deg2rad(philim)
        self.phi_endpoint = phi_endpoint
        self.theta_phi = theta_phi
        self.roll = roll

    def get_nr_points(self, **kwargs):
        return self.ntheta * self.nphi

    def get_local_shape(self, **kwargs):
        if self.theta_phi:
            return (self.ntheta, self.nphi)
        else:
            return (self.nphi, self.ntheta)

    def to_3d(self, pts_local):
        r, T, P = pts_local[0, :], pts_local[1, :], pts_local[2, :]
        a = np.empty((3, pts_local.shape[1]), dtype=self.dtype)
        a[0, :] = (r * np.sin(T) * np.cos(P))
        a[1, :] = (r * np.sin(T) * np.sin(P))
        a[2, :] = (r * np.cos(T) + 0.0 * P)

        rot_pts = np.einsum("ij,jk->ik", self.get_rotation(), a)
        return self.p0.reshape((3, 1)) + rot_pts

    def to_local(self, pts_3d):
        raise NotImplementedError()

    def _make_local_points(self):
        arr = np.empty((3, self.ntheta * self.nphi), dtype=self.dtype)
        theta, phi = self._make_local_axes()
        arr[0, :] = self.r
        # Note: arr[1, :] is always theta and arr[2, :] is always phi
        # what changes is the way the points are ordered in the 2nd dimension
        if self.theta_phi:
            arr[1, :] = np.repeat(theta, self.nphi)
            arr[2, :] = np.tile(phi, self.ntheta)
        else:
            arr[1, :] = np.tile(theta, self.nphi)
            arr[2, :] = np.repeat(phi, self.ntheta)
        return arr

    def _make_local_axes(self):
        theta = np.linspace(self.thetalim[0], self.thetalim[1], self.ntheta + 1,
                            endpoint=True).astype(self.dtype)
        theta = 0.5 * (theta[1:] + theta[:-1])
        phi = np.linspace(self.philim[0], self.philim[1], self.nphi,
                          endpoint=self.phi_endpoint).astype(self.dtype)
        return theta, phi

    def get_rotation(self):
        return make_rotation_matrix([0, 0, 0], [0, 0, 1], self.pole,
                                    roll=self.roll)

    def as_coordinates(self):
        theta, phi = self._make_local_axes()
        if self.theta_phi:
            crds = viscid.wrap_crds("nonuniform_cartesian",
                                    (('x', theta), ('y', phi)))
        else:
            crds = viscid.wrap_crds("nonuniform_cartesian",
                                    (('x', phi), ('y', theta)))
        return crds

    @property
    def _is_whole_sphere(self):
        delta_theta = self.thetalim[1] - self.thetalim[0]
        delta_phi = self.philim[1] - self.philim[0]
        return np.allclose([delta_theta, delta_phi], [np.pi, 2.0 * np.pi])

    def as_mesh(self):
        new_shape = [3] + list(self.local_shape)
        pts = self.get_points().reshape(new_shape)

        if self._is_whole_sphere:
            if self.theta_phi:
                pole0 = np.empty((3, 1, self.nphi), dtype=pts.dtype)
                pole1 = np.empty((3, 1, self.nphi), dtype=pts.dtype)
            else:
                pole0 = np.empty((3, self.nphi, 1), dtype=pts.dtype)
                pole1 = np.empty((3, self.nphi, 1), dtype=pts.dtype)

            pole0[...] = (self.p0 + (self.r * self.pole)).reshape(3, 1, 1)
            pole1[...] = (self.p0 - (self.r * self.pole)).reshape(3, 1, 1)

            if self.theta_phi:
                pts = np.concatenate([pole0, pts, pole1], axis=1)
                pts = np.concatenate([pts, pts[:, :, 0, None]], axis=2)
            else:
                pts = np.concatenate([pole0, pts, pole1], axis=2)
                pts = np.concatenate([pts, pts[:, 0, None, :]], axis=1)

        if not self.theta_phi:
            pts = pts[:, ::-1, :]  # normals out

        return pts

    def arr2mesh(self, arr):
        arr = arr.reshape(self.local_shape)

        if self._is_whole_sphere:
            if self.theta_phi:
                p0 = np.repeat(np.mean(arr[0, :, None], keepdims=1),
                               arr.shape[1], axis=1)
                p1 = np.repeat(np.mean(arr[-1, :, None], keepdims=1),
                               arr.shape[1], axis=1)
                arr = np.concatenate([p0, arr, p1], axis=0)
                arr = np.concatenate([arr, arr[:, 0, None]], axis=1)
            else:
                p0 = np.repeat(np.mean(arr[:, 0, None], keepdims=1),
                               arr.shape[0], axis=0)
                p1 = np.repeat(np.mean(arr[:, -1, None], keepdims=1),
                               arr.shape[0], axis=0)
                arr = np.concatenate([p0, arr, p1], axis=1)
                arr = np.concatenate([arr, arr[0, None, :]], axis=0)

        if not self.theta_phi:
            arr = arr[::-1, :]  # normals out

        return arr


class SphericalCap(Sphere):  # pylint: disable=abstract-class-little-used
    """A spherical cone or cap of seeds

    Defined by a center, and a point indicating the direction of the
    cone, and the half angle of the cone.
    """
    def __init__(self, p0=(0, 0, 0), r=0.0, pole=(0, 0, 1), angle=90.0,
                 ntheta=20, nphi=20, pole_is_vector=True, theta_phi=False,
                 roll=0.0, cache=False, dtype=None):
        """Summary

        Args:
            p0 (list, tuple, or ndarray): Origin of sphere as (x, y, z)
            r (float): Radius of sphere; or calculated from pole if 0
            pole (list, tuple, or ndarray): Vector pointing
                in the direction of the north pole of the sphere.
                Defaults to (0, 0, 1).
            angle (float): cone angle of the cap in degrees
            ntheta (int): Number of points in theta
            nphi (int): Number of points in phi
            pole_is_vector (bool): Whether pole is a vector or a
                vector
            theta_phi (bool): If True, reult can be reshaped as
                (theta, phi), otherwise it's (phi, theta)
        """
        super(SphericalCap, self).__init__(p0, r=r, pole=pole, ntheta=ntheta,
                                           nphi=nphi,
                                           pole_is_vector=pole_is_vector,
                                           theta_phi=theta_phi, roll=roll,
                                           cache=cache, dtype=dtype)
        self.angle = angle * (np.pi / 180.0)

    def to_local(self, pts_3d):
        raise NotImplementedError()

    def _make_local_axes(self):
        theta = np.linspace(self.angle, 0.0, self.ntheta, endpoint=False)
        theta = np.array(theta[::-1], dtype=self.dtype)
        phi = np.linspace(0, 2.0 * np.pi, self.nphi,
                          endpoint=False).astype(self.dtype)
        return theta, phi

    def as_mesh(self):
        pts = super(SphericalCap, self).as_mesh()

        # don't include anti-pole direction since this is a cap
        if self.theta_phi:
            pts = pts[:, :-1, :]
        else:
            pts = pts[:, :, :-1]

        return pts

    def arr2mesh(self, arr):
        arr = super(SphericalCap, self).arr2mesh(arr)

        # don't include anti-pole direction since this is a cap
        if self.theta_phi:
            arr = arr[:-1, :]
        else:
            arr = arr[:, :-1]

        return arr

class Circle(SphericalCap):
    """A circle of seeds

    Defined by a center and a point normal to the plane of the circle
    """
    def __init__(self, p0=(0, 0, 0), pole=(0, 0, 1), n=20, r=None,
                 pole_is_vector=True, roll=0.0, cache=False, dtype=None):
        """Summary

        Args:
            p0 (list, tuple, or ndarray): Center of circle as (x, y, z)
            pole (list, tuple, or ndarray): Vector pointing
                in the direction normal to the circle. Defaults to
                (0, 0, 1).
            n (int): Number of points around the circle
            r (float): Radius of circle; or calculated from pole if 0
            pole_is_vector (bool): Whether pole is a vector or a
                vector
        """
        super(Circle, self).__init__(p0, r=r, pole=pole, angle=90.0, nphi=n,
                                     pole_is_vector=pole_is_vector, roll=roll,
                                     cache=cache, dtype=dtype)

    def to_local(self, pts_3d):
        raise NotImplementedError()


class SphericalPatch(SeedGen):
    """Make a rectangular (in theta and phi) patch on a sphere"""
    def __init__(self, p0=(0, 0, 0), p1=(0, 0, 1), max_alpha=45, max_beta=45,
                 nalpha=20, nbeta=20, roll=0.0, r=0.0, p1_is_vector=True,
                 cache=False, dtype=None):
        super(SphericalPatch, self).__init__(cache=cache, dtype=dtype)

        max_alpha = (np.pi / 180.0) * max_alpha
        max_beta = (np.pi / 180.0) * max_beta

        p0 = np.array(p0, copy=False, dtype=dtype).reshape(-1)
        p1 = np.array(p1, copy=False, dtype=dtype).reshape(-1)

        if not p1_is_vector:
            p1 = p1 - p0

        if not r:
            r = np.linalg.norm(p1)
        else:
            p1 = p1 * (r / np.linalg.norm(p1))

        if np.sin(max_alpha)**2 + np.sin(max_beta)**2 > 1:
            raise ValueError("Invalid alpha/beta ranges, if you want "
                             "something with more angular coverage you'll "
                             "need a SphericalCap")

        self.p0 = p0
        self.p1 = p1
        self.max_alpha = max_alpha
        self.max_beta = max_beta
        self.nalpha = nalpha
        self.nbeta = nbeta
        self.roll = roll
        self.r = r

    def get_nr_points(self, **kwargs):
        return self.nalpha * self.nbeta

    def get_local_shape(self, **kwargs):
        return (self.nalpha, self.nbeta)

    def to_3d(self, pts_local):
        arr = np.zeros((3, pts_local.shape[1]), dtype=self.dtype)
        arr[0, :] = self.r * np.sin(pts_local[0, :])
        arr[1, :] = self.r * np.sin(pts_local[1, :])
        arr[2, :] = np.sqrt(self.r**2 - arr[0, :]**2 - arr[1, :]**2)
        rot_pts = np.einsum("ij,jk->ik", self.get_rotation(), arr)
        return self.p0.reshape((3, 1)) + rot_pts

    def to_local(self, pts_3d):
        raise NotImplementedError()

    def _make_local_points(self):
        alpha, beta = self._make_local_axes()
        arr = np.zeros((2, self.nr_points), dtype=self.dtype)
        arr[0, :] = np.repeat(alpha, self.nbeta)
        arr[1, :] = np.tile(beta, self.nalpha)
        return arr

    def _make_local_axes(self):
        alpha = np.linspace(-1.0 * self.max_alpha, 1.0 * self.max_alpha,
                            self.nalpha)
        beta = np.linspace(-1.0 * self.max_beta, 1.0 * self.max_beta,
                           self.nbeta)
        return alpha, beta

    def get_rotation(self):
        return make_rotation_matrix([0, 0, 0], [0, 0, 1], self.p1,
                                    roll=self.roll)

    def as_coordinates(self):
        alpha, beta = self._make_local_axes()
        crds = viscid.wrap_crds("nonuniform_cartesian",
                                (('x', alpha), ('y', beta)))
        return crds

    def as_mesh(self):
        return self.get_points().reshape(3, self.nalpha, self.nbeta)

    def arr2mesh(self, arr):
        return arr.reshape(self.local_shape)


class PolarIonosphere(Sphere):
    def __init__(self, *args, **kwargs):
        super(PolarIonosphere, self).__init__(*args, **kwargs)
        raise NotImplementedError()

    def to_local(self, **kwargs):
        raise NotImplementedError()

##
## EOF
##
