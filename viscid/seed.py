#!/usr/bin/env python
"""Helpers to generate seed points for streamlines"""

from __future__ import print_function
import itertools

import numpy as np

import viscid
from viscid import NOT_SPECIFIED
from viscid.compat import izip

__all__ = ['to_seeds', 'SeedGen', 'Point',
           'MeshPoints', 'RectilinearMeshPoints', 'Line', 'Spline',
           'Plane', 'Volume', 'Sphere', 'SphericalCap', 'Circle',
           'SphericalPatch', 'PolarIonosphere']


def to_seeds(pts):
    """Try to turn anything into a set of seeds

    Args:
        points (ndarray or list): should look like something that
            np.array(points) can turn to an Nx3 array of xyz points.
            This can be 3xN so long as N != 3.
    """
    if (hasattr(pts, "nr_points") and hasattr(pts, "iter_points") and
        hasattr(pts, "points") and hasattr(pts, "wrap_field")):
        return pts
    else:
        return Point(pts)


class SeedGen(object):
    """All about seeds

    These objects are good for defining the root points of streamlines
    or locations for interpolation.

    - For Developers
        - Mandatory
            Subclasses **must** override

            - :py:meth:`get_nr_points`: get total number of seeds
            - :py:meth:`get_uv_shape`: get the shape of the ndarray
              of uv mesh coordinates IF there exists a 2D
              representation of the seeds
            - :py:meth:`get_local_shape`: get the shape of the ndarray
              of local coordinates
            - :py:meth:`uv_to_local`: transform 2d representation to
              local representation
            - :py:meth:`to_3d`: transform an array in local
              coordinates to 3D space. Should return a 3xN ndarray.
            - :py:meth:`to_local`: transform a 3xN ndarray to an
              ndarray of coordinates in local space. It's ok to raise
              NotImplementedError.
            - :py:meth:`_make_local_points`: Make an ndarray that is
              used in :py:meth:`_make_3d_points`
            - :py:meth:`_make_uv_axes`: Make axes for 2d mesh
              representation, i.e., matches :py:attr:`uv_shape`
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
            - :py:meth:`as_uv_coordinates`
            - :py:meth:`as_local_coordinates`
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

            - :py:attr:`uv_shape`
            - :py:attr:`local_shape`
            - :py:meth:`uv_to_local`
            - :py:meth:`to_local`
            - :py:meth:`as_mesh`
            - :py:meth:`as_uv_coordinates`
            - :py:meth:`as_local_coordinates`
            - :py:meth:`wrap_field`
    """
    _cache = None
    _points = None
    dtype = None
    fill_holes = None

    def __init__(self, cache=False, dtype=None, fill_holes=True):
        self._cache = cache
        self.dtype = dtype
        self.fill_holes = fill_holes

    @property
    def nr_points(self):
        return self.get_nr_points()

    @property
    def uv_shape(self):
        """Shape of uv representation of seeds"""
        return self.get_uv_shape()

    @property
    def local_shape(self):
        """Shape of local representation of seeds"""
        return self.get_local_shape()

    @property
    def uv_extent(self):
        """Try to get low and high coordnate values of uv points

        Raises:
            NotImplementedError: If subclass can't make uv mesh
        """
        crds = self.as_uv_coordinates()
        return np.array([crds.get_xl(), crds.get_xh()]).T

    @property
    def local_extent(self):
        """Try to get low and high coordnate values of local points

        Raises:
            NotImplementedError: If subclass can't make local mesh
        """
        crds = self.as_local_coordinates()
        return np.array([crds.get_xl(), crds.get_xh()]).T

    def get_nr_points(self, **kwargs):
        raise NotImplementedError()

    def get_uv_shape(self, **kwargs):
        raise RuntimeError("{0} does not define get_uv_shape"
                           "".format(type(self).__name__))

    def get_local_shape(self, **kwargs):
        raise NotImplementedError()

    def uv_to_local(self, pts_uv):
        raise RuntimeError("{0} does not define uv_to_local"
                           "".format(type(self).__name__))

    def uv_to_3d(self, pts_uv):
        return self.to_3d(self.uv_to_local(pts_uv))

    def local_to_3d(self, pts_local):
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

    def _make_uv_points(self):
        """Make a set of points in the uv representation"""
        raise NotImplementedError()

    def _make_local_points(self):
        """Make a set of points in the local representation"""
        raise NotImplementedError()

    def _make_uv_axes(self):
        """Make a tuple of arrays that match :py:attr:`local_shape`"""
        raise NotImplementedError()

    def _make_local_axes(self):
        """Make a tuple of arrays that match :py:attr:`local_shape`"""
        raise NotImplementedError()

    def _make_3d_points(self):
        """Make a 3xN ndarray of N xyz points"""
        return self.to_3d(self._make_local_points())

    def __array__(self, *args, **kwargs):
        return self.get_points()

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

    def as_uv_coordinates(self):
        """Make :py:class:`Coordinates` for the uv representation"""
        raise RuntimeError("{0} does not define as_uv_coordinates"
                           "".format(type(self).__name__))

    def as_local_coordinates(self):
        """Make :py:class:`Coordinates` for the local representation"""
        raise RuntimeError("{0} does not define as_local_coordinates"
                           "".format(type(self).__name__))

    def wrap_field(self, data, name="NoName", fldtype="scalar", **kwargs):
        """Wrap an ndarray into a field in the local representation"""
        crds = self.as_local_coordinates()
        return viscid.wrap_field(data, crds, name=name, fldtype=fldtype,
                                 **kwargs)

    def as_mesh(self, fill_holes=NOT_SPECIFIED):  # pylint: disable=unused-argument
        """Make a 3xUxV array that describes a 3D mesh surface"""
        raise RuntimeError("{0} does not define as_mesh"
                           "".format(type(self).__name__))

    def arr2mesh(self, arr, fill_holes=NOT_SPECIFIED):  # pylint: disable=unused-argument
        raise RuntimeError("{0} does not define arr2mesh"
                           "".format(type(self).__name__))

    def wrap_mesh(self, *arrs, **kwargs):
        fill_holes = kwargs.pop('fill_holes', NOT_SPECIFIED)
        assert not kwargs
        arrs = [a.data if isinstance(a, viscid.field.Field) else a for a in arrs]
        vertices = self.as_mesh(fill_holes=fill_holes)
        arrs = [self.arr2mesh(arr, fill_holes=fill_holes).astype(arr.dtype)
                for arr in arrs]
        return [vertices] + arrs


class Point(SeedGen):
    """Collection of points"""
    def __init__(self, pts, local_crds=None, cache=False, dtype=None, **kwargs):
        """Seed with an explicit set of points

        Args:
            pts (ndarray or list): should look like something that
                np.array(pts) can turn to an 3xN array of xyz
                points. This can be Nx3 as long as N != 3.
            local_crds (ndarray or list) : A list of customized local
                coordinates. Must of size N, where 3xN is the shape
                of pts. For example, user can provide an array of type
                datetime.datetime to represent the time of each point.
        """
        super(Point, self).__init__(cache=cache, dtype=dtype, **kwargs)
        self.pts = pts
        self.local_crds = local_crds

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
        if self.local_crds is None:
            return np.arange(self.nr_points)
        else:
            return self.local_crds

    def as_local_coordinates(self):
        x = self._make_local_axes()
        crds = viscid.wrap_crds("nonuniform_cartesian", (('x', x), ))
        return crds


class MeshPoints(SeedGen):
    """Generic points with 2d mesh information


    """
    def __init__(self, pts, cache=False, dtype=None, **kwargs):
        """Generic set of 2d points

        Args:
            pts (ndarray): must have shape 3xNUxNV for the uv mesh
                directions
        """
        super(MeshPoints, self).__init__(cache=cache, dtype=dtype, **kwargs)
        if pts.shape[0] != 3:
            raise ValueError("First index of pts must be xyz")
        self.pts = pts
        self.nu = pts.shape[1]
        self.nv = pts.shape[2]

    def get_nr_points(self, **kwargs):
        return self.nu * self.nv

    def get_uv_shape(self, **kwargs):
        return (self.nu, self.nv)

    def get_local_shape(self, **kwargs):
        return self.get_local_shape()

    def uv_to_local(self, pts_uv):
        return pts_uv

    def to_3d(self, pts_local):
        return pts_local.reshape(3, -1)

    def to_local(self, pts_3d):
        raise NotImplementedError()

    def _make_local_points(self):
        return self.pts

    def _make_uv_axes(self):
        return np.arange(self.nu), np.arange(self.nv)

    def _make_local_axes(self):
        return self._make_uv_axes()

    def as_uv_coordinates(self):
        l, m = self._make_uv_axes()
        crds = viscid.wrap_crds("nonuniform_cartesian", (('x', l), ('y', m)))
        return crds

    def as_local_coordinates(self):
        return self.as_uv_coordinates()

    def as_mesh(self, fill_holes=NOT_SPECIFIED):
        return self.get_points().reshape([3] + list(self.uv_shape))

    def arr2mesh(self, arr, fill_holes=NOT_SPECIFIED):
        return arr.reshape(self.uv_shape)


class RectilinearMeshPoints(MeshPoints):
    """Generic seeds with 2d rect mesh topology"""
    def __init__(self, pts, mesh_axes="xy", cache=False, dtype=None, **kwargs):
        """Generic seeds with 2d rect mesh topology

        Args:
            pts (ndarray): 3xNUxNV array
            mesh_axes (tuple, str): which directions do u and v
                correspond to.
        """
        _lookup = {0: 0, 1: 1, 'x': 0, 'y': 1}
        self.mesh_axes = [_lookup[ax] for ax in mesh_axes]
        super(RectilinearMeshPoints, self).__init__(pts, cache=cache,
                                                    dtype=dtype, **kwargs)

    def _make_uv_axes(self):
        u = self.pts[self.mesh_axes[0], :, 0]
        v = self.pts[self.mesh_axes[1], 0, :]
        return u, v


class Line(SeedGen):
    """A line of seed points"""
    def __init__(self, p0=(0, 0, 0), p1=(1, 0, 0), n=20,
                 cache=False, dtype=None, **kwargs):
        """p0 & p1 are `(x, y, z)` points as

        Args:
            p0 (list, tuple, or ndarray): starting point `(x, y, z)`
            p1 (list, tuple, or ndarray): ending point `(x, y, z)`
            n (int): number of points on the line
        """
        super(Line, self).__init__(cache=cache, dtype=dtype, **kwargs)
        self.p0 = np.asarray(p0, dtype=self.dtype)
        self.p1 = np.asarray(p1, dtype=self.dtype)
        self.n = n

    def get_nr_points(self, **kwargs):
        return self.n

    def get_uv_shape(self, **kwargs):
        return (self.nr_points, 1)

    def get_local_shape(self, **kwargs):
        return (self.nr_points, )

    def uv_to_local(self, pts_uv):
        return pts_uv[:, :1]

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

    def _make_uv_axes(self):
        return np.linspace(0.0, 1.0, self.n), np.array([0.0])

    def _make_local_axes(self):
        return np.linspace(0.0, 1.0, self.n)

    def as_uv_coordinates(self):
        raise NotImplementedError()

    def as_local_coordinates(self):
        dp = self.p1 - self.p0
        dist = np.sqrt(np.dot(dp, dp))
        x = np.linspace(0.0, dist, self.n)
        crd = viscid.wrap_crds("nonuniform_cartesian", (('x', x),))
        return crd

    def as_mesh(self, fill_holes=NOT_SPECIFIED):
        return self.get_points().reshape([3] + list(self.uv_shape))

class Spline(SeedGen):
    """A spline of seed points"""
    def __init__(self, knots, n=-5, splprep_kw=None, cache=False, dtype=None,
                 **kwargs):
        """
        Args:
            knots (sequence, ndarray): `[x, y, z]`, 3xN for N knots
            n (int): number of points on the curve, if negative, then
                `n = abs(n) * n_knots`
            **kwargs: Arguments to be used by scipy.interpolate.splprep
        """
        super(Spline, self).__init__(cache=cache, dtype=dtype, **kwargs)
        self.knots = np.asarray(knots, dtype=self.dtype)
        if self.knots.shape[0] < 3:
            raise ValueError("Knots should have shape 3xN for N knots")
            # ndims, npts = self.knots.shape
            # other = np.zeros((3 - ndims, npts), dtype=self.dtype)
            # self.knots = np.concatenate([self.knots, other], axis=0)
        if n < 0:
            n = -n * self.knots.shape[1]
        self.n = n
        self.splprep_opts = splprep_kw if splprep_kw else {}

    def get_nr_points(self, **kwargs):
        return self.n

    def get_uv_shape(self, **kwargs):
        return (self.nr_points, 1)

    def get_local_shape(self, **kwargs):
        return (self.nr_points, )

    def uv_to_local(self, pts_uv):
        return pts_uv[:, :1]

    def to_3d(self, pts_local, **kwargs):
        import scipy.interpolate as interpolate
        k = self.splprep_opts.pop("k", self.knots.shape[1] - 1)
        tck, u = interpolate.splprep(self.knots, k=k, **self.splprep_opts)
        u = np.linspace(0, 1, self.n + 1, endpoint=True)
        coords = np.vstack(interpolate.splev(u, tck))
        coords = coords[:, (pts_local * self.n).astype(int)].astype(self.dtype)
        return coords

    def to_local(self, pts_3d):
        raise NotImplementedError()

    def _make_local_points(self):
        return self._make_local_axes()

    def _make_uv_axes(self):
        return np.linspace(0.0, 1.0, self.n), np.array([0.0])

    def _make_local_axes(self):
        return np.linspace(0.0, 1.0, self.n)

    def as_uv_coordinates(self):
        raise NotImplementedError()

    def as_local_coordinates(self):
        pts_local = np.linspace(0.0, 1.0, self.n, endpoint=True)
        x = self.to_3d(pts_local)
        dx = x[:, 1:] - x[:, :-1]
        ds = np.zeros_like(x[0])
        ds[1:] = np.linalg.norm(dx, axis=0)
        s = np.cumsum(ds)
        crd = viscid.wrap_crds("nonuniform_cartesian", (('s', s),))
        return crd

    def as_mesh(self, fill_holes=NOT_SPECIFIED):
        return self.get_points().reshape([3] + list(self.uv_shape))

class Plane(SeedGen):
    """A plane of seed points"""

    def __init__(self, p0=(0, 0, 0), pN=(0, 0, 1), pL=(1, 0, 0),
                 len_l=2, len_m=2, nl=20, nm=20, NL_are_vectors=True,
                 cache=False, dtype=None, **kwargs):
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

        super(Plane, self).__init__(cache=cache, dtype=dtype, **kwargs)

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

    def get_uv_shape(self):
        return (self.nl, self.nm)

    def get_local_shape(self):
        return (3, self.nl * self.nm)

    def uv_to_local(self, pts_uv):
        return np.concatenate([pts_uv, np.zeros((1, pts_uv.shape[1]))], axis=0)

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
        l, m, n = self._make_local_axes()
        plane_lmn[0, :] = np.repeat(l, self.nm)
        plane_lmn[1, :] = np.tile(m, self.nl)
        plane_lmn[2, :] = n
        return plane_lmn

    def _make_uv_axes(self):
        len_l, len_m = self.len_l, self.len_m
        l = np.linspace(len_l[0], len_l[1], self.nl).astype(self.dtype)
        m = np.linspace(len_m[0], len_m[1], self.nm).astype(self.dtype)
        return l, m

    def _make_local_axes(self):
        l, m = self._make_uv_axes()
        n = np.array([0.0]).astype(self.dtype)
        return l, m, n

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
            >>> from viscid.plot import vlab
            >>> vlab.mesh_from_seeds(plane)
            >>> vlab.points3d(xyz[0], xyz[1], xyz[2])
            >>> vlab.show()

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
            >>> from viscid.plot import vlab
            >>> verts, s = plane.wrap_mesh(Blmn['z'].data)
            >>> vlab.mesh(verts[0], verts[1], verts[2], scalars=s)
            >>> verts, vx, vy, vz = plane.wrap_mesh(B_interp['x'].data,
            >>>                                      B_interp['y'].data,
            >>>                                      B_interp['z'].data)
            >>> vlab.quiver3d(verts[0], verts[1], verts[2], vx, vy, vz)
            >>> vlab.show()
            >>>
            >>> # use lmn to show in-plane / out-of-plane
            >>> from viscid.plot import vpyplot as vlt
            >>> from matplotlib import pyplot as plt
            >>> vlt.plot(Blmn['z'])  # z means n here
            >>> vlt.plot2d_quiver(Blmn)
            >>> plt.show()
        """
        return np.array([self.Ldir, self.Mdir, self.Ndir]).T

    def as_uv_coordinates(self):
        l, m = self._make_uv_axes()
        crds = viscid.wrap_crds("nonuniform_cartesian", (('x', l), ('y', m)))
        return crds

    def as_local_coordinates(self):
        l, m, n = self._make_local_axes()
        crds = viscid.wrap_crds("nonuniform_cartesian", (('x', l), ('y', m),
                                                         ('z', n)))
        return crds

    def as_mesh(self, fill_holes=NOT_SPECIFIED):
        return self.get_points().reshape([3] + list(self.uv_shape))

    def arr2mesh(self, arr, fill_holes=NOT_SPECIFIED):
        return arr.reshape(self.uv_shape)


class Volume(SeedGen):
    """A volume of seed points

    Defined by two opposite corners of a box in 3D
    """
    def __init__(self, xl=(-1, -1, -1), xh=(1, 1, 1), n=(20, 20, 20),
                 cache=False, dtype=None, **kwargs):
        """Make a volume

        Args:
            xl (list, tuple, or ndarray): Lower corner as (x, y, z)
            xh (list, tuple, or ndarray): Upper corner as (x, y, z)
            n (list, tuple, or ndarray): number of points (nx, ny, nz)
                defaults to (20, 20, 20)
        """
        super(Volume, self).__init__(cache=cache, dtype=dtype, **kwargs)

        self.xl = np.asarray(xl, dtype=self.dtype)
        self.xh = np.asarray(xh, dtype=self.dtype)
        self.n = np.empty_like(self.xl, dtype='i')
        self.n[:] = n

    def _get_uv_xind(self):
        try:
            return self.n.tolist().index(1)
        except ValueError:
            raise RuntimeError("Volume has no length 1 dimension, can't map "
                               "to uv space")

    def get_nr_points(self, **kwargs):
        return np.prod(self.n)

    def get_uv_shape(self, **kwargs):
        n = self.n.tolist()
        n.pop(self._get_uv_xind())
        return n

    def get_local_shape(self, **kwargs):
        return tuple(self.n)

    def uv_to_local(self, pts_uv):
        ind = self._get_uv_xind()
        val = self.xl[ind]
        return np.insert(pts_uv, ind, val * np.ones(pts_uv.shape[1]), axis=0)

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

    def _make_uv_axes(self):
        axes = list(self._make_local_axes())
        axes.pop(self._get_uv_xind())
        return axes

    def _make_local_axes(self):
        x = np.linspace(self.xl[0], self.xh[0], self.n[0]).astype(self.dtype)
        y = np.linspace(self.xl[1], self.xh[1], self.n[1]).astype(self.dtype)
        z = np.linspace(self.xl[2], self.xh[2], self.n[2]).astype(self.dtype)
        return x, y, z

    def iter_points(self, **kwargs):
        x, y, z = self._make_local_axes()
        return itertools.product(x, y, z)

    def as_uv_coordinates(self):
        x, y = self._make_uv_axes()
        crd = viscid.wrap_crds("nonuniform_cartesian",
                               (('x', x), ('y', y)))
        return crd

    def as_local_coordinates(self):
        x, y, z = self._make_local_axes()
        crd = viscid.wrap_crds("nonuniform_cartesian",
                               (('x', x), ('y', y), ('z', z)))
        return crd

    def as_mesh(self, fill_holes=NOT_SPECIFIED):
        return self.get_points().reshape([3] + list(self.uv_shape))

    def arr2mesh(self, arr, fill_holes=NOT_SPECIFIED):
        return arr.reshape(self.uv_shape)


class Sphere(SeedGen):
    """Make seeds on the surface of a sphere"""

    def __init__(self, p0=(0, 0, 0), r=0.0, pole=(0, 0, 1), ntheta=20, nphi=20,
                 thetalim=(0, 180.0), philim=(0, 360.0), roll=0.0, crd_system=None,
                 theta_endpoint='auto', phi_endpoint='auto', pole_is_vector=True,
                 theta_phi=False, cache=False, dtype=None, **kwargs):
        """Make seeds on the surface of a sphere

        Note:
            There is some funny business about the meaning of phi=0 and
            `crd_system`. By default, this seed generator is agnostic
            to coordinate systems and phi=0 always means the +x axis.
            If crd_system is 'gse', 'mhd', or an object whose find_info
            method returns a 'crd_system', then phi=0 means midnight.
            This is important when specifying a phi range or plotting
            on a matplotlib polar plot.

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
            roll (float): Roll the seeds around the pole by this angle
                in deg
            crd_system (str, Field): a crd system ('gse', 'mhd') or an
                object that has 'crd_system' info such that phi=0
                means midnight instead of the +x axis.
            theta_endpoint (str): this is a bit of a hack to keep from
                having redundant seeds at poles. You probably just want
                auto here
            phi_endpoint (bool): if true, then let phi inclue upper
                value. This is false by default since 0 == 2pi.
            pole_is_vector (bool): Whether pole is a vector or a
                vector
            theta_phi (bool): If True, the uv and local representations
                are ordered (theta, phi), otherwise (phi, theta)

        Raises:
            ValueError: if thetalim or philim don't have 2 values each
        """
        super(Sphere, self).__init__(cache=cache, dtype=dtype, **kwargs)

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

        if not len(thetalim) == len(philim) == 2:
            raise ValueError("thetalim and philim must have both min and max")

        try:
            roll = float(roll)
        except TypeError:
            pass

        # square away crd system
        if crd_system:
            if hasattr(crd_system, 'find_info'):
                crd_system = viscid.as_crd_system(crd_system, 'none')
        else:
            crd_system = 'none'
        if crd_system.strip().lower() == 'gse':
            crd_system_roll = -180.0
        else:
            crd_system_roll = 0.0


        self.r = r
        self.ntheta = ntheta
        self.nphi = nphi
        self.thetalim = np.deg2rad(thetalim)
        self.philim = np.deg2rad(philim)
        self.theta_endpoint = theta_endpoint
        self.phi_endpoint = phi_endpoint
        self.theta_phi = theta_phi
        self.roll = roll
        self.crd_system_roll = crd_system_roll

    @property
    def _spans_theta(self):
        return np.isclose(self.thetalim[1] - self.thetalim[0], np.pi)

    @property
    def _spans_phi(self):
        return np.isclose(self.philim[1] - self.philim[0], 2 * np.pi)

    @property
    def _is_whole_sphere(self):
        return self._spans_theta and self._spans_phi

    @property
    def pt_bnds(self):
        which = "0" if self.theta_phi else "1"
        pt = []
        if np.any(np.isclose(self.thetalim[0], [0.0, np.pi])):
            pt.append(which + "-")
        if np.any(np.isclose(self.thetalim[1], [0.0, np.pi])):
            pt.append(which + "+")
        return pt

    @property
    def periodic(self):
        if self._spans_phi:
            if self.theta_phi:
                return (False, "1")
            else:
                return ("1", False)
        else:
            return ()

    def _resolve_theta_endpoints(self):
        if self.theta_endpoint == 'auto':
            theta_endpoints = [True, True]
            for i in range(2):
                if self._includes_pole(i):
                    theta_endpoints[i] = False
        else:
            if isinstance(self.theta_endpoint, (list, tuple)):
                theta_endpoints = self.theta_endpoint  # pylint: disable=redefined-variable-type
            else:
                theta_endpoints = [self.theta_endpoint] * 2
        return theta_endpoints

    def _resolve_phi_endpoint(self):
        phi_endpoint = self.phi_endpoint
        if phi_endpoint == 'auto':
            phi_endpoint = not self._spans_phi
        return phi_endpoint

    def _includes_pole(self, whichlim):
        return np.any(np.isclose(self.thetalim[whichlim], [0, np.pi]))

    def _skipped_poles(self):
        # True if a pole is included in thetalim, but excluded by the linspace
        # endpoint
        inc_poles = [self._includes_pole(i) for i in range(2)]
        endpts = self._resolve_theta_endpoints()
        return np.bitwise_and(inc_poles, np.invert(endpts)).tolist()

    def get_nr_points(self, **kwargs):
        return self.ntheta * self.nphi

    def get_uv_shape(self, **kwargs):
        if self.theta_phi:
            return (self.ntheta, self.nphi)
        else:
            return (self.nphi, self.ntheta)

    def get_local_shape(self, **kwargs):
        return tuple([1] + list(self.uv_shape))

    def uv_to_local(self, pts_uv):
        return np.insert(pts_uv, 0, self.r * np.ones((1, pts_uv.shape[1])),
                         axis=0)

    def to_3d(self, pts_local):
        if self.theta_phi:
            r, T, P = pts_local[0, :], pts_local[1, :], pts_local[2, :]
        else:
            r, P, T = pts_local[0, :], pts_local[1, :], pts_local[2, :]
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
        r, theta, phi = self._make_local_axes()
        arr[0, :] = r
        # Note: arr[1, :] is always theta and arr[2, :] is always phi
        # what changes is the way the points are ordered in the 2nd dimension
        if self.theta_phi:
            arr[1, :] = np.repeat(theta, self.nphi)
            arr[2, :] = np.tile(phi, self.ntheta)
        else:
            arr[1, :] = np.repeat(phi, self.ntheta)
            arr[2, :] = np.tile(theta, self.nphi)
        return arr

    def _make_uv_axes(self):
        # well this decision tree is a mess, sorry about that
        theta_endpoints = self._resolve_theta_endpoints()
        if all(theta_endpoints):
            theta = np.linspace(self.thetalim[0], self.thetalim[1],
                                self.ntheta).astype(self.dtype)
        elif any(theta_endpoints):
            if theta_endpoints[0]:
                theta = np.linspace(self.thetalim[0], self.thetalim[1],
                                    self.ntheta, endpoint=False).astype(self.dtype)
            else:
                theta = np.linspace(self.thetalim[1], self.thetalim[0],
                                    self.ntheta, endpoint=False).astype(self.dtype)
                theta = theta[::-1]
        else:
            theta = np.linspace(self.thetalim[0], self.thetalim[1],
                                self.ntheta + 1).astype(self.dtype)
            theta = 0.5 * (theta[1:] + theta[:-1])

        phi_endpoint = self._resolve_phi_endpoint()
        phi = np.linspace(self.philim[0], self.philim[1], self.nphi,
                          endpoint=phi_endpoint).astype(self.dtype)
        return theta, phi

    def _make_local_axes(self):
        theta, phi = self._make_uv_axes()
        r = np.array([self.r], dtype=self.dtype)
        return r, theta, phi

    def get_rotation(self):
        return viscid.a2b_rot([0, 0, 1], self.pole,
                              roll=self.roll + self.crd_system_roll, unit='deg')

    def as_uv_coordinates(self):
        theta, phi = self._make_uv_axes()
        if self.theta_phi:
            crds = viscid.wrap_crds("nonuniform_spherical",
                                    (('theta', theta), ('phi', phi)),
                                    units="rad")
        else:
            crds = viscid.wrap_crds("nonuniform_spherical",
                                    (('phi', phi), ('theta', theta)),
                                    units="rad")
        return crds

    def as_local_coordinates(self):
        return self.as_uv_coordinates()

    def as_mesh(self, fill_holes=NOT_SPECIFIED):
        new_shape = [3] + list(self.uv_shape)
        pts = self.get_points().reshape(new_shape)

        if fill_holes is NOT_SPECIFIED:
            fill_holes = self.fill_holes
        if not fill_holes:
            return pts

        # If the 'top' pole is in thetalim but not included in pts, then
        # extend the pts to include the pole so there aren't holes in the pole
        skipped_poles = self._skipped_poles()
        sgn = [+1, -1]
        # cat_ax = [1, 2]

        for i in range(2):
            if skipped_poles[i]:
                if self.theta_phi:
                    polei = np.empty((3, 1, self.nphi), dtype=pts.dtype)
                else:
                    polei = np.empty((3, self.nphi, 1), dtype=pts.dtype)

                polei[...] = (self.p0 + sgn[i] * (self.r * self.pole)).reshape(3, 1, 1)

                cat_lst = [pts]
                cat_lst.insert(i, polei)
                if self.theta_phi:
                    pts = np.concatenate(cat_lst, axis=1)
                else:
                    pts = np.concatenate(cat_lst, axis=2)

        # close up the seam at phi = 0 / phi = 2 pi
        if self._spans_phi and not self._resolve_phi_endpoint():
            if self.theta_phi:
                pts = np.concatenate([pts, pts[:, :, 0, None]], axis=2)
            else:
                pts = np.concatenate([pts, pts[:, 0, None, :]], axis=1)

        if not self.theta_phi:
            pts = pts[:, ::-1, :]  # normals out

        return pts

    def arr2mesh(self, arr, fill_holes=NOT_SPECIFIED):
        arr = arr.reshape(self.uv_shape)

        if fill_holes is NOT_SPECIFIED:
            fill_holes = self.fill_holes
        if not fill_holes:
            return arr

        # average values around a pole for the new mesh vertex @ the pole so
        # that there's no gap in the mesh
        skipped_poles = self._skipped_poles()
        rpt_idx = [0, -1]
        for i in range(2):
            if skipped_poles[i]:
                if self.theta_phi:
                    p = np.repeat(np.mean(arr[rpt_idx[i], :, None], keepdims=1),
                                  arr.shape[1], axis=1)
                    cat_lst = [arr]
                    cat_lst.insert(i, p)
                    arr = np.concatenate(cat_lst, axis=0)
                else:
                    p = np.repeat(np.mean(arr[:, rpt_idx[i], None], keepdims=1),
                                  arr.shape[0], axis=0)
                    cat_lst = [arr]
                    cat_lst.insert(i, p)
                    arr = np.concatenate(cat_lst, axis=1)

        # repeat last phi to close the seam at phi = 0 / 2 pi
        if self._spans_phi and not self._resolve_phi_endpoint():
            if self.theta_phi:
                arr = np.concatenate([arr, arr[:, 0, None]], axis=1)
            else:
                arr = np.concatenate([arr, arr[0, None, :]], axis=0)

        if not self.theta_phi:
            arr = arr[::-1, :]  # normals out

        return arr


class SphericalCap(Sphere):
    """A spherical cone or cap of seeds

    Defined by a center, and a point indicating the direction of the
    cone, and the half angle of the cone.

    This is mostly a wrapper for :py:class:`Sphere` that sets
    thetalim, but it also does some special stuff when wrapping
    meshes to remove the last theta value so that the cap isn't
    closed at the bottom.
    """
    def __init__(self, angle0=0.0, angle=90.0, theta_endpoint=[True, False],
                 **kwargs):
        """Make a spherical cap with an optional hole in the middle

        Args:
            angle0 (float): starting angle from pole, useful for making
                a hole in the center of the cap
            angle (float): cone angle of the cap in degrees
            **kwargs: passed to :py:class:`Sphere` constructor
        """
        super(SphericalCap, self).__init__(thetalim=(angle0, angle), **kwargs)

    @property
    def angle0(self):
        return self.thetalim[0]

    @property
    def angle(self):
        return self.thetalim[1]

    def _resolve_theta_endpoints(self):
        if self.theta_endpoint == 'auto':
            ret = [True, False]
        else:
            ret = super(SphericalCap, self)._resolve_theta_endpoints()
        return ret

    def to_local(self, pts_3d):
        raise NotImplementedError()

class Circle(SphericalCap):
    """A circle of seeds

    Defined by a center and a point normal to the plane of the circle
    """
    def __init__(self, n=20, endpoint=None, **kwargs):
        """Circle of seed points

        Args:
            **kwargs: passed to :py:class:`Sphere` constructor

        Note:
            The pole keyword argument is the direction *NORMAL* to the
            circle.
        """
        if endpoint is not None:
            kwargs['phi_endpoint'] = endpoint
        super(Circle, self).__init__(angle0=90.0, angle=90.0, ntheta=1, nphi=n,
                                     **kwargs)

    @property
    def n(self):
        return self.nphi

    @property
    def endpoint(self):
        return self.phi_endpoint

    def to_local(self, pts_3d):
        raise NotImplementedError()


class SphericalPatch(SeedGen):
    """Make a rectangular (in theta and phi) patch on a sphere"""
    def __init__(self, p0=(0, 0, 0), p1=(0, 0, 1), max_alpha=45, max_beta=45,
                 nalpha=20, nbeta=20, roll=0.0, r=0.0, p1_is_vector=True,
                 cache=False, dtype=None, **kwargs):
        super(SphericalPatch, self).__init__(cache=cache, dtype=dtype, **kwargs)

        max_alpha = (np.pi / 180.0) * max_alpha
        max_beta = (np.pi / 180.0) * max_beta

        p0 = np.array(p0, copy=False, dtype=dtype).reshape(-1)
        p1 = np.array(p1, copy=False, dtype=dtype).reshape(-1)

        if not p1_is_vector:
            p1 = p1 - p0

        if r:
            p1 = p1 * (r / np.linalg.norm(p1))
        else:
            r = np.linalg.norm(p1)

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

    def get_uv_shape(self, **kwargs):
        return (self.nalpha, self.nbeta)

    def get_local_shape(self, **kwargs):
        return self.uv_shape

    def uv_to_local(self, pts_uv):
        return pts_uv

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

    def _make_uv_axes(self):
        alpha = np.linspace(-1.0 * self.max_alpha, 1.0 * self.max_alpha,
                            self.nalpha)
        beta = np.linspace(-1.0 * self.max_beta, 1.0 * self.max_beta,
                           self.nbeta)
        return alpha, beta

    def _make_local_axes(self):
        return self._make_uv_axes()

    def get_rotation(self):
        return viscid.a2b_rot([0, 0, 1], self.p1, roll=self.roll, unit='deg')

    def as_uv_coordinates(self):
        alpha, beta = self._make_uv_axes()
        crds = viscid.wrap_crds("nonuniform_cartesian",
                                (('x', alpha), ('y', beta)))
        return crds

    def as_local_coordinates(self):
        return self.as_uv_coordinates()

    def as_mesh(self, fill_holes=NOT_SPECIFIED):
        return self.get_points().reshape(3, self.nalpha, self.nbeta)

    def arr2mesh(self, arr, fill_holes=NOT_SPECIFIED):
        return arr.reshape(self.uv_shape)


class PolarIonosphere(Sphere):
    """Place holder for future seed to cover N+S poles"""
    def __init__(self, *args, **kwargs):
        super(PolarIonosphere, self).__init__(*args, **kwargs)
        raise NotImplementedError()

    def to_local(self, **kwargs):
        raise NotImplementedError()

##
## EOF
##
