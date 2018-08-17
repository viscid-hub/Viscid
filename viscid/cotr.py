#!/usr/bin/env python
"""Numpy implementation of space physics coordinate transformations

The abbreviations and equations in this module follow [Hapgood1992]_,
    - gei: Geocentric equatorial inertial
        - X = First Point of Aries
        - Z = Geographic North Pole
    - geo: Geographic
        - X = Intersection of Greenwich meridian and geographic equator
        - Z = Geographic North Pole
    - gse: Geocentric solar ecliptic
        - X = Earth-Sun line (points toward Sun)
        - Z = Ecliptic North Pole
    - gsm: Geocentric solar magnetospheric
        - X = Earth-Sun line (points toward Sun)
        - Z = Projection of dipole axis on GSE Y-Z plane
    - sm: Solar magnetic
        - Y = Perpendicular to plane containing Earth-Sun line and
          dipole axis. Positive is opposite to Earth's orbital motion
        - Z = Earth's dipole axis
    - mag: Geomagnetic
        - Y = Intersection between geographic equator and the
          geographic meridian 90deg East of the meditian containing
          the dipole axis
        - Z = Earth's dipole axis
    - mhd: OpenGGCM's native coordinate system (based on GSE)
        - X = Sun-Earth line (points AWAY from Sun)
        - Z = Ecliptic North Pole
    - hae: Heliocentric Aries ecliptic
        - X = First Point of Aries
        - Z = Ecliptic North Pole
    - hee: Heliocentric Earth ecliptic
        - X = Sun-Earth line
        - Z = Ecliptic North Pole
    - heeq: Heliocentric Earth equatorial
        - X = Intersection between solar equator and solar central
          meridian as seen from Earth
        - Z = North Pole of solar rotation

Dipole info between 2010-01-01 00:00 and 2010-06-20 23:59,
    - Smallest dipole tilt angle: 0.00639 deg @ 2010-04-16 05:10
    - Largest dipole tilt angle: 33.5 deg @ 2010-06-19 16:59
    - Smallest angle between dip and GSE-Z: 13.3 deg @ 2010-01-03 15:54
    - Largest angle between dip and GSE-Z: 33.6 deg @ 2010-01-13 03:19

Dipole info between 1952-01-01 00:00 and 1955-12-31 23:59,
    - Smallest dipole tilt angle: -0.00145 deg @ 1952-09-01 01:35
    - Largest dipole tilt angle: 35.2 deg @ 1954-06-21 16:38
    - Smallest angle between dip and GSE-Z: 11.7 deg @ 1954-12-14 17:05
    - Largest angle between dip and GSE-Z: 35.2 deg @ 1955-01-02 03:53

Note:
    IGRF coefficients are taken from [IGRF]_, and coefficients are
    assumed constant outside of the given range. This means every 5
    years, the new coefficients should be added manually.

References:
    .. [IGRF] <http://wdc.kugi.kyoto-u.ac.jp/igrf/coef/igrf12coeffs.txt>
    .. [Hapgood1992] Hapgood, M. A., 2011, "Space Physics Coordinate
       Transformations: A User Guide", Planet. Space Sci. Vol 40,
       No. 5. pp. 711-717, 1992.
       <http://www.igpp.ucla.edu/public/vassilis/ESS261/Lecture03/Hapgood_sdarticle.pdf>

This module only depends on npdatetime, which is itself orthogonal to
Viscid, so these two modules can be ripped out and used more generally.
Please note that Viscid is MIT licensed, which requires attribution.

The MIT License (MIT)
Copyright (c) 2017 Kristofor Maynard

"""

# pylint: disable=bad-whitespace

from __future__ import print_function, division
import sys

import numpy as np

try:
    from viscid.npdatetime import (as_datetime64, as_datetime, as_timedelta,  # pylint: disable=import-error
                                   linspace_datetime64, datetime64_as_years,
                                   format_datetime, is_datetime_like)
except ImportError:
    from npdatetime import (as_datetime64, as_datetime, as_timedelta,  # pylint: disable=import-error
                            linspace_datetime64, datetime64_as_years,
                            format_datetime, is_datetime_like)


__all__ = ['as_nvec', 'make_rotation', 'make_translation', 'get_rotation_wxyz',
           'as_crd_system', 'Cotr', 'as_cotr', 'cotr_transform',
           'get_dipole_moment', 'get_dipole_moment_ang', 'get_dipole_angles',
           'dipole_moment2cotr']


# note that these globals are used immutably (ie, not rc file configurable)
DEFAULT_STRENGTH = 1.0 / 3.0574e-5  # Strength of earth's dipole in nT


class _NOT_GIVEN(object):
    pass


# IGRF values are in nT and come from:
# http://wdc.kugi.kyoto-u.ac.jp/igrf/coef/igrf12coeffs.txt
# NOTE: This will need to be updated periodically as dates more recent
#       than the latest coefficient use 0th order extrapolation
#       (ie, the last coefficients are repeated until the end of time)
#                             g10      g11       h11
#                 year    g:n=1,m=0  g:n=1,m=1 h:n=1,m=1
_IGRF = np.array([[1950.0, -30554.00, -2250.00, 5815.00],
                  [1955.0, -30500.00, -2215.00, 5820.00],
                  [1960.0, -30421.00, -2169.00, 5791.00],
                  [1965.0, -30334.00, -2119.00, 5776.00],
                  [1970.0, -30220.00, -2068.00, 5737.00],
                  [1975.0, -30100.00, -2013.00, 5675.00],
                  [1980.0, -29992.00, -1956.00, 5604.00],
                  [1985.0, -29873.00, -1905.00, 5500.00],
                  [1990.0, -29775.00, -1848.00, 5406.00],
                  [1995.0, -29692.00, -1784.00, 5306.00],
                  [2000.0, -29619.40, -1728.20, 5186.10],
                  [2005.0, -29554.63, -1669.05, 5077.99],
                  [2010.0, -29496.57, -1586.42, 4944.26],
                  [2015.0, -29442.00, -1501.00, 4797.10]], dtype='f8')


def _igrf_interpolate(time, notilt1967=True):
    # get time as decimal year, note that epoch is completely arbitrary
    years = datetime64_as_years(time)

    if notilt1967 and np.abs(years - 1967.0) < 0.0006:
        # OpenGGCM special no-tilt hack
        # g10, g11, h11 = np.linalg.norm(IGRF[0, 1:]), 0.0, 0.0
        g10, g11, h11 = -30554.00, 0.0, 0.0
    else:
        g10 = np.interp(years, _IGRF[:, 0], _IGRF[:, 1])
        g11 = np.interp(years, _IGRF[:, 0], _IGRF[:, 2])
        h11 = np.interp(years, _IGRF[:, 0], _IGRF[:, 3])

    return g10, g11, h11

def as_nvec(arr, ndim=4, init_vals=(0, 0, 0, 1)):
    """Extend vectors to `ndim` dimensions

    The first dimension or arr that is the one that is extended. If the
    last dimension has shape >= 4, leave arr unchanged

    Args:
        arr (sequence): array with shape (3,) or (3, N) for N vectors
        ndim (int): extend arr to this many dimensions
        init_vals (sequence): initialized values for new dimensions,
            note that init_vals[i] is the initialization for the ith
            dimension of the result. (0, 0, 0, 1) is useful for 4x4
            rotation+translation matrices

    Returns:
        ndarray: (ndim,) or (ndim, N) depending on input
    """
    arr = np.asarray(arr)
    if arr.shape[0] < ndim:
        dim1234 = np.empty([ndim - arr.shape[0]] + list(arr.shape)[1:],
                           dtype=arr.dtype)
        # dim4 = np.empty(list(arr.shape)[:-1] + [1], dtype=arr.dtype)
        if len(init_vals) < ndim:
            raise ValueError("init vals should have length at least ndim")
        for i, j in enumerate(range(arr.shape[0], ndim)):
            dim1234[i, ...] = init_vals[j]
        arr = np.concatenate([arr, dim1234], axis=0)
    return arr

def make_rotation(theta, axis='z', rdim=3):
    """Make rotation matrix of theta degrees around axis

    Warning:
        The Hapgood paper uses the convention that rotations around y
        are right-handed, but rotations around x and z are left handed!

    Args:
        theta (float): angle (degrees)
        axis (int, str): one of (0, 'x', 1, 'y', 2, 'z')
        rdim (int): 3 to make 3x3 matrices, or 4 to make 4x4 where
            `mat[:, 3] == mat[3, :] == [0, 0, 0, 1]`. 4x4 matrices are
            useful for performing translations.

    Raises:
        ValueError: On invalid axis
    """
    try:
        axis = axis.lower().strip()
    except AttributeError:
        pass

    if rdim not in (2, 3, 4):
        raise ValueError("2x2, 3x3 or 4x4 matrices only")

    theta_rad = np.deg2rad(theta)
    sinT = np.sin(theta_rad)
    cosT = np.cos(theta_rad)

    if theta == 0.0:
        mat = np.eye(rdim)
    elif axis in ('x', 0):
        mat = np.array([[  1.0,   0.0,   0.0, 0.0],
                        [  0.0,  cosT,  sinT, 0.0],
                        [  0.0, -sinT,  cosT, 0.0],
                        [  0.0,   0.0,   0.0, 1.0]])
    elif axis in ('y', 1):
        mat = np.array([[ cosT,   0.0,  sinT, 0.0],
                        [  0.0,   1.0,   0.0, 0.0],
                        [-sinT,   0.0,  cosT, 0.0],
                        [  0.0,   0.0,   0.0, 1.0]])
    elif axis in ('z', 2):
        mat = np.array([[ cosT,  sinT,   0.0, 0.0],
                        [-sinT,  cosT,   0.0, 0.0],
                        [  0.0,   0.0,   1.0, 0.0],
                        [  0.0,   0.0,   0.0, 1.0]])
    else:
        raise ValueError("invalid axis: {0}".format(axis))

    return np.array(mat[:rdim, :rdim])

def make_translation(v):
    """Make a 4x4 translation matrix to act on [x, y, z, 1] vectors

    Args:
        v (sequence): Translation vector, must have at least 3 entries

    Returns:
        (mat, imat): mat will translate an [x, y, z, 1] vector, and
            imat will do the inverse translation
    """
    tmat = np.array([[1, 0, 0, +v[0]],
                     [0, 1, 0, +v[1]],
                     [0, 0, 1, +v[2]],
                     [0, 0, 0,     1]])
    imat = np.array([[1, 0, 0, -v[0]],
                     [0, 1, 0, -v[1]],
                     [0, 0, 1, -v[2]],
                     [0, 0, 0,     1]])
    return tmat, imat

def get_rotation_wxyz(mat, check_inverse=True, quick=True):
    """Find rotation axis and angle of a rotation matrix

    Args:
        mat (ndarray): 3x3 rotation matrix with determinant +1
        check_inverse (bool): check if mat.T == inverse(mat)
        quick (bool): Try a quicker, less well tested method first

    Returns:
        ndarray: [w (rotation angle), ux, uy, uz] where u is the
            normalized axis vector
    """
    if check_inverse and not np.allclose(mat.T, np.linalg.inv(mat)):
        raise ValueError("Matrix transpose != inverse, not a rotation")

    if quick:
        # I'm not sure if I trust this method, although it is probably
        # quicker than calculating eigenvectors
        ux = mat[2, 1] - mat[1, 2]
        uy = mat[0, 2] - mat[2, 0]
        uz = mat[1, 0] - mat[0, 1]
        u = np.array([ux, uy, uz])
    else:
        u = np.array([0.0, 0.0, 0.0])

    if np.isclose(np.linalg.norm(u), 0.0):
        eigval, eigvecs = np.linalg.eig(mat)
        i = np.argmin(np.abs(eigval - 1.0))
        u = eigvecs[:, i].real  # eigenvector for eigenval closest to 1.0

    # normalize axis vector
    u = u / np.linalg.norm(u)

    # find a vector `b1` that is perpendicular to rotation axis `u`
    a = np.array([1, 0, 0], dtype=u.dtype)
    au_diff = np.linalg.norm(a - u)
    if au_diff < 0.1 or au_diff > 1.9:
        a = np.array([0, 1, 0], dtype=u.dtype)
    b1 = np.cross(a, u)
    b1 = b1 / np.linalg.norm(b1)  # minimize effect of roundoff error

    # rotate `b1` and then find the angle between result (`b2`) at `b1`
    b2 = np.dot(mat, b1)
    b2 = b2 / np.linalg.norm(b2)  # minimize effect of roundoff error
    angleB = np.rad2deg(np.arccos(np.dot(b1, b2)))

    # fix sign of `u` to make the rotation angle right handed, note that
    # the angle will always be positive due to range or np.arccos
    if np.dot(np.cross(b1, b2), u) < 0.0:
        u *= -1

    return np.array([angleB, u[0], u[1], u[2]])

def as_crd_system(thing, default=_NOT_GIVEN):
    """Try to figure out a crd_system for thing

    Args:
        thing: Either a string with a crd_system abbreviation, or
            something that that can do thing.find_info('crd_system')
        default (str): fallback value

    Returns:
        str: crd system abbreviation, stripped and lower case

    Raises:
        TypeError: if no default supplied and thing can't be turned
            into a crd_system
    """
    if hasattr(thing, "__crd_system__"):
        crd_system = thing.__crd_system__
    else:
        try:
            crd_system = thing.find_attr("__crd_system__")
        except AttributeError:
            if hasattr(thing, "find_info") and thing.find_info("crd_system", None):
                crd_system = thing.find_info("crd_system")
            else:
                crd_system = thing

    # post-process crd_system - it must be string-like
    try:
        crd_system = crd_system.strip().lower()
    except AttributeError:
        if crd_system is None:
            crd_system = "Unknown"
        elif default is _NOT_GIVEN:
            raise TypeError("Cound not decipher crd_system: {0}"
                            "".format(crd_system))
        else:
            crd_system = as_crd_system(default)

    return crd_system


class Cotr(object):
    """Transform geocentric vectors between crd systems @ a given UTC"""
    # This lookup is a composition of elementary transforms as per
    # [Hapgood1992].
    # In this lookup, the number refers to the index of a elementary matrix,
    # and if it's negative, it indicates the inverse (transpose) of that matrix
    # yay orthoginality :)
    #
    # Note that this dict only contans transformations in one direction, so you
    # may need to reverse the lookup and transpose the result if you want to
    # go the other way
    #
    # Hapgood 1992 caveats:
    #   - equations are simplified, but accurate up to 0.001deg up to year 2100
    #
    # Note that most of these transformations are untested; feel free to
    # verify your favorite transformations and fix them or mark them as tested
    #
    GEOCENTRIC = ['gei', 'geo', 'gse', 'gsm', 'sm', 'mag', 'mhd']
    HELIOCENTRIC = ['hea', 'hee', 'heeq']
    XFORM_LOOKUP = {"gei->gei": (0, ),  # identity
                    "geo->gei": (-1, ),
                    "gse->gei": (-2, ),
                    "gsm->gei": (-2, -3),
                    "sm->gei": (-2, -3, -4),
                    "mag->gei": (-1, -5),
                    "mhd->gei": (-2, 6),
                    "geo->geo": (0, ),  # identity
                    "gse->geo": (1, -2),  # <- tested
                    "gsm->geo": (1, -2, -3),
                    "sm->geo": (1, -2, -3, -4),
                    "mag->geo": (-5, ),
                    "mhd->geo": (1, -2, 6),  # <- tested
                    "gse->gse": (0, ),  # identity
                    "gsm->gse": (-3,),
                    "sm->gse": (-3, -4),  # <- tested
                    "mag->gse": (2, -1, -5),
                    "mhd->gse": (6, ),  # <- tested
                    "gsm->gsm": (0, ),  # identity
                    "sm->gsm": (-4, ),
                    "mag->gsm": (3, 2, -1, -5),
                    "mhd->gsm": (3, 6),
                    "sm->sm": (0, ),  # identity
                    "mag->sm": (4, 3, 2, -1, -5),
                    "mhd->sm": (4, 3, 6),  # <- tested
                    "mag->mag": (0, ),  # identity
                    "mhd->mag": (5, 1, -2, 6),
                    "mhd->mhd": (0, ),  # identity
                    # Heliocentric
                    "hae->hae": (0),  # identity
                    "hee->hae": (-11),
                    "heeq->hae": (-12),
                    "hee->hee": (0),  # identity
                    # "heeq->hee": (0),  # <- unknown transform?
                    "heeq->heeq": (0),  # identity
                   }

    def __init__(self, time='1967-01-01', dip_tilt=None, dip_gsm=None, rdim=3,
                 notilt1967=True):
        """Construct Cotr instance

        Args:
            time (datetime-like): A specific date and time for the
                transformations (sets earth's geographic and dipole
                orientations)
            rdim (int): Dimensionality of rotation matrices
            dip_tilt (float): if given, override the dipole tilt angle
                (mu in degrees, positive points north pole sunward)
            dip_gsm (float): if given, override the dipole gse->gsm
                angle (psi in degrees, positive points north pole
                duskward)
            notilt1967 (bool): if True, then all transforms (except
                <->mhd transforms) are set to the identity if time is
                Jan 1 1967. This is the special OpenGGCM
                no-dipole-tilt-time.
        """
        self._emat_cache = dict()
        self._xform_cache = dict()

        # exact UT time of interest
        time = as_datetime64(time)
        self.time = time

        # dimension of rotation matrices (3 or 4), use 4 to support
        # translations
        self.rdim = rdim

        # modified julian date is days from 00:00UT on 17 Nov 1858
        mjd0 = as_datetime64('1858-11-17T00:00:00.0')
        mjd_td = as_timedelta(time.astype(mjd0.dtype) - mjd0)
        self.mjd = mjd_td.total_seconds() // (24 * 3600)

        # ut is hours from midnight
        dt = as_datetime(time)
        day = as_datetime64("{0:04d}-{1:02d}-{2:02d}"
                            "".format(dt.year, dt.month, dt.day))
        secs = as_timedelta(time - day.astype(time.dtype)).total_seconds()
        self.ut = secs / 3600.0

        # julian centuries (36525 days) since 1 Jan 2000 (epoch 2000)
        self.t0 = (self.mjd - 51544.5) / 36525.0

        # interpolated IGRF
        g10, g11, h11 = _igrf_interpolate(self.time, notilt1967=notilt1967)
        self.g10 = g10
        self.g11 = g11
        self.h11 = h11

        self._disable_mag_crds = dip_tilt is not None or dip_gsm is not None
        years = datetime64_as_years(time)
        notilt = notilt1967 and np.abs(years - 1967.0) < 0.0006

        # do i really want the self._disable_mag_crds part of this? what
        # if i want dip_gsm set by the time, and dip_tilt set by hand...
        # would anyone ever want that? is it really more natural to say that
        # if you specify dip_tilt or dip_gsm then both ignore the time?
        if self._disable_mag_crds or notilt:
            dip_tilt = 0.0 if dip_tilt is None else dip_tilt
            dip_gsm = 0.0 if dip_gsm is None else dip_gsm

        self._cached_sun_ecliptic_longitude = None
        self._cached_q_g = None
        self._cached_q_e = None
        self._cached_dip_geolon_rad = None
        self._cached_dip_geolat_rad = None
        self._cached_dip_tilt = dip_tilt
        self._cached_dip_gsm = dip_gsm

        # hack the cache to set all geocentric transforms to the identity
        # except GSE -> MHD
        if notilt:
            eye = np.eye(self.rdim)
            # matrices 3, 4, and 6 are missing so that gse, gsm, sm, mhd
            # transformations still do something (from dip_tilt and dip_gsm)
            for i in (1, 2, 5):
                self._emat_cache[-i] = eye
                self._emat_cache[i] = eye

    @property
    def __cotr__(self):
        return self

    @property
    def sun_ecliptic_longitude(self):
        # evidently self.ut should really be TDT here, but the error in
        # lam_S is only ~0.0007deg
        if self._cached_sun_ecliptic_longitude is None:
            m = 357.528 + 35999.050 * self.t0 + 0.04107 * self.ut
            m_rad = np.deg2rad(m)
            lam = 280.460 + 36000.772 * self.t0 + 0.04107 * self.ut
            lam_S = (lam + (1.915 - 0.0048 * self.t0) * np.sin(m_rad) +
                     0.020 * np.sin(2 * m_rad))
            self._cached_sun_ecliptic_longitude = lam_S
        return self._cached_sun_ecliptic_longitude

    @property
    def dip_geolon_rad(self):
        if self._cached_dip_geolon_rad is None:
            # note that adding pi is a little bit of a hack to put
            # lamda in the 4th quadrant... it doesn't really change anything,
            # but in practice, it keeps dip_geolat from going > 90deg
            lam_rad = np.arctan2(self.h11, self.g11) + np.pi
            # lam_rad = np.deg2rad(289.1 + 1.413e-2 * ((self.mjd - 46066) / 365.25))  # <<<<<<<
            self._cached_dip_geolon_rad = lam_rad
        return self._cached_dip_geolon_rad

    @property
    def dip_geolat_rad(self):
        if self._cached_dip_geolat_rad is None:
            lam_rad = self.dip_geolon_rad
            cos_lam = np.cos(lam_rad)
            sin_lam = np.sin(lam_rad)
            _top = self.g11 * cos_lam + self.h11 * sin_lam
            phi_rad = (np.pi / 2) - np.arcsin(_top / self.g10)
            # phi_rad = np.deg2rad(78.8 + 4.283e-2 * ((self.mjd - 46066) / 365.25))  # <<<<<<<
            self._cached_dip_geolat_rad = phi_rad
        return self._cached_dip_geolat_rad

    @property
    def Q_g(self):
        if self._cached_q_g is None:
            lam_rad = self.dip_geolon_rad
            cos_lam = np.cos(lam_rad)
            sin_lam = np.sin(lam_rad)
            phi_rad = self.dip_geolat_rad
            cos_phi = np.cos(phi_rad)
            sin_phi = np.sin(phi_rad)
            q_g = np.array([cos_phi * cos_lam, cos_phi * sin_lam, sin_phi])
            q_g = as_nvec(q_g, ndim=self.rdim, init_vals=(0, 0, 0, 1))
            self._cached_q_g = q_g
        return self._cached_q_g

    @property
    def Q_e(self):
        if self._cached_q_e is None:
            self._cached_q_e = self.transform("geo", "gse", self.Q_g)
        return self._cached_q_e

    @property
    def dip_geolon(self):
        return np.rad2deg(self.dip_geolon_rad)

    @property
    def dip_geolat(self):
        return np.rad2deg(self.dip_geolat_rad)

    @property
    def dip_gsm(self):
        """Angle psi between GSE-Z and dipole projected into GSE-YZ

        Note:
            SM -> GSE, rotation is <-mu, Y> * <-psi, X> which is
            equivalent to <psi, X> * <mu, Y>
        """
        if self._cached_dip_gsm is None:
            _, y_e, z_e = self.Q_e[:3]
            self._cached_dip_gsm = np.rad2deg(np.arctan2(y_e, z_e))
        return self._cached_dip_gsm

    @property
    def dip_tilt(self):
        """Dipole tilt angle mu (angle between GSM-Z and dipole)

        Note:
            SM -> GSE, rotation is <-mu, Y> * <-psi, X> which is
            equivalent to <psi, X> * <mu, Y>
        """
        if self._cached_dip_tilt is None:
            x_e, y_e, z_e = self.Q_e[:3]
            mu = np.rad2deg(np.arctan2(x_e, np.sqrt(y_e**2 + z_e**2)))
            self._cached_dip_tilt = mu
        return self._cached_dip_tilt

    dip_psi = dip_gsm
    dip_mu = dip_tilt

    def _get_emat(self, which):
        # create the elementary transform if it doesn't already exist
        if which not in self._emat_cache:
            abs_which = abs(which)

            if abs_which == 0:
                # identity
                mat = np.eye(self.rdim)
                imat = mat.T

            elif abs_which == 1:
                # gei -> geo: "rotate in the plane of earth's geographic
                #              equator from 1st pt of Aries to Greenwich
                #              meridian" [Hapgood1992]_
                theta = 100.461 + 36000.770 * self.t0 + 15.04107 * self.ut
                mat = make_rotation(theta, 'z', rdim=self.rdim)
                imat = mat.T

            elif abs_which == 2:
                # gei -> gse: e2 rotates "from the Earth's equator to the plane
                #             of the ecliptic", while e1 rotates "in the plane
                #             of the ecliptic from the First Point of Aries to
                #             the Earth-Sun direction" [Hapgood1992]_
                epsilon = 23.439 - 0.013 * self.t0
                lam_S = self.sun_ecliptic_longitude
                e1 = make_rotation(lam_S, 'z', rdim=self.rdim)
                e2 = make_rotation(epsilon, 'x', rdim=self.rdim)
                mat = np.dot(e1, e2)
                imat = mat.T

            elif abs_which == 3:
                # gse -> gsm: rotate around GSE x-axis
                mat = make_rotation(-self.dip_psi, 'x', rdim=self.rdim)
                imat = mat.T

            elif abs_which == 4:
                # gsm -> sm: rotate by dipole tilt around GSM y-axis
                mat = make_rotation(-self.dip_mu, 'y', rdim=self.rdim)
                imat = mat.T

            elif abs_which == 5:
                # geo -> mag: e2 rotates "in the plane of Earth's equator from
                #             the Greenwich meridian to the meridian containing
                #             the dipole pole" and e1 rotates "in that meridian
                #             from the geographic pole to the dipole pole"
                #             [Hapgood1992]_
                if self._disable_mag_crds:
                    raise RuntimeError("Setting dip_tilt or dip_gsm by hand "
                                       "makes Geomagnetic crds meaningless.")
                phi = self.dip_geolat
                lam = self.dip_geolon
                e1 = make_rotation(phi - 90.0, 'y', rdim=self.rdim)
                e2 = make_rotation(lam, 'z', rdim=self.rdim)
                mat = np.dot(e1, e2)
                imat = mat.T

            elif abs_which == 6:
                # gse -> mhd == mhd -> gse: OpenGGCM's flipped x/y axes
                mat = np.diag([-1, -1, 1, 1][:self.rdim])
                imat = mat.T

            elif abs_which == 11:
                # hae->hee:
                raise NotImplementedError("Heliocentric transforms not implemented")

            elif abs_which == 12:
                # hae->heeq:
                raise NotImplementedError("Heliocentric transforms not implemented")

            else:
                raise ValueError("No such elementary transformation {0}/{1}"
                                 "".format(which, abs_which))

            # Note that for translations, imat != mat.T
            self._emat_cache[-abs_which] = imat
            self._emat_cache[abs_which] = mat

        # ok, now the transform we need should be cached
        return self._emat_cache[which]

    def get_transform_matrix(self, from_system, to_system):
        """Get a transformation matrix from one system to another

        Args:
            from_system (str): abbreviation of crd_system
            to_system (str): abbreviation of crd_system

        Returns:
            ndarray: self.rdim x self.rdim transformation matrix
        """
        from_system = as_crd_system(from_system)
        to_system = as_crd_system(to_system)

        xform = "->".join([from_system, to_system])

        if xform not in self._xform_cache:
            xform_rev = "->".join([to_system, from_system])

            # emat_lst is always the list of transformations to go in the
            # desired from_system -> to_system direction
            if xform in self.XFORM_LOOKUP:
                emat_lst = list(self.XFORM_LOOKUP[xform])
            elif xform_rev in self.XFORM_LOOKUP:
                emat_lst = [-v for v in reversed(self.XFORM_LOOKUP[xform_rev])]
            else:
                raise ValueError("Unknown crd transformation: {0}".format(xform))

            mat = np.eye(self.rdim)
            for emati in emat_lst:
                mat = np.dot(mat, self._get_emat(emati))

            self._xform_cache[xform] = mat
            self._xform_cache[xform_rev] = mat.T

        return self._xform_cache[xform]

    def transform(self, from_system, to_system, vec):
        """Transform a vector from one system to another

        Args:
            from_system (str): abbreviation of crd_system
            to_system (str): abbreviation of crd_system
            vec (ndarray): should have shape (3, ) or (3, N) to
                transform N vectors at once

        Returns:
            ndarray: vec in new crd system shaped either (3, ) or
                (3, N) mirroring the shape of vec
        """
        from_system = as_crd_system(from_system)
        to_system = as_crd_system(to_system)

        in_vec = np.asarray(vec)
        vec = as_nvec(in_vec, ndim=self.rdim)
        if len(vec.shape) == 1:
            vec = vec.reshape([-1, 1])

        if from_system in self.HELIOCENTRIC and to_system in self.GEOCENTRIC:
            ret = None
            raise NotImplementedError("Heliocentric -> Geocentric not implemented")
        elif from_system in self.GEOCENTRIC and to_system in self.HELIOCENTRIC:
            ret = None
            raise NotImplementedError("Geocentric -> Heliocentric not implemented")
        else:
            mat = self.get_transform_matrix(from_system, to_system)
            ret = np.dot(mat, vec)

        # the slice is in case as_nvec made vec larger to accomidate
        # translation, but make sure we return at least 3-D
        ret = np.array(ret[:max(in_vec.shape[0], 3), ...])

        if len(in_vec.shape) == 1:
            ret = ret[:, 0]
        return ret

    def get_rotation_wxyz(self, from_system, to_system):
        """Get rotation axis and angle for a transformation

        Args:
            from_system (str): abbreviation of crd_system
            to_system (str): abbreviation of crd_system

        Returns:
            ndarray: [w, x, y, z] where w is the angle around the axis
                vector [x, y, z]
        """
        mat = self.get_transform_matrix(from_system, to_system)
        return get_rotation_wxyz(mat)

    def get_dipole_moment(self, crd_system='gse', strength=DEFAULT_STRENGTH):
        """Get Earth's dipole moment

        Args:
            crd_system (str): crd_system of resulting moment
            strength (float): magnitude of dipole moment

        Returns:
            ndarray: 3-vector of dipole tilt in crd_system
        """
        return self.transform('sm', crd_system, [0, 0, -strength])

    def get_dipole_angles(self):
        """Get rotation angles between dipole and GSE

        Note:
            `psi` refers to the angle between GSE and GSM, while `mu`
            is the dipole tilt angle.

        Returns:
            (tilt, gsm): The SM -> GSE, rotation is <-mu, Y> * <-psi, X>
                which is equivalent to <psi, X> * <mu, Y>
        """
        return self.dip_tilt, self.dip_gsm

def as_cotr(thing=None, default=_NOT_GIVEN):
    """Try to make cotr into a Cotr instance

    Inputs understood are:
      - None for North-South dipole
      - anything with a `__cotr__` property
      - datetimes or similar, specifies the time for finding dip angles
      - mapping (dict or similar), passed to Cotr constructor as kwargs

    Args:
        thing (obj): something to turn into a Cotr
        default (None, obj): fallback value

    Returns:
        Cotr instance

    Raises:
        TypeError: if no default supplied and thing can't be turned
            into a Cotr instance
    """
    cotr = _NOT_GIVEN

    if thing is None:
        cotr = Cotr(dip_tilt=0.0, dip_gsm=0.0)
    elif hasattr(thing, "__cotr__"):
        cotr = thing.__cotr__
    elif is_datetime_like(thing):
        cotr = Cotr(time=thing)
    else:
        try:
            cotr = thing.find_attr("__cotr__")
        except AttributeError:
            if hasattr(thing, "find_info"):
                if thing.find_info("cotr", None):
                    cotr = as_cotr(thing.find_info("cotr"))  # recursive
                elif thing.find_info("dipoletime", None):
                    cotr = as_cotr(thing.find_info("dipoletime"))  # recursive

    if cotr is _NOT_GIVEN:
        if default is _NOT_GIVEN:
            raise TypeError("Cound not decipher cotr: {0}".format(thing))
        else:
            # recursive to support default == None -> 0 tilt Cotr
            cotr = as_cotr(default)

    return cotr

def cotr_transform(date_time, from_system, to_system, vec, notilt1967=True):
    """Transform a vector from one system to another

    Args:
        date_time (str, datetime64): datetime of transformation
        from_system (str): abbreviation of crd_system
        to_system (str): abbreviation of crd_system
        vec (ndarray): should have shape (3, ) or (N, 3) to
            transform N vectors at once
        notilt1967 (bool): is 1 Jan 1967 the special notilt time?

    Returns:
        ndarray: vec in new crd system
    """
    c = Cotr(date_time, notilt1967=notilt1967)
    return c.transform(from_system, to_system, vec)

def get_dipole_moment(date_time, crd_system='gse', strength=DEFAULT_STRENGTH,
                      notilt1967=True):
    """Get Earth's dipole moment at datetime

    Args:
        date_time (str, datetime64): datetime to get dipole
        crd_system (str): crd_system of resulting moment
        strength (float): magnitude of dipole moment
        notilt1967 (bool): is 1 Jan 1967 the special notilt time?

    Returns:
        ndarray: 3-vector of dipole tilt in crd_system
    """
    c = Cotr(date_time, notilt1967=notilt1967)
    return c.get_dipole_moment(crd_system, strength=strength)

def get_dipole_moment_ang(dip_tilt=0.0, dip_gsm=0.0, crd_system='gse',
                          strength=DEFAULT_STRENGTH):
    """Get dipole moment from tilt and gsm angles

    Args:
        dip_tilt (float): if given, override the dipole tilt angle
            (mu degrees, positive points north pole sunward)
        dip_gsm (float): if given, override the dipole gse->gsm
            angle (psi in degrees, positive points north pole
            duskward)
        crd_system (str): coordinate system of the result
        strength (float): magnitude of the dipole moment

    Returns:
        ndarray: dipole moment [mx, my, mz]
    """
    c = Cotr(0, dip_tilt=dip_tilt, dip_gsm=dip_gsm)
    return c.get_dipole_moment(strength=strength, crd_system=crd_system)

def get_dipole_angles(date_time, notilt1967=True):
    """Get rotation angles between dipole and GSE

    Args:
        date_time (str, datetime64): datetime to get dipole
        notilt1967 (bool): is 1 Jan 1967 the special notilt time?

    Note:
        `psi` refers to the angle between GSE and GSM, while `mu`
        is the dipole tilt angle.

    Returns:
        (tilt, gsm): The SM -> GSE, rotation is <-mu, Y> * <-psi, X>
            which is equivalent to <psi, X> * <mu, Y>
    """
    c = Cotr(date_time, notilt1967=notilt1967)
    return c.get_dipole_angles()

def dipole_moment2cotr(m, crd_system='gse'):
    """Turn dipole moment vector into a Cotr instance

    Args:
        m (sequence): dipole moment as sequence with length 3
        crd_system (str): crd_system of m

    Returns:
        Cotr instance
    """
    if as_crd_system(crd_system) not in ('gse', 'mhd'):
        raise ValueError('bad crd_system: {0}'.format(crd_system))
    m = Cotr(dip_tilt=0.0, dip_gsm=0.0).transform(crd_system, 'gse', m)
    gsm = np.rad2deg(np.arctan2(m[1], m[2]))
    tilt = np.rad2deg(np.arctan2(m[0], np.linalg.norm(m[1:])))
    return Cotr(dip_tilt=tilt, dip_gsm=gsm)


def _main():
    """This _main is a faux unit-test of cotr

    It makes 3d plots using both Viscid and Mayavi
    """
    plot_mpl = True
    plot_vlab = True
    crd_system = 'gse'

    dtfmt = "%Y-%m-%d %H:%M"

    t = "2010-06-23T00:00:00.0"
    print("Dipole Moment at {0},".format(format_datetime(t, dtfmt)))
    print("    - mhd:", get_dipole_moment(t, crd_system='gse', strength=1.0))
    print("    - gse:", get_dipole_moment(t, crd_system='mhd', strength=1.0))

    if plot_mpl:
        try:
            from viscid.plot import vpyplot as vlt
        except ImportError:
            pass
        from matplotlib import pyplot as plt
        import matplotlib.dates as mdates

        times = linspace_datetime64("2010-01-01T00:00:00.0",
                                    "2010-06-21T00:00:00.0", n=(365//2*24))
        # times = linspace_datetime64("1952-01-01T00:00:00.0",
        #                             "1956-01-01T00:00:00.0", n=(4*365*24*2))

        t_dt = as_datetime(times).tolist()
        m_gse = np.empty((len(times), 3), dtype='f8')
        m_sm = np.empty((len(times), 3), dtype='f8')
        psi = np.empty((len(times), ), dtype='f8')
        mu = np.empty((len(times), ), dtype='f8')
        message_cadence = len(times) // 20
        for i, t in enumerate(times):
            if i % message_cadence == 0:
                print("Getting moment for: {0} {1} ({2:.0f}% complete)"
                      "".format(i, t, 100 * i / len(times)))
            c = Cotr(t)
            m_gse[i, :] = c.get_dipole_moment(crd_system='gse', strength=1)
            m_sm[i, :] = c.get_dipole_moment(crd_system='sm', strength=1)
            mu[i], psi[i] = c.get_dipole_angles()
        dip_angle = np.rad2deg(np.arccos(np.sum(m_gse * m_sm, axis=1)))

        i_smallest_diptilt = np.argmin(np.abs(mu))
        i_largest_diptilt = np.argmax(np.abs(mu))
        i_smallest_dipangle = np.argmin(np.abs(dip_angle))
        i_largest_dipangle = np.argmax(np.abs(dip_angle))

        print()
        print("Dipole info between {0} and {1},"
              "".format(format_datetime(times[0], dtfmt),
                        format_datetime(times[-1], dtfmt)))
        print("    - Smallest dipole tilt angle: {1:.03g} deg @ {0}"
              "".format(format_datetime(times[i_smallest_diptilt], dtfmt),
                        mu[i_smallest_diptilt]))
        print("    - Largest dipole tilt angle: {1:.03g} deg @ {0}"
              "".format(format_datetime(times[i_largest_diptilt], dtfmt),
                        mu[i_largest_diptilt]))
        print("    - Smallest angle between dip and GSE-Z: {1:.03g} deg @ {0}"
              "".format(format_datetime(times[i_smallest_dipangle], dtfmt),
                        dip_angle[i_smallest_dipangle]))
        print("    - Largest angle between dip and GSE-Z: {1:.03g} deg @ {0}"
              "".format(format_datetime(times[i_largest_dipangle], dtfmt),
                        dip_angle[i_largest_dipangle]))

        plt.clf()
        ax0 = plt.subplot(211)
        plt.plot(t_dt, psi, label='GSM Angle')
        plt.plot(t_dt, mu, label='DIP Tilt Angle')
        plt.gca().get_xaxis().set_visible(False)
        plt.legend(loc=0)
        plt.subplot(212, sharex=ax0)
        plt.plot(t_dt, dip_angle, label='Angle between dip and GSE-Z')
        # plt.ylim(11.62, 11.73)  # to see knee on 1954-12-14 @ 17:05
        dateFmt = mdates.DateFormatter(dtfmt)
        plt.gca().xaxis.set_major_formatter(dateFmt)
        plt.gcf().autofmt_xdate()
        plt.legend(loc=0)
        plt.subplots_adjust(left=0.15, right=0.98, top=0.97)
        plt.show()

        # m = m_gse
        # plt.clf()
        # plt.subplot(311)
        # plt.plot(times, m[:, 0], label='M$_{0}$')
        # plt.subplot(312)
        # plt.plot(times, m[:, 1], label='M$_{1}$')
        # plt.subplot(313)
        # plt.plot(times, m[:, 2], label='M$_{2}$')
        # plt.show()

    if plot_vlab:
        import os
        import viscid
        from viscid.plot import vlab

        vlab.figure(size=(768, 768), fgcolor=(0, 0, 0), bgcolor=(1, 1, 1),
                    offscreen=True)

        def _plot_time_range(times, figname):
            for i, t in enumerate(times):
                vlab.clf()
                cotr = Cotr(t)

                vlab.plot_blue_marble(r=1.0, rotate=t, crd_system=crd_system,
                                      nphi=256, ntheta=128, res=4, lines=True)

                vlab.plot_earth_3d(radius=1.005, crd_system=crd_system,
                                   night_only=True, opacity=0.5)

                mag_north = cotr.transform('sm', crd_system, [0, 0, 1.0])

                vlab.mlab.points3d(*mag_north, scale_factor=0.05, mode='sphere',
                                   color=(0.992, 0.455, 0.0), resolution=32)
                vlab.orientation_axes(line_width=4.0)

                vlab.mlab.text(0.325, 0.95, viscid.format_datetime(t))

                vlab.view(azimuth=0.0, elevation=90.0, distance=5.0,
                          focalpoint=[0, 0, 0])
                vlab.savefig("{0}_eq_{1:06d}.png".format(figname, i))
                vlab.view(azimuth=0.0, elevation=0.0, distance=5.0,
                          focalpoint=[0, 0, 0])
                vlab.savefig("{0}_pole_{1:06d}.png".format(figname, i))

        path = os.path.expanduser("~/Desktop/day/{0}/".format(crd_system))
        if not os.path.exists(path):
            os.makedirs(path)
        print()
        print("Writing 3D images in:", path)

        # summer solstice (earth-sun line @ tropic of cancer)
        print()
        print("Making a movie for the June solstice")
        solstice_times = linspace_datetime64("2010-06-21T00:00:00.0",
                                             "2010-06-22T00:00:00.0", n=49)
        _plot_time_range(solstice_times, path + "solstice")
        pfx = path + "solstice_eq"
        viscid.make_animation(pfx + '.mp4', pfx, framerate=23.976, yes=1)
        pfx = path + "solstice_pole"
        viscid.make_animation(pfx + '.mp4', pfx, framerate=23.976, yes=1)

        # autumnal equinox (earth-sun line @ equator)
        print()
        print("Making a movie for the September equinox")
        equinox_times = linspace_datetime64("2010-09-23T00:00:00.0",
                                            "2010-09-24T00:00:00.0", n=49)
        _plot_time_range(equinox_times, path + "equinox")
        pfx = path + "equinox_eq"
        viscid.make_animation(pfx + '.mp4', pfx, framerate=23.976, yes=1)
        pfx = path + "equinox_pole"
        viscid.make_animation(pfx + '.mp4', pfx, framerate=23.976, yes=1)

        # Watching the magnetic pole move year-by-year
        print()
        print("Making a movie to show magnetic pole motion")
        years = range(1950, 2016, 1)
        times = [as_datetime64("{0:04d}-06-21".format(y)) for y in years]
        _plot_time_range(times, path + "year")
        pfx = path + "year_eq"
        viscid.make_animation(pfx + '.mp4', pfx, yes=1, framerate=8)
        pfx = path + "year_pole"
        viscid.make_animation(pfx + '.mp4', pfx, yes=1, framerate=8)

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
