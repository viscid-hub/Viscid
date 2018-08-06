#!/usr/bin/env python
"""utility for translating spherical fields (theta, phi) <-> (lat, lon)"""

from __future__ import division, print_function
import sys

import numpy as np
import viscid
from viscid.coordinate import wrap_crds


__all__ = ["as_mapfield", "as_spherefield", "as_polar_mapfield",
           "pts2polar_mapfield", "convert_coordinates",
           "lat2theta", "lon2phi", "phi2lon", "theta2lat",
           "cart2sph", "cart2latlon", "sph2cart", "latlon2cart",
           "great_circle"]


def theta2lat(theta, unit='deg'):
    """spherical theta -> latitude transformation"""
    p90 = np.pi / 2 if unit == 'rad' else 90.0
    lat = p90 - np.asarray(theta)
    if np.any(lat < -p90) or np.any(lat > p90):
        raise ValueError("Latitude is not bound between -{0} and {0}".format(p90))
    return lat

def phi2lon(phi, unit='deg'):  # pylint: disable=unused-argument
    """spherical phi -> longitude transform; currently a no-op"""
    return phi

def lat2theta(lat, unit='deg'):
    """spherical latitude -> theta transformation"""
    p90 = np.pi / 2 if unit == 'rad' else 90.0
    p180 = np.pi if unit == 'rad' else 180.0
    theta = p90 - np.asarray(lat)
    if np.any(theta < 0.0) or np.any(theta > p180):
        raise ValueError("Theta is not bound between 0 and {0}".format(p180))
    return theta

def lon2phi(lon, unit='deg'):  # pylint: disable=unused-argument
    """spherical longitude -> phi transform; currently a no-op"""
    return lon

def arr_noop(x, **_):
    return x

def fld_noop(fld, base, target, unit=''):
    """Although this is called noop, it could do a unit transform"""
    crd_xform = arr_noop
    dat_xform = arr_noop
    return _fld_xform_common(fld, base, target, unit, crd_xform, dat_xform)

def _prep_units(order, units='', fld=None):
    if not isinstance(units, (list, tuple)):
        units = [units] * len(order)
    units += [None] * (len(units) - len(order))
    units = ['' if u is None else u.strip() for u in units]
    if fld is not None:
        crds = fld.crds
        units = [u if u else crds.get_unit(ax) for ax, u in zip(order, units)]
    return units

def _make_transform_unit_func(base_unit, target_unit):
    if (base_unit and not target_unit) or base_unit == target_unit:
        ret = arr_noop
    elif target_unit and not base_unit:
        raise ValueError("If target_unit is given, then base unit must be valid")
    elif base_unit == 'deg' and target_unit == 'rad':
        ret = lambda x: (np.pi / 180.0) * np.asarray(x)
    elif base_unit == 'rad' and target_unit == 'deg':
        ret = lambda x: (180.0 / np.pi) * np.asarray(x)
    else:
        raise ValueError("No transform to go {0} -> {1}".format(base_unit,
                                                                target_unit))
    return ret

def _get_axinds(fld, ax):
    dcrd = fld.crds.ind(ax)
    ddat = dcrd + 1 if fld.nr_comps and fld.nr_comp < dcrd else dcrd
    return dcrd, ddat

def _fld_xform_common(fld, base, target, unit, crd_xform, dat_xform, raw=False):
    dcrd, _ = _get_axinds(fld, base)

    unit_xform = _make_transform_unit_func(fld.crds.get_unit(base), unit)
    if not unit:
        unit = fld.crds.get_unit(base)

    xforms = (unit_xform, crd_xform, dat_xform)
    if base == target and all(f is arr_noop for f in xforms):
        return fld
    else:
        clist = fld.get_clist()
        clist[dcrd][0] = target
        if fld.crds.is_uniform() and not raw:
            clist[dcrd][1][:2] = crd_xform(unit_xform(clist[dcrd][1][:2]))
        else:
            clist[dcrd][1] = crd_xform(unit_xform(clist[dcrd][1]))

        units = list(fld.crds.units)
        units[dcrd] = unit
        crds = wrap_crds(fld.crds.crdtype, clist, dtype=fld.crds.dtype,
                         units=units)
        ctx = dict(crds=crds)
        return fld.wrap(dat_xform(fld.data), context=ctx)

def fld_phi2lon(fld, base='phi', target='lon', unit='deg'):
    # FIXME: shouldn't be a noop, but longitude is a stupid unit on account
    # of that bizarre -180 == +180 seam
    return fld_noop(fld, base=base, target=target, unit=unit)

def fld_theta2lat(fld, base='theta', target='lat', unit='deg'):
    _, ddat = _get_axinds(fld, base)
    crd_xform = lambda x: theta2lat(x, unit=unit)[::-1]
    dslc = [slice(None)] * fld.nr_dims
    dslc[ddat] = slice(None, None, -1)
    dat_xform = lambda a: a[tuple(dslc)]
    return _fld_xform_common(fld, base, target, unit, crd_xform, dat_xform)

def fld_lon2phi(fld, base='lon', target='phi', unit='deg'):
    # FIXME: shouldn't be a noop, but longitude is a stupid unit on account
    # of that bizarre -180 == +180 seam
    return fld_noop(fld, base=base, target=target, unit=unit)

def fld_lat2theta(fld, base='lat', target='theta', unit='deg'):
    dcrd, ddat = _get_axinds(fld, base)
    crd_xform = lambda x: lat2theta(x, unit=unit)[::-1]
    dslc = [slice(None)] * fld.nr_dims
    dslc[ddat] = slice(None, None, -1)
    dat_xform = lambda a: a[tuple(dslc)]
    ret = _fld_xform_common(fld, base, target, unit, crd_xform, dat_xform)
    if ret.xh[dcrd] - ret.xl[dcrd] < 0:
        p90 = np.pi / 2 if unit == 'rad' else 90.0
        p180 = np.pi if unit == 'rad' else 180.0
        viscid.logger.warning("Spherical Fields expect the data to be "
                              "ordered {0}..-{0} in latitude (theta: 0..{1})"
                              "".format(p90, p180))
    return ret

def convert_coordinates(fld, order, crd_mapping, units=''):
    """Convert a Field's coordinates

    Args:
        fld (Field): Description
        order (sequence): target coordinates
        crd_mapping (dict): summarizes functions that go from
            base -> target
        units (str, optional): Additional info if you need to convert
            units too

    Raises:
        RuntimeError: If no mapping is found to go base -> target
    """
    base_axes = fld.crds.axes

    # select the bases in crd_mapping that match the axes in fld.crds
    # _mapping is like {'t': {base: 't', base2target: fld_noop},
    #                   'lon': {base: 'phi', base2aslas: fld_phi2lon},
    #                   'lat': {base: 'theta', base2aslas: fld_theta2lat}}
    # where the keys are the aliases (target crds), and the bases are the
    # crds that actuall exist in fld, unlike crd_mapping which contains many
    # bases for each target as a possible transformation
    _mapping = {}
    for target, available_bases in crd_mapping.items():
        for base, transform_func in available_bases.items():
            if base in base_axes:
                _mapping[target] = dict(base=base, base2target=transform_func)
                break
        if not target in _mapping:
            if target in order:
                _mapping[target] = dict(base=target, base2target=fld_noop)
            else:
                raise RuntimeError("no mapping found to go {0} -> {1}"
                                   "".format(base_axes, target))

    bases_target_order = [_mapping[ax]['base'] for ax in order]
    transforms = [_mapping[ax]['base2target'] for ax in order]
    units = _prep_units(order, units=units)

    ret = fld
    for base, target, transform, unit in zip(bases_target_order, order, transforms, units):
        ret = transform(ret, base=base, target=target, unit=unit)

    if list(order) != list(ret.crds.axes):
        ret = ret.spatial_transpose(*order)
    return ret

def as_mapfield(fld, order=('lon', 'lat'), units=''):
    """Make sure a field has (lon, lat) coordinates

    Args:
        fld (Field): some field to transform
        order (tuple, optional): Desired axes of the result
        units (str, optional): units of the result (deg/rad)

    Returns:
        Viscid.Field: a field with (lon, lat) coordinates. The data
        will be loaded into memory if not already.
    """
    mapping = dict(lon={'phi': fld_phi2lon, 'lon': fld_noop},
                   lat={'theta': fld_theta2lat, 'lat': fld_noop})

    ret = convert_coordinates(fld, order, mapping, units=units)
    return ret

def as_spherefield(fld, order=('phi', 'theta'), units=''):
    """Make sure fld has (phi, theta) coordinates

        fld (Field): some field to transform
        order (tuple, optional): Desired axes of the result
        units (str, optional): units of the result (deg/rad)

    Returns:
        Viscid.Field: a field with (lon, lat) coordinates. The data
        will be loaded into memory if not already.
    """
    mapping = dict(phi={'lon': fld_lon2phi, 'phi': fld_noop},
                   theta={'lat': fld_lat2theta, 'theta': fld_noop})
    ret = convert_coordinates(fld, order, mapping, units=units)
    return ret

def as_polar_mapfield(fld, bounding_lat=None, hemisphere='north',
                      make_periodic=False):
    """Prepare a theta/phi or lat/lon field for polar projection

    Args:
        fld (Field): Some scalar or vector field
        bounding_lat (float, optional): Used to slice the field, i.e.,
            gives data from the pole to bounding_lat degrees
            equator-ward of the pole
        hemisphere (str, optional): 'north' or 'south'

    Returns:
        Field: a field that can be mapfield plotted on polar axes

    Raises:
        ValueError: on bad hemisphere
    """
    hemisphere = hemisphere.strip().lower()

    mfld = as_mapfield(fld, order=('lon', 'lat'), units='rad')

    # set a sensible default for bounding_lat... full spheres get cut off
    # at 40 deg, but hemispheres or smaller are shown in full
    if bounding_lat is None:
        if abs(mfld.xh[1] - mfld.xl[1]) >= 1.01 * np.pi:
            bounding_lat = 40.0
        else:
            bounding_lat = 90.0

    bounding_lat = (np.pi / 180.0) * bounding_lat
    abs_bounding_lat = abs(bounding_lat)

    if hemisphere in ("north", 'n'):
        if np.all(mfld.get_crd('lat') < abs_bounding_lat):
            raise ValueError("fld {0} contains no values north of bounding lat "
                             "{1:g} deg"
                             "".format(fld.name, bounding_lat * 180 / np.pi))
        # mfld = mfld["lat=:{0}j:-1".format(abs_bounding_lat)]
        mfld = mfld.loc[:, :np.pi / 2 - abs_bounding_lat:-1]
    elif hemisphere in ("south", 's'):
        if np.all(mfld.get_crd('lat') > -abs_bounding_lat):
            raise ValueError("fld {0} contains no values south of bounding lat "
                             "{1:g} deg"
                             "".format(fld.name, bounding_lat * 180 / np.pi))
        # mfld = mfld["lat=:{0}j".format(-abs_bounding_lat)]
        mfld = mfld.loc[:, :-np.pi / 2 + abs_bounding_lat]
    else:
        raise ValueError("hemisphere should be either north or south")

    clist = mfld.get_clist()
    offset = np.pi / 2
    scale = -1 if hemisphere in ('north', 'n') else 1

    if mfld.crds.is_uniform():
        for i in range(2):
            clist[1][1][i] = scale * clist[1][1][i] + offset
    else:
        clist[1][1] = scale * clist[1][1] + offset

    mfld_dat = mfld.data
    if make_periodic:
        if mfld.crds.is_uniform():
            phi_diff = clist[0][1][1] - clist[0][1][0]
        else:
            phi_diff = clist[0][1][-1] - clist[0][1][0]

        if phi_diff < 2 * np.pi - 1e-5:
            if mfld.crds.is_uniform():
                clist[0][1][1] += phi_diff / clist[0][1][2]
                if not np.isclose(clist[0][1][1] - clist[0][1][0], 2 * np.pi):
                    viscid.logger.warning("Tried unsuccessfully to make uniform "
                                          "polar mapfield periodic: {0:g}"
                                          "".format(clist[0][1][1] - clist[0][1][0]))
            else:
                clist[0][1] = np.concatenate([clist[0][1],
                                              [clist[0][1][0] + 2 * np.pi]])
            mfld_dat = np.concatenate([mfld_dat, mfld_dat[0:1, :]], axis=0)

    crds = wrap_crds(mfld.crds.crdtype, clist, dtype=mfld.crds.dtype)
    ctx = dict(crds=crds)
    return mfld.wrap(mfld_dat, context=ctx)

def pts2polar_mapfield(pts, pts_axes, pts_unit='deg', hemisphere='north'):
    """Prepare theta/phi or lat/lon points for a polar plot

    Args:
        pts (ndarray): A 2xN array of N points where the spatial
            dimensions encode phi, theta, lat, or lon.
        pts_axes (sequence): sequence of strings that say what each
            axis of pts encodes, i.e., ('theta', 'phi').
        pts_unit (str, optional): units of pts ('deg' or 'rad')
        hemisphere (str, optional): 'north' or 'south'

    Returns:
        ndarray: 2xN array of N points in radians where the axes
        are ('lon', 'lat'), such that they can be given straight to
        matplotlib.pyplot.plot()

    Raises:
        ValueError: On bad pts_axes

    Example:
        This example plots total field align currents in the northern
        hemisphere, then plots a curve onto the same plot

        >>> f = viscid.load_file("$SRC/Viscid/sample/*_xdmf.iof.*.xdmf")
        >>> vlt.plot(1e9 * f['fac_tot'], symmetric=True)
        >>> # now make a curve from dawn to dusk spanning 20deg in lat
        >>> N = 64
        >>> pts = np.vstack([np.linspace(-90, 90, N),
                             np.linspace(10, 30, N)])
        >>> pts_polar = viscid.pts2polar_mapfield(pts, ('phi', 'theta'),
                                                  pts_unit='deg')
        >>> plt.plot(pts_polar[0], pts_polar[1])
        >>> vlt.show()

    """
    hemisphere = hemisphere.strip().lower()

    pts = np.array(pts)
    pts_axes = list(pts_axes)

    # convert pts to lat/lon
    if pts_unit == 'deg':
        _to_rad = lambda x: (np.pi / 180.0) * np.asarray(x)
    elif pts_unit == 'rad':
        _to_rad = lambda x: x
    else:
        raise ValueError("bad unit: {0}".format(pts_unit))

    mapping = {'phi': ('lon', phi2lon), 'theta': ('lat', theta2lat)}

    for i, ax in enumerate(pts_axes):
        if ax in mapping:
            pts[i, :] = mapping[ax][1](_to_rad(pts[i, :]), unit='rad')
            pts_axes[i] = mapping[ax][0]
        else:
            pts[i, :] = _to_rad(pts[i, :])

    # reorder axes so pts[0, :] are lon and pts[1, :] are lat
    try:
        transpose_inds = [pts_axes.index(ax) for ax in ('lon', 'lat')]
        pts = pts[transpose_inds, :]
    except ValueError:
        raise ValueError("pts array must have both lon+lat or phi+theta")

    # clist = mfld.get_clist()
    offset = np.pi / 2
    scale = -1 if hemisphere in ('north', 'n') else 1

    pts[1, :] = scale * pts[1, :] + offset

    return pts

def _as_3xN(arr, d=3):
    """return a copy of arr as a 3xN array"""
    arr = np.array(arr)
    was_single_point = arr.shape == (d, )
    arr = arr.reshape((d, -1))
    return arr, was_single_point

def cart2sph(arr, deg=False):
    """Convert cartesian points to spherical (rad by default)

    Args:
        arr (sequence): shaped (3, ) or (3, N) for N xyz points
        deg (bool): if result should be in degrees

    Returns:
        ndarray: shaped (3, ) or (3, N) r, theta, phi points in radians
            by default
    """
    arr, was_single_point = _as_3xN(arr)
    ret = np.zeros_like(arr)
    ret[0, :] = np.sqrt(arr[0, :]**2 + arr[1, :]**2 + arr[2, :]**2)
    ret[1, :] = np.arccos(arr[2, :] / ret[0, :])
    ret[2, :] = np.arctan2(arr[1, :], arr[0, :])
    if deg:
        arr[1:] *= 180.0 / np.pi
    if was_single_point:
        return ret[:, 0]
    else:
        return ret

def sph2cart(arr, deg=False):
    """Convert cartesian points to spherical (rad by default)

    Args:
        arr (sequence): shaped (3, ) or (3, N) for N r, theta, phi
            points in radians by default
        deg (bool): if arr is in dergrees

    Returns:
        ndarray: shaped (3, ) or (3, N) xyz
    """
    arr, was_single_point = _as_3xN(arr)
    if deg:
        arr[1:] *= np.pi / 180.0
    ret = np.zeros_like(arr)
    ret[0, :] = arr[0, :] * np.sin(arr[1, :]) * np.cos(arr[2, :])
    ret[1, :] = arr[0, :] * np.sin(arr[1, :]) * np.sin(arr[2, :])
    ret[2, :] = arr[0, :] * np.cos(arr[1, :])
    if was_single_point:
        return ret[:, 0]
    else:
        return ret

def latlon2cart(arr, r=1.0, deg=True):
    """Convert cartesian points to latitude longitude (deg by default)

    Args:
        arr (sequence): shaped (2, ) or (2, N) for N r, theta, phi
            points in radians by default
        deg (bool): if arr is in dergrees

    Returns:
        ndarray: shaped (3, ) or (3, N) xyz
    """
    quarter_circ = 90 if deg else np.pi / 2
    half_circ = 180 if deg else np.pi
    arr, was_single_point = _as_3xN(arr, d=2)
    arr = np.concatenate([r * np.ones_like(arr[:1, :]), arr], axis=0)
    arr[1, :] = quarter_circ - arr[1, :]
    arr[2, :] += half_circ
    arr_cart = sph2cart(arr, deg=deg)
    if was_single_point:
        return arr_cart[:, 0]
    else:
        return arr_cart

def cart2latlon(arr, deg=True):
    """Convert latitude longitude (deg by default) to cartesian

    Args:
        arr (sequence): shaped (3, ) or (3, N) for N r, theta, phi
            points in radians by default
        deg (bool): if arr is in dergrees

    Returns:
        ndarray: shaped (2, ) or (2, N) xyz
    """
    quarter_circ = 90 if deg else np.pi / 2
    half_circ = 180 if deg else np.pi
    arr, was_single_point = _as_3xN(arr)
    arr_sph = cart2sph(arr, deg=deg)
    arr_sph[1, :] = quarter_circ - arr_sph[1, :]
    arr_sph[2, :] -= half_circ
    if was_single_point:
        return arr_sph[1:, 0]
    else:
        return arr_sph[1:]

def great_circle(p1, p2, origin=(0, 0, 0), n=32):
    """Get great circle path between two points in 3d

    Args:
        p1 (sequence): first point as [x, y, z]
        p2 (sequence): second point as [x, y, z]
        origin (sequence): origin of the sphere
        n (int): Number of line segments along the great circle

    Returns:
        3xN ndarray
    """
    origin = _as_3xN(origin)[0][:, 0]
    p1 = _as_3xN(p1)[0][:, 0] - origin
    p2 = _as_3xN(p2)[0][:, 0] - origin

    r1, _, _ = cart2sph(p1)
    r2, _, _ = cart2sph(p2)

    # generate points on sphere B whose equator (theta = pi / 2) will be the
    # shortest path between p0 and p1

    # Convention: A labels the p0/p1 normal xyz coordinate system, while B
    #             denotes the rotated system

    pole = np.cross(p1, p2)

    # handle edge case colinear p1 == p2
    if np.isclose(np.linalg.norm(pole), 0.0):
        viscid.logger.warning("Great circle says p1 and p2 are colinear.")
        pole = np.array([0, -1, 0])
        if np.isclose(np.linalg.norm(np.cross(pole, p1)), 0.0):
            pole = np.array([0, 0, -1])

    matBtoA = viscid.a2b_rot([0, 0, 1], pole, new_x=p1)

    # this is some code to validate rotation matrix
    _xrot = np.dot(matBtoA, [1, 0, 0])
    _xrot = _xrot / np.linalg.norm(_xrot)
    _p1 = p1 / np.linalg.norm(p1)
    if not np.all(np.isclose(_p1, _xrot)):
        viscid.logger.error("Great circle says new_x: {0} != {1}"
                            "".format(_xrot, _p1))

    dphi = np.arctan2(np.linalg.norm(np.cross(p1, p2)), np.dot(p1, p2))

    r = np.linspace(r1, r2, n)
    theta = (np.pi / 2) * np.ones_like(r)
    phi = np.linspace(0, dphi, n)
    cartB = sph2cart(np.vstack([r, theta, phi]))

    # now rotate our special coordinates into the same coordinates ax p0 and p1
    cartA = np.einsum('ij,jk->ik', matBtoA, cartB)
    return cartA + origin.reshape(3, 1)

def _main():
    try:
        # raise ImportError
        from viscid.plot import vlab
        _HAS_MVI = True
    except ImportError:
        _HAS_MVI = False

    def _test(_p1, _p2, r1=None, r2=None, color=(0.8, 0.8, 0.8)):
        if r1 is not None:
            _p1 = r1 * np.asarray(_p1) / np.linalg.norm(_p1)
        if r2 is not None:
            _p2 = r2 * np.asarray(_p2) / np.linalg.norm(_p2)
        circ = great_circle(_p1, _p2)
        if not np.all(np.isclose(circ[:, 0], _p1)):
            print("!! great circle error P1:", _p1, ", P2:", _p2)
            print("             first_point:", circ[:, 0], "!= P1")
        if not np.all(np.isclose(circ[:, -1], _p2)):
            print("!! great circle error P1:", _p1, ", P2:", _p2)
            print("              last_point:", circ[:, -1], "!= P2")

        if _HAS_MVI:
            vlab.plot_lines([circ], tube_radius=0.02, color=color)

    print("TEST 1")
    _test([1, 0, 0], [0, 1, 0], r1=1.0, r2=1.0, color=(0.8, 0.8, 0.2))
    print("TEST 2")
    _test([1, 0, 0], [-1, 0, 0], r1=1.0, r2=1.0, color=(0.2, 0.8, 0.8))
    print("TEST 3")
    _test([1, 1, 0.01], [-1, -1, 0.01], r1=1.0, r2=1.5, color=(0.8, 0.2, 0.8))

    print("TEST 4")
    _test([-0.9947146, 1.3571029, 2.6095123], [-0.3371437, -1.5566425, 2.6634643],
          color=(0.8, 0.2, 0.2))
    print("TEST 5")
    _test([0.9775307, -1.3741084, 2.6030273], [0.3273931, 1.5570284, 2.6652965],
          color=(0.2, 0.2, 0.8))

    if _HAS_MVI:
        vlab.plot_blue_marble(r=1.0, lines=False, ntheta=64, nphi=128)
        vlab.plot_earth_3d(radius=1.01, night_only=True, opacity=0.5)
        vlab.show()

    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
