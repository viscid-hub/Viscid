#!/usr/bin/env python
"""utility for translating spherical fields (theta, phi) <-> (lat, lon)"""

from __future__ import division, print_function

import numpy as np
import viscid
from viscid.coordinate import wrap_crds


__all__ = ["as_mapfield", "as_spherefield", "as_polar_mapfield",
           "pts2polar_mapfield", "convert_coordinates",
           "lat2theta", "lon2phi", "phi2lon", "theta2lat"]


def theta2lat(theta, unit='deg'):
    p90 = np.pi / 2 if unit == 'rad' else 90.0
    lat = p90 - np.asarray(theta)
    if np.any(lat < -p90) or np.any(lat > p90):
        raise ValueError("Latitude is not bound between -{0} and {0}".format(p90))
    return lat

def phi2lon(phi, unit='deg'):  # pylint: disable=unused-argument
    return phi

def lat2theta(lat, unit='deg'):
    p90 = np.pi / 2 if unit == 'rad' else 90.0
    p180 = np.pi if unit == 'rad' else 180.0
    theta = p90 - np.asarray(lat)
    if np.any(theta < 0.0) or np.any(theta > p180):
        raise ValueError("Theta is not bound between 0 and {0}".format(p180))
    return theta

def lon2phi(lon, unit='deg'):  # pylint: disable=unused-argument
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
    dat_xform = lambda a: a[dslc]
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
    dat_xform = lambda a: a[dslc]
    ret = _fld_xform_common(fld, base, target, unit, crd_xform, dat_xform)
    if ret.xh[dcrd] - ret.xl[dcrd] < 0:
        p90 = np.pi / 2 if unit == 'rad' else 90.0
        p180 = np.pi if unit == 'rad' else 180.0
        viscid.logger.warn("Spherical Fields expect the data to be "
                           "ordered {0}..-{0} in latitude (theta: 0..{1})"
                           "".format(p90, p180))
    return ret

def convert_coordinates(fld, order, crd_mapping, units=''):
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

def as_polar_mapfield(fld, bounding_lat=40.0, hemisphere='north'):
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
    bounding_lat = (np.pi / 180.0) * bounding_lat
    abs_bounding_lat = abs(bounding_lat)

    mfld = as_mapfield(fld, order=('lon', 'lat'), units='rad')

    if hemisphere == "north":
        # mfld = mfld["lat=:{0}f:-1".format(abs_bounding_lat)]
        mfld = mfld.loc[:, :np.pi / 2 - abs_bounding_lat:-1]
    elif hemisphere == "south":
        # mfld = mfld["lat=:{0}f".format(-abs_bounding_lat)]
        mfld = mfld.loc[:, :-np.pi / 2 + abs_bounding_lat]
    else:
        raise ValueError("hemisphere should be either north or south")

    clist = mfld.get_clist()
    offset = np.pi / 2
    scale = -1 if hemisphere == 'north' else 1

    if mfld.crds.is_uniform():
        for i in range(2):
            clist[1][1][i] = scale * clist[1][1][i] + offset
    else:
        clist[1][1] = scale * clist[1][1] + offset

    crds = wrap_crds(mfld.crds.crdtype, clist, dtype=mfld.crds.dtype)
    ctx = dict(crds=crds)
    return mfld.wrap(mfld.data, context=ctx)

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
        >>> mpl.plot(1e9 * f['fac_tot'], symmetric=True)
        >>> # now make a curve from dawn to dusk spanning 20deg in lat
        >>> N = 64
        >>> pts = np.vstack([np.linspace(-90, 90, N),
                             np.linspace(10, 30, N)])
        >>> pts_polar = viscid.pts2polar_mapfield(pts, ('phi', 'theta'),
                                                  pts_unit='deg')
        >>> mpl.plt.plot(pts_polar[0], pts_polar[1])
        >>> mpl.show()

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
    scale = -1 if hemisphere == 'north' else 1

    pts[1, :] = scale * pts[1, :] + offset

    return pts

##
## EOF
##
