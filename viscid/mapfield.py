#!/usr/bin/env python
"""utility for translating spherical fields (theta, phi) <-> (lat, lon)"""

from __future__ import division, print_function

import numpy as np
import viscid
from viscid.coordinate import wrap_crds


__all__ = ["as_mapfield", "as_spherefield", "as_polar_mapfield",
           "convert_coordinates", "lat2theta", "lon2phi",
           "phi2lon", "theta2lat"]


def theta2lat(theta, unit='deg'):
    p90 = np.pi / 2 if unit == 'rad' else 90.0
    lat = p90 - np.asarray(theta)
    if np.any(lat < -p90) or np.any(lat > p90):
        raise ValueError("Latitude is not bound between -{0} and {0}".format(p90))
    return lat

def phi2lon(phi, unit='deg'):
    p180 = np.pi if unit == 'rad' else 180.0
    lon = np.asarray(phi) - p180
    if np.any(lon < -p180) or np.any(lon > p180):
        raise ValueError("Longitude is not bound between -{0} and {0}".format(p180))
    return lon

def lat2theta(lat, unit='deg', warn=True):
    p90 = np.pi / 2 if unit == 'rad' else 90.0
    p180 = np.pi if unit == 'rad' else 180.0
    theta = p90 - np.asarray(lat)
    if np.any(theta < 0.0) or np.any(theta > p180):
        raise ValueError("Theta is not bound between 0 and {0}".format(p180))
    return theta

def lon2phi(lon, unit='deg'):
    p180 = np.pi if unit == 'rad' else 180.0
    p360 = 2 * np.pi if unit == 'rad' else 360.0
    phi = np.asarray(lon) + p180
    if np.any(phi < 0.0) or np.any(phi > p360):
        raise ValueError("Phi is not bound between 0 and {0}".format(p360))
    return phi

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
        fld (TYPE): Description
        order (tuple, optional): Description
        radians (bool, optional): Description

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

    Args:
        fld (TYPE): Description
        order (tuple, optional): Description
        radians (bool, optional): Description

    Returns:
        Viscid.Field: a field with (phi, theta) coordinates. The data
        will be loaded into memory if not already.
    """
    mapping = dict(phi={'lon': fld_lon2phi, 'phi': fld_noop},
                   theta={'lat': fld_lat2theta, 'theta': fld_noop})
    ret = convert_coordinates(fld, order, mapping, units=units)
    return ret

def as_polar_mapfield(fld, bounding_lat=40.0, hemisphere='north'):
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

##
## EOF
##
