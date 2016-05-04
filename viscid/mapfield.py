#!/usr/bin/env python
"""utility for translating spherical fields (theta, phi) <-> (lat, lon)"""

from __future__ import division, print_function

import numpy as np
import viscid
from viscid.coordinate import wrap_crds


__all__ = ["as_mapfield", "as_spherefield", "pass_through",
           "convert_coordinates", "lat2theta", "lon2phi", "mlt2phi",
           "phi2lon", "phi2mlt", "theta2lat"]

def _prep_units(units, fld=None):
    if units is None:
        units = 'none'
    units = units.strip().lower()
    # divine units from fld if needed
    if units == 'none':
        if fld is None:
            raise ValueError("if units is none, fld must be valid")
        units = fld.crds.meta.get('units', 'deg')
    # clean units to be either 'deg' or 'rad'
    if units.startswith('deg'):
        units = 'deg'
    elif units.startswith('rad'):
        units = 'rad'
    else:
        raise ValueError()
    return units

def theta2lat(theta, units='deg'):
    units = _prep_units(units)
    p90 = np.pi / 2 if units == 'rad' else 90.0
    lat = p90 - theta
    if np.any(lat < -p90) or np.any(lat > p90):
        raise ValueError("Latitude is not bound between -{0} and {0}".format(p90))
    return lat

def phi2lon(phi, units='deg'):
    units = _prep_units(units)
    p180 = np.pi if units == 'rad' else 180.0
    lon = phi - p180
    if np.any(lon < -p180) or np.any(lon > p180):
        raise ValueError("Longitude is not bound between -{0} and {0}".format(p180))
    return lon

def phi2mlt(phi, units='deg'):
    units = _prep_units(units)
    p360 = 2 * np.pi if units == 'rad' else 360.0
    mlt = (24 / p360) * phi
    if np.any(mlt < 0.0) or np.any(mlt > 24.0):
        raise ValueError("MLT is not bound between 0 and 24")
    return mlt

def lat2theta(lat, units='deg', warn=True):
    units = _prep_units(units)
    p90 = np.pi / 2 if units == 'rad' else 90.0
    p180 = np.pi if units == 'rad' else 180.0
    try:
        if warn and len(lat) >= 2 and lat[0] < lat[1]:
            viscid.logger.warn("Spherical Fields expect the data to be "
                               "ordered {0}..-{0} in latitude (theta: 0..{1})"
                               "".format(p90, p180))
    except TypeError:
        pass
    theta = p90 - lat
    if np.any(theta < 0.0) or np.any(theta > p180):
        raise ValueError("Theta is not bound between 0 and {0}".format(p180))
    return theta

def lon2phi(lon, units='deg'):
    units = _prep_units(units)
    p180 = np.pi if units == 'rad' else 180.0
    p360 = 2 * np.pi if units == 'rad' else 360.0
    phi = lon + p180
    if np.any(phi < 0.0) or np.any(phi > p360):
        raise ValueError("Phi is not bound between 0 and {0}".format(p360))
    return phi

def mlt2phi(mlt, units='deg'):
    units = _prep_units(units)
    p360 = 2 * np.pi if units == 'rad' else 360.0
    phi = (p360 / 24) * mlt
    if np.any(phi < 0.0) or np.any(phi > p360):
        raise ValueError("Phi is not bound between 0 and {0}".format(p360))
    return phi

def pass_through(x):
    return x

def convert_coordinates(fld, order, crd_mapping):
    base_axes = fld.crds.axes

    # select the bases in crd_mapping that match the axes in fld.crds
    _mapping = {}
    for target, base_info in crd_mapping.items():
        for base, info in base_info.items():
            if base in base_axes:
                info = dict(info)
                info['base'] = base
                _mapping[target] = info
                break
        if not target in _mapping:
            raise RuntimeError("no mapping found to go {0} -> {1}"
                               "".format(base_axes, target))

    bases_reordered = [_mapping[ax]['base'] for ax in order]
    transform = [_mapping[ax]['base2alias'] for ax in order]
    reverse = [_mapping[ax]['reverse'] for ax in order]

    all_pass_through = all(x == pass_through for x in transform)

    if all_pass_through and list(base_axes) == list(order):
        ret = fld
    else:
        dat_transpose = [base_axes.index(_mapping[ax]['base']) for ax in order]
        fwd, bkwd = slice(None), slice(None, None, -1)
        dat_slice = [bkwd if rev else fwd for rev in reverse]

        # for fields with vector components, the slice / transpose were just
        # done using the coordinates... so adjust the values to include the
        # component dimension
        if fld.nr_comps:
            for i, _ in enumerate(dat_transpose):
                if dat_transpose[i] >= fld.nr_comp:
                    dat_transpose[i] += 1
            dat_transpose.insert(fld.nr_comp, fld.nr_comp)
            dat_slice.insert(slice(None), fld.nr_comp)

        # change the coordinates
        uniform = fld.crds.is_uniform()
        clist = fld.crds.get_clist(axes=bases_reordered)

        for i, c in enumerate(clist):
            c[0] = order[i]
            if uniform:
                c[1][0] = transform[i](c[1][0])
                c[1][1] = transform[i](c[1][1])
                if reverse[i]:
                    c[1][0], c[1][1] = c[1][1], c[1][0]
            else:
                c[1] = transform[i](c[1])
                if reverse[i]:
                    c[1] = c[1][::-1]

        crds = wrap_crds(fld.crds.crdtype, clist, dtype=fld.crds.dtype)

        # adjust the data
        dat = fld.data
        if list(dat_transpose) != list(range(len(dat_transpose))):
            dat = np.transpose(fld.data, axes=dat_transpose)

        ctx = dict(crds=crds)
        ret = fld.wrap(dat, context=ctx)
    return ret

def _make_unit_xform(fld, units):
    units = _prep_units(units, fld=fld)
    crd_units = _prep_units(fld.crds.meta.get("units", 'none'), fld=fld)
    if units == crd_units:
        pre_xform = pass_through
    elif crd_units == 'rad' and units == 'deg':
        pre_xform = lambda x: (180.0 / np.pi) * x
    elif crd_units == 'deg' and units == 'rad':
        pre_xform = lambda x: (np.pi / 180.0) * x
    else:
        raise ValueError()
    return pre_xform

def as_mapfield(fld, order=('lon', 'lat'), units='none'):
    """Make sure a field has (lon, lat) coordinates

    Args:
        fld (TYPE): Description
        order (tuple, optional): Description
        radians (bool, optional): Description

    Returns:
        Viscid.Field: a field with (lon, lat) coordinates. The data
        will be loaded into memory if not already.
    """
    units = _prep_units(units, fld=fld)
    pre_xform = _make_unit_xform(fld, units)
    _phi2lon = lambda arr: phi2lon(pre_xform(arr), units=units)
    _theta2lat = lambda arr: theta2lat(pre_xform(arr), units=units)

    mapping = dict(lon={'phi': dict(base2alias=_phi2lon, reverse=False),
                        'lon': dict(base2alias=pre_xform, reverse=False)},
                   lat={'theta': dict(base2alias=_theta2lat, reverse=True),
                        'lat': dict(base2alias=pre_xform, reverse=False)})

    ret = convert_coordinates(fld, order, mapping)
    ret.crds.meta['units'] = units
    return ret

def as_spherefield(fld, order=('phi', 'theta'), units='none'):
    """Make sure fld has (phi, theta) coordinates

    Args:
        fld (TYPE): Description
        order (tuple, optional): Description
        radians (bool, optional): Description

    Returns:
        Viscid.Field: a field with (phi, theta) coordinates. The data
        will be loaded into memory if not already.
    """
    units = _prep_units(units, fld=fld)
    pre_xform = _make_unit_xform(fld, units)
    _lon2phi = lambda arr: lon2phi(pre_xform(arr), units=units)
    _lat2theta = lambda arr: lat2theta(pre_xform(arr), units=units)

    mapping = dict(phi={'lon': dict(base2alias=_lon2phi, reverse=False),
                        'phi': dict(base2alias=pre_xform, reverse=False)},
                   theta={'lat': dict(base2alias=_lat2theta, reverse=True),
                          'theta': dict(base2alias=pre_xform, reverse=False)})

    ret = convert_coordinates(fld, order, mapping)
    ret.crds.meta['units'] = units
    return ret

##
## EOF
##
