"""A collection of tools for dipoles"""
# pylint: disable=bad-whitespace

from __future__ import print_function, division
import sys

import numpy as np
import viscid
from viscid import field
from viscid import seed
from viscid.calculator import interp_trilin
# from viscid import vutil

try:
    import numexpr as ne  # pylint: disable=wrong-import-order
    _HAS_NUMEXPR = True
except ImportError:
    _HAS_NUMEXPR = False


__all__ = ['guess_dipole_moment', 'make_dipole', 'fill_dipole', 'calc_dip',
           'set_in_region', 'make_spherical_mask', 'xyz2lsrlp', 'dipole_map',
           'dipole_map_value']


# note that this global is used immutably (ie, not rc file configurable)
DEFAULT_STRENGTH = 1.0 / 3.0574e-5


def guess_dipole_moment(b, r=2.0, strength=DEFAULT_STRENGTH, cap_angle=40,
                        cap_ntheta=121, cap_nphi=121, plot=False):
    """guess dipole moment from a B field"""
    viscid.logger.warning("guess_dipole_moment doesn't seem to do better than "
                          "1.6 degrees, you may want to use cotr instead.")
    cap = seed.SphericalCap(r=r, angle=cap_angle, ntheta=cap_ntheta,
                            nphi=cap_nphi)
    b_cap = interp_trilin(b, cap)

    # FIXME: this doesn't get closer than 1.6 deg @ (theta, mu) = (0, 7.5)
    #        so maybe the index is incorrect somehow?
    idx = np.argmax(viscid.magnitude(b_cap).data)
    pole = cap.points()[:, idx]
    # FIXME: it should be achievabe to get strength from the magimum magnitude,
    #        up to the direction
    pole = strength * pole / np.linalg.norm(pole)
    # # not sure where 0.133 comes from, this is probably not correct
    # pole *= 0.133 * np.dot(pole, b_cap.data.reshape(-1, 3)[idx, :]) * r**3

    if plot:
        from viscid.plot import vpyplot as vlt
        from matplotlib import pyplot as plt
        vlt.plot(viscid.magnitude(b_cap))
        vlt.plot(viscid.magnitude(b_cap), style='contour', levels=10,
                 colors='k', colorbar=False, ax=plt.gca())
        vlt.show()
    return pole

def make_dipole(m=(0, 0, -DEFAULT_STRENGTH), strength=None, l=None, h=None,
                n=None, center='cell', dtype='f8', twod=False,
                nonuniform=False, crd_system='gse', name='b'):
    """Generate a dipole field with magnetic moment m [x, y, z]"""
    if l is None:
        l = [-5] * 3
    if h is None:
        h = [5] * 3
    if n is None:
        n = [256] * 3

    if center.strip().lower() == 'cell':
        n = [ni + 1 for ni in n]

    x = np.array(np.linspace(l[0], h[0], n[0]), dtype=dtype)
    y = np.array(np.linspace(l[1], h[1], n[1]), dtype=dtype)
    z = np.array(np.linspace(l[2], h[2], n[2]), dtype=dtype)
    if twod:
        y = np.array(np.linspace(-0.1, 0.1, 2), dtype=dtype)

    if nonuniform:
        z += 0.01 * ((h[2] - l[2]) / n[2]) * np.sin(np.linspace(0, np.pi, n[2]))

    B = field.empty([x, y, z], nr_comps=3, name=name, center=center,
                    layout='interlaced', dtype=dtype)
    B.set_info('crd_system', viscid.as_crd_system(crd_system))
    B.set_info('cotr', viscid.dipole_moment2cotr(m, crd_system=crd_system))
    return fill_dipole(B, m=m, strength=strength)

def fill_dipole(B, m=(0, 0, -DEFAULT_STRENGTH), strength=None, mask=None):
    """set B to a dipole with magnetic moment m

    Args:
        B (Field): Field to fill with a dipole
        m (ndarray, or datetime64-like): Description
        strength (float): if given, rescale the dipole moment
            even if it was given explicitly
        mask (Field): boolean field as mask, B will be filled where
            the mask is True

    Returns:
        Field: B
    """
    # FIXME: should really be taking the curl of a vector field
    if mask:
        Bdip = field.empty_like(B)
    else:
        Bdip = B

    # Xcc, Ycc, Zcc = B.get_crds_cc(shaped=True)  # pylint: disable=W0612
    Xv, Yv, Zv = B.get_crds_vector(shaped=True)  # pylint: disable=W0612
    _crd_lst = [[_x, _y, _z] for _x, _y, _z in zip(Xv, Yv, Zv)]

    dtype = B.dtype
    one = np.array([1.0], dtype=dtype)  # pylint: disable=W0612
    three = np.array([3.0], dtype=dtype)  # pylint: disable=W0612
    if viscid.is_datetime_like(m):
        m = viscid.get_dipole_moment(m, crd_system=B)
    else:
        m = np.asarray(m, dtype=dtype)

    if strength is not None:
        m = (strength / np.linalg.norm(m)) * m
    mx, my, mz = m  # pylint: disable=W0612

    # geneate a dipole field for the entire grid
    # Note: this is almost the exact same as calc_dip, but since components
    # are done one-at-a-time, it requires less memory since it copies the
    # result of each component into Bdip separately
    if _HAS_NUMEXPR:
        for i, cn in enumerate("xyz"):
            _X, _Y, _Z = _crd_lst[i]
            _XI = _crd_lst[i][i]
            _mi = m[i]
            rsq = ne.evaluate("_X**2 + _Y**2 + _Z**2")  # pylint: disable=W0612
            mdotr = ne.evaluate("mx * _X + my * _Y + mz * _Z")  # pylint: disable=W0612
            Bdip[cn] = ne.evaluate("((three * _XI * mdotr / rsq) - _mi) / rsq**1.5")
    else:
        for i, cn in enumerate("xyz"):
            _X, _Y, _Z = _crd_lst[i]
            _XI = _crd_lst[i][i]
            _mi = m[i]
            rsq = _X**2 + _Y**2 + _Z**2
            mdotr = mx * _X + my * _Y + mz * _Z
            Bdip[cn] = ((three * _XI * mdotr / rsq) - _mi) / rsq**1.5

    if mask:
        B.data[...] = np.choose(mask.astype('i'), [B, Bdip])
    return B

def calc_dip(pts, m=(0, 0, -DEFAULT_STRENGTH), strength=None, crd_system='gse',
             dtype=None):
    """Calculate a dipole field at various points

    Args:
        pts (ndarray): Nx3 array of points at which to calculate the
            dipole. Should use the same crd system as `m`
        m (sequence, datetime): dipole moment
        strength (None, float): If given, rescale m to this magnitude
        crd_system (str): Something from which cotr can divine the
            coordinate system for both `pts` and `m`. This is only used
            if m is given as a datetime and we need to figure out the
            dipole moment at a given time in a given crd system
        dtype (str, np.dtype): dtype of the result, defaults to
            the same datatype as `pts`

    Returns:
        ndarray: Nx3 dipole field vectors for N points
    """
    pts = np.asarray(pts, dtype=dtype)
    if len(pts.shape) == 1:
        pts = pts.reshape(1, 3)
        single_pt = True
    else:
        single_pt = False

    if dtype is None:
        dtype = pts.dtype

    one = np.array([1.0], dtype=dtype)  # pylint: disable=W0612
    three = np.array([3.0], dtype=dtype)  # pylint: disable=W0612
    if viscid.is_datetime_like(m):
        m = viscid.get_dipole_moment(m, crd_system=crd_system)
    else:
        m = np.asarray(m, dtype=dtype)

    if strength is not None:
        m = (strength / np.linalg.norm(m)) * m
    mx, my, mz = m  # pylint: disable=W0612

    m = m.reshape(1, 3)

    # geneate a dipole field for the entire grid
    # Note: this is almost the same as fill_dipole, but all components
    #       are calculated simultaneously, and so this uses more memory
    if _HAS_NUMEXPR:
        _X, _Y, _Z = pts.T
        rsq = ne.evaluate("_X**2 + _Y**2 + _Z**2")  # pylint: disable=W0612
        mdotr = ne.evaluate("mx * _X + my * _Y + mz * _Z")  # pylint: disable=W0612
        Bdip = ne.evaluate("((three * pts * mdotr / rsq) - m) / rsq**1.5")
    else:
        _X, _Y, _Z = pts.T
        rsq = _X**2 + _Y**2 + _Z**2
        mdotr = mx * _X + my * _Y + mz * _Z
        Bdip = ((three * pts * mdotr / rsq) - m) / rsq**1.5

    if single_pt:
        Bdip = Bdip[0, :]

    return Bdip

def set_in_region(a, b, alpha=1.0, beta=1.0, mask=None, out=None):
    """set `ret = alpha * a + beta * b` where mask is True"""
    alpha = np.asarray(alpha, dtype=a.dtype)
    beta = np.asarray(beta, dtype=a.dtype)
    a_dat = a.data if isinstance(a, viscid.field.Field) else a
    b_dat = b.data if isinstance(b, viscid.field.Field) else b
    b = None

    if _HAS_NUMEXPR:
        vals = ne.evaluate("alpha * a_dat + beta * b_dat")
    else:
        vals = alpha * a_dat + beta * b_dat
    a_dat = b_dat = None

    if out is None:
        out = field.empty_like(a)

    if mask is None:
        out.data[...] = vals
    else:
        if hasattr(mask, "nr_comps") and mask.nr_comps:
            mask = mask.as_centered(a.center).as_layout(a.layout)
        try:
            out.data[...] = np.choose(mask, [out.data, vals])
        except ValueError:
            out.data[...] = np.choose(mask.data.reshape(list(mask.sshape) + [1]),
                                      [out.data, vals])
    return out

def make_spherical_mask(fld, rmin=0.0, rmax=None, rsq=None):
    """make a mask that is True between rmin and rmax"""
    if rmax is None:
        rmax = np.sqrt(0.9 * np.finfo('f8').max)

    if True and fld.nr_comps and fld.center.lower() in ('edge', 'face'):
        mask = np.empty(fld.shape, dtype='bool')
        Xv, Yv, Zv = fld.get_crds_vector(shaped=True)  # pylint: disable=W0612
        _crd_lst = [[_x, _y, _z] for _x, _y, _z in zip(Xv, Yv, Zv)]
        # csq = [c**2 for c in fld.get_crds_vector(shaped=True)]
        for i in range(3):
            rsq = np.sum([c**2 for c in _crd_lst[i]], axis=0)
            _slc = [slice(None)] * len(fld.shape)
            _slc[fld.nr_comp] = i
            mask[_slc] = np.bitwise_and(rsq >= rmin**2, rsq < rmax**2)
        return fld.wrap_field(mask, dtype='bool')
    else:
        rsq = np.sum([c**2 for c in fld.get_crds(shaped=True)], axis=0)
        mask = np.bitwise_and(rsq >= rmin**2, rsq < rmax**2)
        if fld.nr_comps:
            fld = fld['x']
        return fld.wrap_field(mask, dtype='bool')

def _precondition_pts(pts):
    """Make sure pts are a 2d ndarray with length 3 in 1st dim"""
    pts = np.asarray(pts)
    if len(pts.shape) == 1:
        pts = pts.reshape((3, 1))
    return pts

def xyz2lsrlp(pts, cotr=None, crd_system='gse'):
    """Ceovert x, y, z -> l-shell, r, lambda, phi [sm coords]

      - r, theta, phi = viscid.cart2sph(pts in x, y, z)
      - lambda = 90deg - theta
      - r = L cos^2(lambda)

    Args:
        pts (ndarray): 3xN for N (x, y, z) points
        cotr (None): if given, use cotr to perform mapping to / from sm
        crd_system (str): crd system of pts

    Returns:
        ndarray: 4xN array of N (l-shell, r, lamda, phi) points
    """
    pts = _precondition_pts(pts)
    crd_system = viscid.as_crd_system(crd_system)
    cotr = viscid.as_cotr(cotr)

    # pts -> sm coords
    pts_sm = cotr.transform(crd_system, 'sm', pts)
    del pts

    # sm xyz -> r theta phi
    pts_rlp = viscid.cart2sph(pts_sm)
    # theta -> lamda (latitude)
    pts_rlp[1, :] = 0.5 * np.pi - pts_rlp[1, :]
    # get the L-shell from lamda and r
    lshell = pts_rlp[0:1, :] / np.cos(pts_rlp[1:2, :])**2

    return np.concatenate([lshell, pts_rlp], axis=0)

def dipole_map(pts, r=1.0, cotr=None, crd_system='gse', as_spherical=False):
    """Map pts along an ideal dipole to radius r

    lambda = 90deg - theta; r = L cos^2(lambda)

    cos^2(lambda) = cos^2(lambda_0) * (r / r_0)

    Args:
        pts (ndarray): 3xN for N (x, y, z) points
        r (float): radius to map to
        cotr (None): if given, use cotr to perform mapping to / from sm
        crd_system (str): crd system of pts
        as_spherical(bool): if True, then the return array is
            (t, theta, phi) with theta in the range [0, 180] and phi
            [0, 360] (in degrees)

    Returns:
        ndarray: 3xN array of N (x, y, z) points all at a distance
            r_mapped from the center of the dipole
    """
    pts = _precondition_pts(pts)
    crd_system = viscid.as_crd_system(crd_system)
    cotr = viscid.as_cotr(cotr)

    lsrlp = xyz2lsrlp(pts, cotr=cotr, crd_system=crd_system)
    del pts

    # this masking causes trouble
    # lsrlp = np.ma.masked_where(r lsrlp[0:1, :], lsrlp)
    # lsrlp = np.ma.masked_where(np.array([[r] * 3]).T > lsrlp[0:1, :], lsrlp)

    # rlp: r, lamda (latitude), phi
    rlp_mapped = np.empty_like(lsrlp[1:, :])
    rlp_mapped[0, :] = r
    # root is determined by sign of latitude in sm?
    root = np.sign(lsrlp[2:3, :])
    rlp_mapped[1, :] = root * np.arccos(np.sqrt(r / lsrlp[0:1, :]))
    rlp_mapped[2, :] = lsrlp[3:4, :]
    del lsrlp

    rlp_mapped[1, :] = 0.5 * np.pi - rlp_mapped[1, :]    # lamda (latitude) -> theta

    if as_spherical:
        ret = rlp_mapped  # is now r, theta, phi
        ret[1:, :] = np.rad2deg(ret[1:, :])
        # rotate angle phi by 360 until ret[2, :] is all between 0 and 360
        ret[2, :] -= 360.0 * (ret[2, :] // 360.0)
    else:
        ret = cotr.transform('sm', crd_system, viscid.sph2cart(rlp_mapped))
    return ret

def dipole_map_value(fld, pts, r=1.0, fillna=None, cotr=None,
                     crd_system=None, interp_kind='linear'):
    """Map values assuming they're constant along ideal dipole lines

    Args:
        fld (Field): values to interpolate onto the mapped pts
        pts (ndarray): 3xN for N (x, y, z) points that will be mapped
        r (float): radius of resulting map
        cotr (None): if given, use cotr to perform mapping in sm
        crd_system (str): crd system of pts
        interp_kind (str): how to interpolate fld onto source points

    Returns:
        ndarray: ndarray of mapped values, one for each of the N points
    """
    if crd_system is None:
        crd_system = viscid.as_crd_system(fld, 'gse')

    if fld.is_spherical:
        # TODO: verify that crd_system works as expected for ionosphere
        #       fields (ie, meaning of +x and phi = 0)
        fld = viscid.as_spherefield(fld, order=('theta', 'phi'))['r=newaxis, ...']
    else:
        pass

    # pts should be shaped 3xNX*NY*NZ or similar such that the points
    # are in the same order as the flattened c-contiguous array
    mapped_pts = dipole_map(pts, r=r, cotr=cotr, crd_system=crd_system,
                            as_spherical=fld.is_spherical)
    ret = viscid.interp(fld, mapped_pts, kind=interp_kind, wrap=False)
    if fillna is not None:
        ret[np.isnan(ret)] = fillna
    return ret

def _main():
    crd_system = 'gse'
    print(viscid.get_dipole_moment_ang(dip_tilt=45.0, dip_gsm=0.0,
                                       crd_system=crd_system))
    print(viscid.get_dipole_moment_ang(dip_tilt=0.0, dip_gsm=45.0,
                                       crd_system=crd_system))
    print(viscid.get_dipole_moment_ang(dip_tilt=45.0, dip_gsm=45.0,
                                       crd_system=crd_system))

    print("---")
    ptsNP = np.array([[+2, -2, +2], [+2, -1, +2], [+2, 1, +2], [+2, 2, +2]]).T
    ptsSP = np.array([[+2, -2, -2], [+2, -1, -2], [+2, 1, -2], [+2, 2, -2]]).T

    ptsNN = np.array([[-2, -2, +2], [-2, -1, +2], [-2, 1, +2], [-2, 2, +2]]).T
    ptsSN = np.array([[-2, -2, -2], [-2, -1, -2], [-2, 1, -2], [-2, 2, -2]]).T

    mapped_ptsNP = dipole_map(ptsNP)
    mapped_ptsNN = dipole_map(ptsNN)
    mapped_ptsSP = dipole_map(ptsSP)
    mapped_ptsSN = dipole_map(ptsSN)

    try:
        from viscid.plot import vlab
        colors1 = np.array([(0.6, 0.2, 0.2),
                            (0.2, 0.2, 0.6),
                            (0.6, 0.6, 0.2),
                            (0.2, 0.6, 0.6)])
        colors2 = colors1 * 0.5

        vlab.points3d(ptsNP, scale_factor=0.4, color=tuple(colors1[0]))
        vlab.points3d(ptsNN, scale_factor=0.4, color=tuple(colors1[1]))
        vlab.points3d(ptsSP, scale_factor=0.4, color=tuple(colors1[2]))
        vlab.points3d(ptsSN, scale_factor=0.4, color=tuple(colors1[3]))

        vlab.points3d(mapped_ptsNP, scale_factor=0.4, color=tuple(colors2[0]))
        vlab.points3d(mapped_ptsNN, scale_factor=0.4, color=tuple(colors2[1]))
        vlab.points3d(mapped_ptsSP, scale_factor=0.4, color=tuple(colors2[2]))
        vlab.points3d(mapped_ptsSN, scale_factor=0.4, color=tuple(colors2[3]))

        b = make_dipole()

        vlab.plot_lines(viscid.calc_streamlines(b, mapped_ptsNP, ibound=0.5)[0])
        vlab.plot_lines(viscid.calc_streamlines(b, mapped_ptsNN, ibound=0.5)[0])

        vlab.show()

    except ImportError:
        print("Mayavi not installed, no 3D plots", file=sys.stderr)

if __name__ == "__main__":
    _main()
