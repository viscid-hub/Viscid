"""A collection of tools for dipoles"""
# pylint: disable=bad-whitespace

from __future__ import print_function, division

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


__all__ = ['get_dipole_moment_sm', 'guess_dipole_moment', 'get_dipole',
           'fill_dipole', 'set_in_region', 'make_spherical_mask']


DEFAULT_STRENGTH = -1.0 / 3.0574e-5


def get_dipole_moment_sm(strength=DEFAULT_STRENGTH, theta=0.0, mu=0.0,
                         crd_system='gse'):
    """Get dipole moment from sm given theta and mu angles"""
    # theta and mu are given by:
    #     http://jsoc.stanford.edu/doc/keywords/Chris_Russel/
    #     Geophysical%20Coordinate%20Transformations.htm
    # theta is the angle between GSE and GSM (+ is duskward)
    # mu is the dipole tilt angle from GSM to SM (+ is sunward)
    m_sm = np.array([0.0, 0.0, strength])
    theta = (np.pi / 180.0) * theta
    mu = (np.pi / 180.0) * mu

    if crd_system.strip().lower() == "mhd":
        theta *= -1
        mu *= -1
    elif crd_system.strip().lower() != "gse":
        raise ValueError("crd_system == {0}, not mhd or gse".format(crd_system))

    gsm2gse = np.array([[1.0,                   0.0,            0.0],
                        [0.0,        +np.cos(theta), -np.sin(theta)],
                        [0.0,        +np.sin(theta), +np.cos(theta)]]).T
    sm2gsm = np.array([[+np.cos(mu),            0.0, -np.sin(mu)],
                       [        0.0,            1.0,         0.0],
                       [+np.sin(mu),            0.0, +np.cos(mu)]]).T

    ret = np.dot(gsm2gse, np.dot(sm2gsm, m_sm))
    assert np.isclose(np.linalg.norm(ret), np.abs(strength))
    return ret

def guess_dipole_moment(b, r=2.0, strength=DEFAULT_STRENGTH, cap_angle=40,
                        cap_ntheta=121, cap_nphi=121, plot=False):
    """guess dipole moment from a B field"""
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
        from viscid.plot import mpl
        mpl.plot(viscid.magnitude(b_cap), show=True)
    return pole

def get_dipole(m=(0, 0, DEFAULT_STRENGTH), l=None, h=None, n=None,
               twod=False, dtype='f8', nonuniform=False):
    """Generate a dipole field with magnetic moment m [x, y, z]"""
    if l is None:
        l = [-5] * 3
    if h is None:
        h = [5] * 3
    if n is None:
        n = [256] * 3
    x = np.array(np.linspace(l[0], h[0], n[0]), dtype=dtype)
    y = np.array(np.linspace(l[1], h[1], n[1]), dtype=dtype)
    z = np.array(np.linspace(l[2], h[2], n[2]), dtype=dtype)
    if twod:
        y = np.array(np.linspace(-0.1, 0.1, 2), dtype=dtype)

    if nonuniform:
        z += 0.01 * ((h[2] - l[2]) / n[2]) * np.sin(np.linspace(0, np.pi, n[2]))

    B = field.empty([x, y, z], nr_comps=3, name="B", center='cell',
                    layout='interlaced', dtype=dtype)
    return fill_dipole(B, m=m)

def fill_dipole(B, m=(0, 0, -1 / 3.0574e-05), mask=None):
    """set B to a dipole with magnetic moment m"""
    # FIXME: should really be taking the curl of a vector field
    if mask:
        Bdip = field.empty_like(B)
    else:
        Bdip = B

    Xcc, Ycc, Zcc = B.get_crds_cc(shaped=True)  # pylint: disable=W0612

    dtype = B.dtype
    one = np.array([1.0], dtype=dtype)  # pylint: disable=W0612
    three = np.array([3.0], dtype=dtype)  # pylint: disable=W0612
    m = np.asarray(m, dtype=dtype)
    mx, my, mz = m  # pylint: disable=W0612

    # geneate a dipole field for the entire grid
    if _HAS_NUMEXPR:
        rsq = ne.evaluate("Xcc**2 + Ycc**2 + Zcc**2")  # pylint: disable=W0612
        mdotr = ne.evaluate("mx * Xcc + my * Ycc + mz * Zcc")  # pylint: disable=W0612
        Bdip['x'] = ne.evaluate("((three * Xcc * mdotr / rsq) - mx) / rsq**1.5")
        Bdip['y'] = ne.evaluate("((three * Ycc * mdotr / rsq) - my) / rsq**1.5")
        Bdip['z'] = ne.evaluate("((three * Zcc * mdotr / rsq) - mz) / rsq**1.5")
    else:
        rsq = Xcc**2 + Ycc**2 + Zcc**2
        mdotr = mx * Xcc + my * Ycc + mz * Zcc
        Bdip['x'] = ((three * Xcc * mdotr / rsq) - mx) / rsq**1.5
        Bdip['y'] = ((three * Ycc * mdotr / rsq) - my) / rsq**1.5
        Bdip['z'] = ((three * Zcc * mdotr / rsq) - mz) / rsq**1.5

    if mask:
        B.data[...] = np.choose(mask, [B, Bdip])
    return B

def set_in_region(a, b, alpha=1.0, beta=1.0, mask=None, out=None):
    """set `ret = alpha * a + beta * b` where mask is True"""
    alpha = np.asarray(alpha, dtype=a.dtype)
    beta = np.asarray(beta, dtype=a.dtype)
    a_dat, b_dat, b = a.data, b.data, None

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
        out.data[...] = np.choose(mask, [a.data, vals])
    return out

def make_spherical_mask(fld, rmin=0.0, rmax=None, rsq=None):
    """make a mask that is True between rmin and rmax"""
    if rmax is None:
        rmax = np.sqrt(0.9 * np.finfo('f8').max)
    rsq = np.sum([c**2 for c in fld.get_crds(shaped=True)], axis=0)
    mask = np.bitwise_and(rsq >= rmin**2, rsq < rmax**2)
    if fld.nr_comps:
        fld = fld['x']
    return fld.wrap_field(mask, dtype='bool')

def _main():
    crd_system = 'gse'
    print(get_dipole_moment_sm(theta=45.0, mu=0.0, crd_system=crd_system))
    print(get_dipole_moment_sm(theta=0.0, mu=45.0, crd_system=crd_system))
    print(get_dipole_moment_sm(theta=45.0, mu=45.0, crd_system=crd_system))

if __name__ == "__main__":
    _main()
