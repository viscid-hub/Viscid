#!/usr/bin/env python
""" some equations that are useful for plasmas """

from __future__ import print_function, division

import numpy as np
try:
    import numexpr as ne
    _HAS_NUMEXPR = True
except ImportError:
    _HAS_NUMEXPR = False

from viscid import field
# from viscid.calculator import calc


def calc_psi(B):
    """Calc Flux function (only valid in 2d)

    Parameters:
        B (VectorField): magnetic field, should only have two
            spatial dimensions so we can infer the symmetry dimension

    Returns:
        ScalarField: 2-D scalar flux function

    Raises:
        ValueError: If B has <> 2 spatial dimensions

    """
    # TODO: if this is painfully slow, i bet just putting this exact
    # code in a cython module would make it a bunch faster, the problem
    # being that the loops are in python instead of some broadcasting
    # numpy type thing

    if B.nr_sdims != 2:
        raise ValueError("flux function implemented for 2D fields")

    comps = ""
    for comp in "zyx":
        if comp in B.crds.axes:
            comps += comp
    # ex: comps = "zy", comp_inds = [2, 1]
    comp_inds = [dict(x=0, y=1, z=2)[comp] for comp in comps]

    # Note: what follows says y, z, but it has been generalized
    # to any two directions, so hy isn't necessarily hy, but it's
    # easier to see at a glance if it's correct using a specific
    # example
    zcc, ycc = B.get_crds_cc(comps)
    comp_views = B.component_views()
    hz, hy = comp_views[comp_inds[0]], comp_views[comp_inds[1]]
    dz = zcc[1:] - zcc[:-1]
    dy = ycc[1:] - ycc[:-1]
    nz, ny = len(zcc), len(ycc)

    A = np.empty((nz, ny), dtype=B.dtype)
    A[0, 0] = 0.0
    for i in range(1, nz):
        A[i, 0] = A[i - 1, 0] + dz[i - 1] * 0.5 * (hy[i, 0] + hy[i - 1, 0])

    for j in range(1, ny):
        A[:, j] = A[:, j - 1] - dy[j - 1] * 0.5 * (hz[:, j - 1] + hz[:, j])

    return field.wrap_field("Scalar", "psi", B.crds, A,
                            center="Cell", pretty_name=r"$\psi$")

def calc_beta(pp, B, scale=1.0):
    """Calc plasma beta (2*p/B^2)

    Parameters:
        pp (ScalarField or ndarray): pressure
        B (VectorField): magnetic field
        scale (float, optional): overall scale factor

    Returns:
        ScalarField: Plasma beta

    """
    two = np.array([2.0], dtype=pp.dtype)
    bx, by, bz = B.component_views()
    if _HAS_NUMEXPR:
        ldict = dict(scale=scale, bx=bx, by=by, bz=bz,
                     pp=pp, two=two)
        result = ne.evaluate("scale * two * pp / sqrt(bx**2 + by**2 + bz**2)",
                             local_dict=ldict)
    else:
        result = scale * two * pp / np.sqrt(bx**2 + by**2 + bz**2)
    context = dict(name="beta", pretty_name=r"$\beta_{pl}$")
    return pp.wrap(result, context=context)

##
## EOF
##
