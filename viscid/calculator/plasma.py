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
from viscid import logger
# from viscid.calculator import calc

__all__ = ["calc_psi", "calc_beta"]


def calc_psi(B, rev=False):
    """Calc Flux function (only valid in 2d)

    Parameters:
        B (VectorField): magnetic field, should only have two
            spatial dimensions so we can infer the symmetry dimension
        rev (bool): since this integration doesn't like going
            through undefined regions (like within 1 earth radius of
            the origin for openggcm), you can use this to start
            integrating from the opposite corner.

    Returns:
        ScalarField: 2-D scalar flux function

    Raises:
        ValueError: If B has <> 2 spatial dimensions

    """
    # TODO: if this is painfully slow, i bet just putting this exact
    # code in a cython module would make it a bunch faster, the problem
    # being that the loops are in python instead of some broadcasting
    # numpy type thing

    B = B.slice_reduce(":")

    # try to guess if a dim of a 3D field is invariant
    reduced_axes = []
    if B.nr_sdims > 2:
        slcs = [slice(None)] * B.nr_sdims
        for i, nxi in enumerate(B.sshape):
            if nxi <= 2:
                slcs[i] = 0
                reduced_axes.append(B.crds.axes[i])
        slcs.insert(B.nr_comp, slice(None))
        B = B[slcs]

    # ok, so the above didn't work... just nip out the smallest dim?
    if B.nr_sdims == 3:
        slcs = [slice(None)] * B.nr_sdims
        i = np.argmin(B.sshape)
        slcs[i] = 0
        reduced_axes.append(B.crds.axes[i])
        logger.warning("Tried to get the flux function of a 3D field. "
                       "I can't do that, so I'm\njust ignoring the {0} "
                       "dimension".format(reduced_axes[-1]))
        slcs.insert(B.nr_comp, slice(None))
        B = B[slcs]

    if B.nr_sdims != 2:
        raise ValueError("flux function only implemented for 2D fields")

    comps = ""
    for comp in "xyz":
        if comp in B.crds.axes:
            comps += comp
    # ex: comps = "yz", comp_inds = [1, 2]
    comp_inds = [dict(x=0, y=1, z=2)[comp] for comp in comps]

    # Note: what follows says y, z, but it has been generalized
    # to any two directions, so hy isn't necessarily hy, but it's
    # easier to see at a glance if it's correct using a specific
    # example
    ycc, zcc = B.get_crds(comps)
    comp_views = B.component_views()
    hy, hz = comp_views[comp_inds[0]], comp_views[comp_inds[1]]
    dy = ycc[1:] - ycc[:-1]
    dz = zcc[1:] - zcc[:-1]
    ny, nz = len(ycc), len(zcc)

    A = np.empty((ny, nz), dtype=B.dtype)

    if rev:
        A[-1, -1] = 0.0
        for i in range(ny - 2, -1, -1):
            A[i, -1] = A[i + 1, -1] - dy[i] * 0.5 * (hz[i, -1] + hz[i + 1, -1])

        for j in range(nz - 2, -1, -1):
            A[:, j] = A[:, j + 1] + dz[j] * 0.5 * (hy[:, j + 1] + hy[:, j])
    else:
        A[0, 0] = 0.0
        for i in range(1, ny):
            A[i, 0] = A[i - 1, 0] + dy[i - 1] * 0.5 * (hz[i, 0] + hz[i - 1, 0])

        for j in range(1, nz):
            A[:, j] = A[:, j - 1] - dz[j - 1] * 0.5 * (hy[:, j - 1] + hy[:, j])

    psi = field.wrap_field(A, B.crds, name="psi", center=B.center,
                           pretty_name=r"$\psi$", parents=[B])
    if reduced_axes:
        slc = "..., " + ", ".join("{0}=None".format(ax) for ax in reduced_axes)
        psi = psi[slc]
    return psi

def calc_beta(pp, B, scale=1.0):
    """Calc plasma beta (2.0 * p / B^2)

    Parameters:
        pp (ScalarField or ndarray): pressure
        B (VectorField): magnetic field
        scale (float, optional): overall scale factor

    Returns:
        ScalarField: Plasma beta

    Note:
        For OpenGGCM, where pp is in pPa and B is in nT, scale should
        be 40.0.
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
