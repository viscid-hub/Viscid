#!/usr/bin/env python
""" some equations that are useful for plasmas """

from __future__ import print_function, division

import numpy as np
try:
    import numexpr as ne
    _HAS_NUMEXPR = True
except ImportError:
    _HAS_NUMEXPR = False

from viscid.calculator import calc


def calc_psi(B):
    """Calc Flux function (only valid in 2d)

    Args:
        B (VectorField): magnetic field, should only have two
            spatial dimensions so we can infer the symmetry dimension

    Returns:
        ScalarField: 2-D scalar flux function

    Raises:
        ValueError: If B has <> 2 spatial dimensions

    """
    raise NotImplementedError()
    # if B.nr_sdims != 2:
    #     raise ValueError("")
    # crd_z, crd_y = B.get_crds_nc()
    # comps = B.component_views()
    # hz, hy = comps[2], comps[1]
    # crd_z, crd_y = hz.get_crds_nc(['z', 'y'])
    # dz = crd_z[1] - crd_z[0]
    # dy = crd_y[1] - crd_y[0]
    # nz, ny, _ = hy.shape

    # A = np.empty((nz, ny))
    # hz = hz.data.reshape(nz, ny)
    # hy = hy.data.reshape(nz, ny)
    # A[0, 0] = 0.0
    # for i in range(1, nz):
    #     A[i, 0] = A[i - 1, 0] + dz * (hy[i, 0] + hy[i - 1, 0]) / 2.0

    # for j in range(1, ny):
    #     A[:, j] = A[:, j - 1] - dy * (hz[:, j - 1] + hz[:, j]) / 2.0

    # return field.wrap_field("Scalar", "psi", self["hz"].crds, A,
    #                         center="Cell")

def calc_beta(pp, B, norm=1.0):
    """Calc plasma beta (2*p/B^2)

    Args:
        pp (ScalarField or ndarray): pressure
        B (VectorField): magnetic field
        norm (float, optional): overall scale factor

    Returns:
        ScalarField: Plasma beta

    """
    two = np.array([2.0], dtype=pp.dtype)
    bx, by, bz = B.component_views()
    if _HAS_NUMEXPR:
        ldict = dict(norm=norm, bx=bx, by=by, bz=bz,
                     pp=pp, two=two)
        result = ne.evaluate("norm * two * pp / sqrt(bx**2 + by**2 + bz**2)",
                             local_dict=ldict)
    else:
        result = norm * two * pp / np.sqrt(bx**2 + by**2 + bz**2)
    context = dict(name="beta", pretty_name=r"$\beta_{pl}$")
    return pp.wrap(result, context=context)

##
## EOF
##
