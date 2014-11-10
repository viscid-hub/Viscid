#!/usr/bin/env python
""" This has a mechanism to dispatch to different backends. You can change the
order of preference of the backend by changing default_backends.
TODO: this dispatch mechanism is very simple, maybe something a little more
flexable would be useful down the line? """

from __future__ import print_function
from itertools import count
try:
    from collections import OrderedDict
except ImportError:
    from viscid.compat import OrderedDict

import numpy as np

from viscid import logger
from viscid import verror
from viscid.calculator import seed

try:
    from viscid.calculator import cycalc
    has_cython = True
except ImportError:
    has_cython = False

try:
    from viscid.calculator import necalc
    has_numexpr = True
except ImportError as e:
    has_numexpr = False

class Operation(object):
    default_backends = ["numexpr", "cython", "numpy"]

    _imps = None  # implementations
    opname = None
    short_name = None

    def __init__(self, name, short_name, implementations=[]):
        self.opname = name
        self.short_name = short_name
        self._imps = OrderedDict()
        self.add_implementations(implementations)

    def add_implementation(self, name, func):
        self._imps[name] = func

    def add_implementations(self, implementations):
        for name, func in implementations:
            self.add_implementation(name, func)

    def _get_imp(self, preferred, only=False):
        if not isinstance(preferred, (list, tuple)):
            if preferred is None:
                preferred = list(self._imps.keys())
            else:
                preferred = [preferred]

        for name in preferred:
            if name in self._imps:
                return self._imps[name]

        msg = "{0} :: {1}".format(self.opname, preferred)
        if only:
            raise verror.BackendNotFound(msg)
        logger.info("No preferred backends available: " + msg)

        for name in self.default_backends:
            if name in self._imps:
                return self._imps[name]

        if len(self._imps) == 0:
            raise verror.BackendNotFound("No backends available")
        return list(self._imps.values())[0]

    def __call__(self, *args, **kwargs):
        preferred = kwargs.pop("preferred", None)
        only = kwargs.pop("only", False)
        func = self._get_imp(preferred, only)
        return func(*args, **kwargs)


class UnaryOperation(Operation):
    def __call__(self, a, **kwargs):
        ret = super(UnaryOperation, self).__call__(a, **kwargs)
        ret.name = "{0} {1}".format(self.short_name, a.name)
        return ret

class BinaryOperation(Operation):
    def __call__(self, a, b, **kwargs):
        ret = super(BinaryOperation, self).__call__(a, b, **kwargs)
        ret.name = "{0} {1} {2}".format(a.name, self.short_name, b.name)
        return ret

add = BinaryOperation("add", "+")
diff = BinaryOperation("diff", "-")
mul = BinaryOperation("mul", "*")
relative_diff = BinaryOperation("relative diff", "%-")
abs_diff = BinaryOperation("abs diff", "|-|")
abs_val = UnaryOperation("abs val", "absval")
abs_max = Operation("abs max", "absmax")
abs_min = Operation("abs min", "absmin")
magnitude = UnaryOperation("magnitude", "magnitude")
dot = BinaryOperation("dot", "dot")
cross = BinaryOperation("cross", "x")
div = UnaryOperation("div", "div")
curl = UnaryOperation("curl", "curl")

if has_numexpr:
    add.add_implementation("numexpr", necalc.add)
    diff.add_implementation("numexpr", necalc.diff)
    mul.add_implementation("numexpr", necalc.mul)
    relative_diff.add_implementation("numexpr", necalc.relative_diff)
    abs_diff.add_implementation("numexpr", necalc.abs_diff)
    abs_val.add_implementation("numexpr", necalc.abs_val)
    abs_max.add_implementation("numexpr", necalc.abs_max)
    abs_min.add_implementation("numexpr", necalc.abs_min)
    magnitude.add_implementation("numexpr", necalc.magnitude)
    dot.add_implementation("numexpr", necalc.dot)
    cross.add_implementation("numexpr", necalc.cross)
    div.add_implementation("numexpr", necalc.div)
    curl.add_implementation("numexpr", necalc.curl)

# numpy versions
add.add_implementation("numpy", lambda a, b: a + b)
diff.add_implementation("numpy", lambda a, b: a - b)
mul.add_implementation("numpy", lambda a, b: a * b)
relative_diff.add_implementation("numpy", lambda a, b: (a -b) / a)
abs_diff.add_implementation("numpy", lambda a, b: np.abs(a - b))
abs_val.add_implementation("numpy", np.abs)
abs_max.add_implementation("numpy", lambda a: np.max(np.abs(a)))
abs_min.add_implementation("numpy", lambda a: np.min(np.abs(a)))

def magnitude_np(fld):
    vx, vy, vz = fld.component_views()
    return np.sqrt((vx**2) + (vy**2) + (vz**2))

magnitude.add_implementation("numpy", magnitude_np)

# native versions
def magnitude_native(fld):
    vx, vy, vz = fld.component_views()
    mag = np.empty_like(vx)
    for i in range(mag.shape[0]):
        for j in range(mag.shape[1]):
            for k in range(mag.shape[2]):
                mag[i, j, k] = np.sqrt(vx[i, j, k]**2 + vy[i, j, k]**2 + \
                                       vz[i, j, k]**2)
    return vx.wrap(mag, context={"name": "{0} magnitude".format(fld.name)})

magnitude.add_implementation("native", magnitude_native)

def local_vector_points(B, z, y, x, dz=None, dy=None, dx=None):
    """
    Get B at 6 points surrounding X = [z, y, x] with
    spacing [+/-dz, +/-dy, +/-dx]

    If dz|dy|ix == None, then their set to the grid spacing
    at the point of interest

    Returns:
        bs: ndarray with shape (6, 3) where 0-3 -> Bx,By,Bz
            and 0-6 -> X-dz, X+dz, X-dy, X+dy, X-dx, X+dx
        pts: ndarray with shape 6, 3 of the location of the
            points of the bs, but this time, 0-3 -> z,y,x
        dcrd: list of dz, dy, dx
    """
    assert has_cython  # if a problem, you need to build Viscid
    assert B.iscentered("Cell")

    z, y, x = [np.array(c).reshape(1, 1) for c in [z, y, x]]
    crds = B.get_crds("zyx")
    inds = [0] * len(crds)
    dcrd = [0] * len(crds)
    # This makes points in zyx order
    pts = np.tile([z, y, x], 6).reshape(3, -1).T
    for i, crd, loc, d in zip(count(), crds, [z, y, x], [dz, dy, dx]):
        inds[i] = cycalc.closest_ind(crd, loc)
        if d is None:
            dcrd[i] = crd[inds[i] + 1] - crd[inds[i]]
        else:
            dcrd[i] = d
        pts[2 * i + 1, i] += dcrd[i]
        pts[2 * i + 0, i] -= dcrd[i]
    bs = cycalc.interp_trilin(B, seed.Point(pts))
    # import code; code.interact("in local_vector_points", local=locals())
    return bs, pts, dcrd

def jacobian_at_point(B, z, y, x, dz=None, dy=None, dx=None):
    """Get the Jacobian at a point

    If dz|dy|ix == None, then their set to the grid spacing
    at the point of interest

    Returns: The jacobian as a 3x3 ndarray
        The result is in xyz order, in other words:
        [ [d_x Bx, d_y Bx, d_z Bx],
          [d_x By, d_y By, d_z By],
          [d_x Bz, d_y Bz, d_z Bz] ]
    """
    bs, _, dcrd = local_vector_points(B, z, y, x, dz, dy, dx)
    gradb = np.empty((3, 3), dtype=B.dtype)

    # bs is in zyx spatial order, but components are in xyz order
    # dcrd is in zyx order
    # gradb has xyz order for everything
    for i in range(3):
        gradb[i, 0] = (bs[5, i] - bs[4, i]) / (2.0 * dcrd[2])  # d_x Bi
        gradb[i, 1] = (bs[3, i] - bs[2, i]) / (2.0 * dcrd[1])  # d_y Bi
        gradb[i, 2] = (bs[1, i] - bs[0, i]) / (2.0 * dcrd[0])  # d_z Bi
    return gradb

def jacobian_at_ind(B, iz, iy, ix):
    """Get the Jacobian at index

    Returns: The jacobian as a 3x3 ndarray
        The result is in xyz order, in other words:
        [ [d_x Bx, d_y Bx, d_z Bx],
          [d_x By, d_y By, d_z By],
          [d_x Bz, d_y Bz, d_z Bz] ]
    """
    bx, by, bz = B.component_views()
    z, y, x = B.get_crds("zyx")
    gradb = np.empty((3, 3), dtype=B.dtype)
    for i, bi in enumerate([bz, by, bx]):
        gradb[i, 0] = (bi[iz, iy, ix + 1] - bi[iz, iy, ix - 1]) / (x[ix + 1] - x[ix - 1])  # d_x Bi
        gradb[i, 1] = (bi[iz, iy + 1, iz] - bi[iz, iy - 1, iz]) / (y[iy + 1] - y[iy - 1])  # d_y Bi
        gradb[i, 2] = (bi[iz + 1, iy, ix] - bi[iz - 1, iy, ix]) / (z[iz + 1] - z[iz - 1])  # d_z Bi
    return gradb

def jacobian_eig_at_point(B, z, y, x, dz=None, dy=None, dx=None):
    """Get the eigen vals/vecs of the jacobian

    Returns: evals, evecs (3x3 ndarray)
        The evec[:, i] corresponds to evals[i].
        Eigen vectors are returned in zyx order, aka
        evec[:, 0] is [z, y, x] for the 0th eigen vector
    """
    gradb = jacobian_at_point(B, z, y, x, dz, dy, dx)
    evals, evecs = np.linalg.eig(gradb)
    # change eigen vectors xyz -> zyx
    for i in range(3):
        evecs[:, i] = evecs[::-1, i]
    return evals, evecs

def jacobian_eig_at_ind(B, iz, iy, ix):
    """Get the eigen vals/vecs of the jacobian

    Returns: evals, evecs (3x3 ndarray)
        The evec[:, i] corresponds to evals[i].
        Eigen vectors are returned in zyx order, aka
        evec[:, 0] is [z, y, x] for the 0th eigen vector
    """
    gradb = jacobian_at_ind(B, iz, iy, ix)
    evals, evecs = np.linalg.eig(gradb)
    # change eigen vectors xyz -> zyx
    for i in range(3):
        evecs[:, i] = evecs[::-1, i]
    return evals, evecs

def div_at_point(A, z, y, x, dz=None, dy=None, dx=None):
    """Returns divergence at a point"""
    As, _, dcrd = local_vector_points(A, z, y, x, dz, dy, dx)
    d = 0.0
    for i in range(3):
        d += (As[2 * i + 1, i] - As[2 * i + 0, i]) / (2.0 * dcrd[i])
    return d

def curl_at_point(A, z, y, x, dz=None, dy=None, dx=None):
    """Returns curl at point as ndarray with shape (3,) xyz"""
    As, _, dcrd = local_vector_points(A, z, y, x, dz, dy, dx)
    c = np.zeros(3, dtype=A.dtype)

    # this is confusing: In As, first index is zyx, 2nd index is xyz
    #             xi+dxi  comp      xi-dxi   comp /   (2 * dxi)
    c[0] =  (As[2 * 1 + 1, 2] - As[2 * 1 + 0, 2]) / (2.0 * dcrd[1]) - \
            (As[2 * 0 + 1, 1] - As[2 * 0 + 0, 1]) / (2.0 * dcrd[0])
    c[1] = -(As[2 * 2 + 1, 2] - As[2 * 2 + 0, 2]) / (2.0 * dcrd[2]) + \
            (As[2 * 0 + 1, 0] - As[2 * 0 + 0, 0]) / (2.0 * dcrd[0])
    c[2] =  (As[2 * 2 + 1, 1] - As[2 * 2 + 0, 1]) / (2.0 * dcrd[2]) - \
            (As[2 * 1 + 1, 0] - As[2 * 1 + 0, 0]) / (2.0 * dcrd[1])
    return c

def closest1d_ind(arr, val):
    """ DEPRECATED """
    #i = np.argmin(ne.evaluate("abs(arr - val)"))
    return np.argmin(np.abs(arr - val))

def closest1d_val(arr, val):
    """ DEPRECATED """
    #i = np.argmin(ne.evaluate("abs(arr - val)"))
    i = np.argmin(np.abs(arr - val))
    return arr[i]

def nearest_val(fld, point):
    """ DEPRECATED
    find value of field closest to point
    """
    x, y, z = point
    xind = closest1d_ind(fld.crds['x'], x)
    yind = closest1d_ind(fld.crds['y'], y)
    zind = closest1d_ind(fld.crds['z'], z)
    return fld[zind, yind, xind]

##
## EOF
##
