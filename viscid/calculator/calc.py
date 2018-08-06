#!/usr/bin/env python
""" This has a mechanism to dispatch to different backends. You can change the
order of preference of the backend by changing default_backends.
TODO: this dispatch mechanism is very simple, maybe something a little more
flexable would be useful down the line? """

from __future__ import print_function
from itertools import count

import numpy as np

import viscid
from viscid import field
from viscid import logger
from viscid import verror
from viscid import seed
from viscid.compat import izip, OrderedDict

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

__all__ = ['neg', 'scale', 'add', 'diff', 'mul', 'axpby', 'relative_diff',
           'abs_diff', 'abs_val', 'abs_max', 'abs_min', 'magnitude', 'dot',
           'cross', 'grad', 'convective_deriv', 'div', 'curl', 'normalize',
           'project', 'project_vector', 'project_along_line', 'resample_lines',
           'integrate_along_line', 'integrate_along_lines', 'jacobian_at_point',
           'jacobian_at_ind', 'jacobian_eig_at_point', 'jacobian_eig_at_ind',
           'div_at_point', 'curl_at_point', 'extend_boundaries',
           'extend_boundaries_ndarr']


class Operation(object):
    default_backends = ["numexpr", "cython", "numpy"]

    _imps = None  # implementations
    opname = None
    short_name = None

    def __init__(self, name, short_name, implementations=(), doc=""):
        self.opname = name
        self.short_name = short_name
        self._imps = OrderedDict()
        self.add_implementations(implementations)
        setattr(self, "__doc__", doc)

    @property
    def __name__(self):
        return self.opname

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
    def __call__(self, a, b=None, **kwargs):
        ret = super(BinaryOperation, self).__call__(a, b, **kwargs)
        try:
            ret.name = "{0} {1} {2}".format(a.name, self.short_name, b.name)
        except AttributeError:
            try:
                ret.name = "{0} {1}".format(self.short_name, b.name)
            except AttributeError:
                try:
                    ret.name = "{0} {1}".format(self.short_name, a.name)
                except AttributeError:
                    ret.name = self.short_name
        return ret

neg = UnaryOperation("neg", "-", doc="Callable, calculates -a")
scale = BinaryOperation("scale", "*=", doc="Callable, scales a")
add = BinaryOperation("add", "+", doc="Callable, calculates a + b")
diff = BinaryOperation("diff", "-", doc="Callable, calculates a - b")
mul = BinaryOperation("mul", "*", doc="Callable, calculates a * b")
axpby = Operation("axpby", "+", doc="Callable, calculates a * x + b * y")
relative_diff = BinaryOperation("relative diff", "%-",
                                doc="Callable, calculates abs(a - b) / a")
abs_diff = BinaryOperation("abs diff", "|-|", doc="Callable, calculates abs(a - b)")
abs_val = UnaryOperation("abs val", "absval", doc="Callable, calculates abs(a)")
abs_max = Operation("abs max", "absmax", doc="Callable, calculates max(abs(a))")
abs_min = Operation("abs min", "absmin", doc="Callable, calculates min(abs(a))")
magnitude = UnaryOperation("magnitude", "magnitude",
                           doc="Callable, calculates L2 Norm of a vectors in a vector field")
dot = BinaryOperation("dot", "dot", doc="Callable, calculates a dot b")
cross = BinaryOperation("cross", "x", doc="Callable, calculates a cross b")
project = BinaryOperation("project", "dot mag",
                          doc="Callable, scalar projection of a onto b; a dot b / norm(b)")
normalize = UnaryOperation("normalize", "normalize",
                           doc="Callable, divide a vector field by its magnitude")
grad = UnaryOperation("grad", "grad", doc="Callable, gradient of a scalar field")
div = UnaryOperation("div", "div", doc="Callable, divergence of a vector field")
curl = UnaryOperation("curl", "curl", doc="Callable, curl of a vector field")

if has_numexpr:
    neg.add_implementation("numexpr", necalc.neg)
    scale.add_implementation("numexpr", necalc.scale)
    add.add_implementation("numexpr", necalc.add)
    diff.add_implementation("numexpr", necalc.diff)
    mul.add_implementation("numexpr", necalc.mul)
    axpby.add_implementation("numexpr", necalc.axpby)
    relative_diff.add_implementation("numexpr", necalc.relative_diff)
    abs_diff.add_implementation("numexpr", necalc.abs_diff)
    abs_val.add_implementation("numexpr", necalc.abs_val)
    abs_max.add_implementation("numexpr", necalc.abs_max)
    abs_min.add_implementation("numexpr", necalc.abs_min)
    magnitude.add_implementation("numexpr", necalc.magnitude)
    dot.add_implementation("numexpr", necalc.dot)
    cross.add_implementation("numexpr", necalc.cross)
    project.add_implementation("numexpr", necalc.project)
    normalize.add_implementation("numexpr", necalc.normalize)
    grad.add_implementation("numexpr", necalc.grad)
    div.add_implementation("numexpr", necalc.div)
    curl.add_implementation("numexpr", necalc.curl)

# numpy versions
neg.add_implementation("numpy", lambda a: -a)
scale.add_implementation("numpy", lambda a, b: np.asarray(a, dtype=b.dtype) * b)
add.add_implementation("numpy", lambda a, b: a + b)
diff.add_implementation("numpy", lambda a, b: a - b)
mul.add_implementation("numpy", lambda a, b: a * b)
axpby.add_implementation("numpy", lambda a, x, b, y: a * x + b * y)
relative_diff.add_implementation("numpy", lambda a, b: (a - b) / a)
abs_diff.add_implementation("numpy", lambda a, b: np.abs(a - b))
abs_val.add_implementation("numpy", np.abs)
abs_max.add_implementation("numpy", lambda a: np.max(np.abs(a)))
abs_min.add_implementation("numpy", lambda a: np.min(np.abs(a)))

def _magnitude_np(fld):
    return np.sqrt((np.sum(fld * fld, axis=fld.nr_comp)))
magnitude.add_implementation("numpy", _magnitude_np)

def _project_np(a, b):
    """ project a along b (a dot b / |b|) """
    return (np.sum(a * b, axis=b.nr_comp) /
            np.sqrt(np.sum(b * b, axis=b.nr_comp)))
project.add_implementation("numpy", _project_np)

def _normalize_np(a):
    """ normalize a vector field """
    shp0 = list(a.shape)
    shp0[a.nr_comp] = 1
    return a / np.linalg.norm(a, axis=a.nr_comp).reshape(shp0)
normalize.add_implementation("numpy", _normalize_np)

def _dot_np(fld_a, fld_b):
    """dot product of two vector fields"""
    if fld_a.nr_comp != fld_b.nr_comp:
        raise ValueError("field must have same layout (flat or interlaced)")
    return np.sum(fld_a * fld_b, axis=fld_a.nr_comp)
dot.add_implementation("numpy", _dot_np)

def _cross_np(fld_a, fld_b):
    """cross product of two vector fields"""
    ax, ay, az = fld_a.component_views()
    bx, by, bz = fld_b.component_views()
    prodx = ay * bz - az * by
    prody = -ax * bz + az * bx
    prodz = ax * by - ay * bx
    return fld_a.wrap([prodx, prody, prodz])
cross.add_implementation("numpy", _cross_np)

def _grad_np(fld, bnd=True):
    """2nd order centeral diff, 1st order @ boundaries if bnd"""
    if bnd:
        fld = viscid.extend_boundaries(fld, order=0, crd_order=0)

    if fld.iscentered("Cell"):
        crdx, crdy, crdz = fld.get_crds_cc(shaped=True)
        # divcenter = "Cell"
        # divcrds = coordinate.NonuniformCartesianCrds(fld.crds.get_clist(np.s_[1:-1]))
        # divcrds = fld.crds.slice_keep(np.s_[1:-1, 1:-1, 1:-1])
    elif fld.iscentered("Node"):
        crdx, crdy, crdz = fld.get_crds_nc(shaped=True)
        # divcenter = "Node"
        # divcrds = coordinate.NonuniformCartesianCrds(fld.crds.get_clist(np.s_[1:-1]))
        # divcrds = fld.crds.slice_keep(np.s_[1:-1, 1:-1, 1:-1])
    else:
        raise NotImplementedError("Can only do cell and node centered gradients")

    v = fld.data
    g = viscid.zeros(fld['x=1:-1, y=1:-1, z=1:-1'].crds, nr_comps=3)

    xp, xm = crdx[2:,  :,  :], crdx[:-2, :  , :  ]  # pylint: disable=bad-whitespace
    yp, ym = crdy[ :, 2:,  :], crdy[:  , :-2, :  ]  # pylint: disable=bad-whitespace
    zp, zm = crdz[ :,  :, 2:], crdz[:  , :  , :-2]  # pylint: disable=bad-whitespace

    vxp, vxm = v[2:  , 1:-1, 1:-1], v[ :-2, 1:-1, 1:-1]  # pylint: disable=bad-whitespace
    vyp, vym = v[1:-1, 2:  , 1:-1], v[1:-1,  :-2, 1:-1]  # pylint: disable=bad-whitespace
    vzp, vzm = v[1:-1, 1:-1, 2:  ], v[1:-1, 1:-1,  :-2]  # pylint: disable=bad-whitespace

    g['x'].data[...] = (vxp - vxm) / (xp - xm)
    g['y'].data[...] = (vyp - vym) / (yp - ym)
    g['z'].data[...] = (vzp - vzm) / (zp - zm)
    return g
grad.add_implementation("numpy", _grad_np)

def _div_np(fld, bnd=True):
    """2nd order centeral diff, 1st order @ boundaries if bnd"""
    if fld.iscentered("Face"):
        # dispatch fc div immediately since that does its own pre-processing
        return viscid.div_fc(fld, bnd=bnd)

    if bnd:
        fld = viscid.extend_boundaries(fld, order=0, crd_order=0)

    vx, vy, vz = fld.component_views()

    if fld.iscentered("Cell"):
        crdx, crdy, crdz = fld.get_crds_cc(shaped=True)
        divcenter = "Cell"
        # divcrds = coordinate.NonuniformCartesianCrds(fld.crds.get_clist(np.s_[1:-1]))
        divcrds = fld.crds.slice_keep(np.s_[1:-1, 1:-1, 1:-1])
    elif fld.iscentered("Node"):
        crdx, crdy, crdz = fld.get_crds_nc(shaped=True)
        divcenter = "Node"
        # divcrds = coordinate.NonuniformCartesianCrds(fld.crds.get_clist(np.s_[1:-1]))
        divcrds = fld.crds.slice_keep(np.s_[1:-1, 1:-1, 1:-1])
    else:
        raise NotImplementedError("Can only do cell and node centered divs")

    xp, xm = crdx[2:,  :,  :], crdx[:-2, :  , :  ]  # pylint: disable=bad-whitespace
    yp, ym = crdy[ :, 2:,  :], crdy[:  , :-2, :  ]  # pylint: disable=bad-whitespace
    zp, zm = crdz[ :,  :, 2:], crdz[:  , :  , :-2]  # pylint: disable=bad-whitespace

    vxp, vxm = vx[2:  , 1:-1, 1:-1], vx[ :-2, 1:-1, 1:-1]  # pylint: disable=bad-whitespace
    vyp, vym = vy[1:-1, 2:  , 1:-1], vy[1:-1,  :-2, 1:-1]  # pylint: disable=bad-whitespace
    vzp, vzm = vz[1:-1, 1:-1, 2:  ], vz[1:-1, 1:-1,  :-2]  # pylint: disable=bad-whitespace

    div_arr = ((vxp - vxm) / (xp - xm) + (vyp - vym) / (yp - ym) +
               (vzp - vzm) / (zp - zm))
    return field.wrap_field(div_arr, divcrds, name="div " + fld.name,
                            center=divcenter, time=fld.time, parents=[fld])
div.add_implementation("numpy", _div_np)

def _curl_np(fld, bnd=True):
    """2nd order centeral diff, 1st order @ boundaries if bnd"""
    if bnd:
        fld = viscid.extend_boundaries(fld, order=0, crd_order=0)

    vx, vy, vz = fld.component_views()

    if fld.iscentered("Cell"):
        crdx, crdy, crdz = fld.get_crds_cc(shaped=True)
        curlcenter = "cell"
        # curlcrds = coordinate.NonuniformCartesianCrds(fld.crds.get_clist(np.s_[1:-1]))
        curlcrds = fld.crds.slice_keep(np.s_[1:-1, 1:-1, 1:-1])
    elif fld.iscentered("Node"):
        crdx, crdy, crdz = fld.get_crds_nc(shaped=True)
        curlcenter = "node"
        # curlcrds = coordinate.NonuniformCartesianCrds(fld.crds.get_clist(np.s_[1:-1]))
        curlcrds = fld.crds.slice_keep(np.s_[1:-1, 1:-1, 1:-1])
    else:
        raise NotImplementedError("Can only do cell and node centered divs")

    xp, xm = crdx[2:,  :,  :], crdx[:-2, :  , :  ]  # pylint: disable=bad-whitespace
    yp, ym = crdy[ :, 2:,  :], crdy[:  , :-2, :  ]  # pylint: disable=bad-whitespace
    zp, zm = crdz[ :,  :, 2:], crdz[:  , :  , :-2]  # pylint: disable=bad-whitespace

    vxpy, vxmy = vx[1:-1, 2:  , 1:-1], vx[1:-1,  :-2, 1:-1]  # pylint: disable=bad-whitespace
    vxpz, vxmz = vx[1:-1, 1:-1, 2:  ], vx[1:-1, 1:-1,  :-2]  # pylint: disable=bad-whitespace

    vypx, vymx = vy[2:  , 1:-1, 1:-1], vy[ :-2, 1:-1, 1:-1]  # pylint: disable=bad-whitespace
    vypz, vymz = vy[1:-1, 1:-1, 2:  ], vy[1:-1, 1:-1,  :-2]  # pylint: disable=bad-whitespace

    vzpx, vzmx = vz[2:  , 1:-1, 1:-1], vz[ :-2, 1:-1, 1:-1]  # pylint: disable=bad-whitespace
    vzpy, vzmy = vz[1:-1, 2:  , 1:-1], vz[1:-1,  :-2, 1:-1]  # pylint: disable=bad-whitespace

    curl_x = (vzpy - vzmy) / (yp - ym) - (vypz - vymz) / (zp - zm)
    curl_y = -(vzpx - vzmx) / (xp - xm) + (vxpz - vxmz) / (zp - zm)
    curl_z = (vypx - vymx) / (xp - xm) - (vxpy - vxmy) / (yp - ym)

    return field.wrap_field([curl_x, curl_y, curl_z], curlcrds,
                            name="curl " + fld.name, fldtype="Vector",
                            center=curlcenter, time=fld.time,
                            parents=[fld])
curl.add_implementation("numpy", _curl_np)

def convective_deriv(a, b=None, bnd=True):
    r"""Compute (a \dot \nabla) b for vector fields a and b"""
    # [(B \dot \nabla) B]_j = B_i \partial_i B_j
    # FIXME: this is a lot of temporary arrays
    if bnd:
        if b is None:
            b = viscid.extend_boundaries(a, order=0, crd_order=0)
        else:
            b = viscid.extend_boundaries(b, order=0, crd_order=0)
    else:
        if b is None:
            b = a
        a = a['x=1:-1, y=1:-1, z=1:-1']

    if b.nr_comps > 1:
        diBj = [[None, None, None], [None, None, None], [None, None, None]]
        for j, jcmp in enumerate('xyz'):
            g = grad(b[jcmp], bnd=False)
            for i, icmp in enumerate('xyz'):
                diBj[i][j] = g[icmp]
        dest = viscid.zeros(a.crds, nr_comps=3)
        for i, icmp in enumerate('xyz'):
            for j, jcmp in enumerate('xyz'):
                dest[jcmp][...] += a[icmp] * diBj[i][j]
    else:
        dest = dot(a, grad(b, bnd=False))
    return dest

# native versions  NOTE: magnitude_native is really only for benchmarking
def _magnitude_native(fld):
    vx, vy, vz = fld.component_views()
    mag = np.empty_like(vx)
    for i in range(mag.shape[0]):
        for j in range(mag.shape[1]):
            for k in range(mag.shape[2]):
                mag[i, j, k] = np.sqrt(vx[i, j, k]**2 + vy[i, j, k]**2 +
                                       vz[i, j, k]**2)
    return vx.wrap(mag, context={"name": "{0} magnitude".format(fld.name)})

magnitude.add_implementation("native", _magnitude_native)

def project_vector(a, b):
    """Calculates the vector (a dot b_hat) * b_hat"""
    bnorm = normalize(b)
    # print(">>", type(bnorm), bnorm.shape, bnorm.nr_comps,
    #       "MIN", np.min(np.linalg.norm(bnorm, axis=b.nr_comp)),
    #       "MAX", np.max(np.linalg.norm(bnorm, axis=b.nr_comp)))
    return project(a, bnorm) * bnorm

def resample_lines(lines, factor=1, kind="linear"):
    """Resample a bunch of lines

    Args:
        lines (sequence): a list of 3xN ndarrays that are lines
        factor (float): how to resample. 2 will give a point at
            the midpoint of all points, 3 will give two points, etc.
            0.5 will leave half the number of points, etc. This should
            be > 0.0.
        kind (str): How to interpolate; must be a value recognized by
            :py:func:`scipy.interpolate.interp1d`

    Raises:
        ValueError: List of resampled 3xN ndarrays
    """
    # FIXME: this is wicked slow
    if factor <= 0:
        raise ValueError("factor should be > 0")
    elif factor == 1:
        return lines

    for i, line in enumerate(lines):
        nr_sdims, nold = line.shape
        nnew = np.ceil(factor * (nold - 1)) + 1
        newline = np.empty((nr_sdims, nnew), dtype=line.dtype)
        sold = np.linspace(0, 1, nold)
        snew = np.linspace(0, 1, nnew)

        try:
            # raise ImportError
            from scipy.interpolate import interp1d
            for j in range(nr_sdims):
                newline[j, :] = interp1d(sold, line[j, :], kind=kind)(snew)
        except ImportError:
            if kind != 'linear':
                viscid.logger.error("Scipy is required to do anything "
                                    "other than linear interpolation")
                raise
            for j in range(nr_sdims):
                newline[j, :] = np.interp(snew, sold, line[j, :])
        lines[i] = newline
    return lines

def project_along_line(line, fld, interp_kind='trilin'):
    """Project a Vector Field Parallel to a streamline

    Args:
        line (ndarray): 3xN of points along the
        fld (VectorField): Field to interpolate and project onto the
            line
        interp_kind (str): which interpolation to use, const or trilin
    """
    fld_on_verts = viscid.interp(fld, line, kind=interp_kind)
    # FIXME: here in lies a bug, which axis am I actually summing over?
    #        maybe that's not the bug, but there must be something wrong
    #        here as indicated by kris_scripts/work/iono_xi_mismatch/analytic_pot.py
    dsvec = line[:, 2:] - line[:, :-2]
    dsvec = np.concatenate([dsvec[:, 0:1], dsvec, dsvec[:, -2:-1]], axis=1)
    dsvec = dsvec / np.linalg.norm(dsvec, axis=0)
    return np.sum(fld_on_verts * dsvec.T, axis=1)

def integrate_along_line(line, fld, reduction="dot", mask_func=None,
                         interp_kind='trilin'):
    """Integrate the value of fld along a line

    Args:
        line (ndarray): 3xN ndarray
        fld (Field): Field to interpolate / integrate
        reduction (str): If fld is a vector field, what quantity to
            integrate. Can be "dot" to dot the vectors with ds along
            the line, or "mag" to integrate the magnitude.
        interp_kind (str): which interpolation to use, const or trilin

    Returns:
        a scalar with the same dtype as fld
    """
    return integrate_along_lines([line], fld, reduction=reduction,
                                 mask_func=mask_func, interp_kind=interp_kind)[0]

def integrate_along_lines(lines, fld, reduction="dot", mask_func=None,
                          interp_kind='trilin'):
    """Integrate the value of fld along a list of lines

    Args:
        lines (list): list of 3xN ndarrays, N needs not be the same for
            all lines
        fld (Field): Field to interpolate / integrate
        reduction (str): If fld is a vector field, what quantity to
            integrate. Can be "dot" to dot the vectors with ds along
            the line, or "mag" to integrate the magnitude.
        interp_kind (str): which interpolation to use, const or trilin

    Returns:
        ndarray with shape (len(lines), )
    """
    arr = np.zeros((len(lines),), dtype=fld.dtype)

    cum_n = np.cumsum([0] + [line.shape[1] for line in lines])
    all_verts = np.concatenate(lines, axis=1)
    fld_on_verts = viscid.interp(fld, all_verts, kind=interp_kind).data

    for i, start, stop in izip(count(), cum_n[:-1], cum_n[1:]):
        ds = np.linalg.norm(lines[i][:, 1:] - lines[i][:, :-1], axis=0)

        if len(fld_on_verts.shape) > 1:
            reduction = reduction.strip().lower()
            if reduction == "dot":
                dsvec = lines[i][:, 1:] - lines[i][:, :-1]
                dsvec = dsvec / np.linalg.norm(dsvec, axis=0)
                values = 0.5 * (fld_on_verts[start:stop - 1, :] +
                                fld_on_verts[start + 1:stop, :])
                values = values * dsvec.T
                if mask_func is not None:
                    values = np.ma.masked_where(mask_func(values), values)
                values = np.sum(values, axis=1)
            elif reduction in ["mag", "magnitude", "norm"]:
                mag = np.linalg.norm(fld_on_verts[start:stop], axis=1)
                values = 0.5 * (mag[start:stop - 1] + mag[start + 1:stop])
            else:
                raise ValueError("Unknown reduction: {0}".format(reduction))
        else:
            values = 0.5 * (fld_on_verts[start:stop - 1] +
                            fld_on_verts[start + 1:stop])

        arr[i] = np.sum(values * ds)

    return arr

def local_vector_points(B, x, y, z, dx=None, dy=None, dz=None):
    """Get B at 6 points surrounding X

    X = [x, y, z] with spacing [+/-dx, +/-dy, +/-dz]

    Args:
        B (VectorField): B field
        x (float, ndarray, list): x (single value)
        y (float, ndarray, list): y (single value)
        z (float, ndarray, list): z (single value)
        dx (float, optional): dx, one grid cell if None
        dy (float, optional): dy, one grid cell if None
        dz (float, optional): dz, one grid cell if None

    Returns:
        (bs, pts, dcrd)

        * bs (ndarary): shape (6, 3) where 0-3 -> Bx,By,Bz
          and 0-6 -> X-dx, X+dx, X-dy, X+dy, X-dz, X+dz
        * pts (ndarray): shape (6, 3); the location of the
          points of the bs, but this time, 0-3 -> x,y,z
        * dcrd (list): [dx, dy, dz]
    """
    assert has_cython  # if a problem, you need to build Viscid
    assert B.iscentered("Cell")

    x, y, z = [np.array(c).reshape(1, 1) for c in [x, y, z]]
    crds = B.get_crds("xyz")
    inds = [0] * len(crds)
    dcrd = [0] * len(crds)
    # This makes points in xyz order
    pts = np.tile([x, y, z], 6).reshape(3, -1).T
    for i, crd, loc, d in zip(count(), crds, [x, y, z], [dx, dy, dz]):
        inds[i] = cycalc.closest_ind(crd, loc)
        if d is None:
            dcrd[i] = crd[inds[i] + 1] - crd[inds[i]]
        else:
            dcrd[i] = d
        pts[2 * i + 1, i] += dcrd[i]
        pts[2 * i + 0, i] -= dcrd[i]
    bs = cycalc.interp_trilin(B, seed.Point(pts))
    # viscid.interact(banner="in local_vector_points")
    return bs, pts, dcrd

def jacobian_at_point(B, x, y, z, dx=None, dy=None, dz=None):
    """Get the Jacobian at a point

    If dx|dy|dz == None, then their set to the grid spacing
    at the point of interest

    Returns:
        The jacobian as a 3x3 ndarray. The result is in xyz order,
        in other words::

          [ [d_x Bx, d_y Bx, d_z Bx],
            [d_x By, d_y By, d_z By],
            [d_x Bz, d_y Bz, d_z Bz] ]
    """
    bs, _, dcrd = local_vector_points(B, x, y, z, dx, dy, dz)
    gradb = np.empty((3, 3), dtype=B.dtype)

    # bs is in xyz spatial order, but components are in xyz order
    # dcrd is in xyz order
    # gradb has xyz order for everything
    for i in range(3):
        gradb[i, 0] = (bs[1, i] - bs[0, i]) / (2.0 * dcrd[0])  # d_x Bi
        gradb[i, 1] = (bs[3, i] - bs[2, i]) / (2.0 * dcrd[1])  # d_y Bi
        gradb[i, 2] = (bs[5, i] - bs[4, i]) / (2.0 * dcrd[2])  # d_z Bi
    return gradb

def jacobian_at_ind(B, ix, iy, iz):
    """Get the Jacobian at index

    Returns:
        The jacobian as a 3x3 ndarray. The result is in xyz order,
        in other words::

          [ [d_x Bx, d_y Bx, d_z Bx],
            [d_x By, d_y By, d_z By],
            [d_x Bz, d_y Bz, d_z Bz] ]
    """
    bx, by, bz = B.component_views()
    x, y, z = B.get_crds("xyz")
    gradb = np.empty((3, 3), dtype=B.dtype)
    for i, bi in enumerate([bx, by, bz]):
        # d_x Bi, d_y Bi, d_z Bi
        gradb[i, 0] = (bi[ix + 1, iy, iz] - bi[ix - 1, iy, iz]) / (x[ix + 1] - x[ix - 1])
        gradb[i, 1] = (bi[ix, iy + 1, iz] - bi[ix, iy - 1, iz]) / (y[iy + 1] - y[iy - 1])
        gradb[i, 2] = (bi[ix, iy, iz + 1] - bi[ix, iy, iz - 1]) / (z[iz + 1] - z[iz - 1])
    return gradb

def jacobian_eig_at_point(B, x, y, z, dx=None, dy=None, dz=None):
    """Get the eigen vals/vecs of the jacobian

    Returns: evals, evecs (3x3 ndarray)
        The evec[:, i] corresponds to evals[i].
        Eigen vectors are returned in xyz order, aka
        evec[:, 0] is [x, y, z] for the 0th eigen vector
    """
    gradb = jacobian_at_point(B, x, y, z, dx, dy, dz)
    evals, evecs = np.linalg.eig(gradb)
    return evals, evecs

def jacobian_eig_at_ind(B, ix, iy, iz):
    """Get the eigen vals/vecs of the jacobian

    Returns: evals, evecs (3x3 ndarray)
        The evec[:, i] corresponds to evals[i].
        Eigen vectors are returned in xyz order, aka
        evec[:, 0] is [x, y, z] for the 0th eigen vector
    """
    gradb = jacobian_at_ind(B, ix, iy, iz)
    evals, evecs = np.linalg.eig(gradb)
    return evals, evecs

def div_at_point(A, x, y, z, dx=None, dy=None, dz=None):
    """Returns divergence at a point"""
    As, _, dcrd = local_vector_points(A, x, y, z, dx, dy, dz)
    d = 0.0
    for i in range(3):
        d += (As[2 * i + 1, i] - As[2 * i + 0, i]) / (2.0 * dcrd[i])
    return d

def curl_at_point(A, x, y, z, dx=None, dy=None, dz=None):
    """Returns curl at point as ndarray with shape (3,) xyz"""
    As, _, dcrd = local_vector_points(A, x, y, z, dx, dy, dz)
    c = np.zeros(3, dtype=A.dtype)

    # this is confusing: In As, first index is xyz, 2nd index is xyz
    #             xi+dxi  comp      xi-dxi   comp /   (2 * dxi)
    c[0] = ((As[2 * 1 + 1, 2] - As[2 * 1 + 0, 2]) / (2.0 * dcrd[1]) -
            (As[2 * 0 + 1, 1] - As[2 * 0 + 0, 1]) / (2.0 * dcrd[2]))
    c[1] = ((As[2 * 0 + 1, 0] - As[2 * 0 + 0, 0]) / (2.0 * dcrd[2]) -
            (As[2 * 2 + 1, 2] - As[2 * 2 + 0, 2]) / (2.0 * dcrd[0]))
    c[2] = ((As[2 * 2 + 1, 1] - As[2 * 2 + 0, 1]) / (2.0 * dcrd[0]) -
            (As[2 * 1 + 1, 0] - As[2 * 1 + 0, 0]) / (2.0 * dcrd[1]))
    return c

def extend_boundaries_ndarr(arr, nl=1, nh=1, axes='all', nr_comp=None,
                            order=1, invarient_dx=0.0):
    """Extend and pad boundaries of ndarray (leaves new corners @ 0.0)

    Args:
        arr (ndarray): Array to extend
        nl (int): extend this many cells in lower direction
        nh (int): extend this many cells in upper direction
        axes (list): list of ints that correspond to axes that should
            be extended. If nr_comp != None, then axes > nr_comp will
            be shifted by 1.
        nr_comp (int, None): index of shape that corresponds to vector
            component dimension
        order (int): 0 for repeating boundary values or 1 for linear
            extrapolation
        invarient_dx (float): if len(arr) == 1 and order == 1, then
            the array is extended with constant dx, using 0 here is
            synonymous with 0th order.

    Returns:
        ndarray: A new extended / padded ndarray
    """
    if nr_comp is not None:
        axes = [ax if ax < nr_comp else ax + 1 for ax in axes]

    shape = list(arr.shape)
    target_shape = [s + nl + nh if i in axes else s for i, s in enumerate(shape)]

    s0 = []
    for i, s in enumerate(shape):
        if i in axes:
            _nl = nl if nl else None
            _nh = -nh if nh else None
            s0.append(slice(_nl, _nh))
        else:
            s0.append(slice(None))

    v = np.zeros(target_shape, dtype=arr.dtype)

    v[tuple(s0)] = arr[...]

    for _, axi in enumerate(axes):
        ni = shape[axi] + nl
        dest_slc = [slice(None)] * len(v.shape)
        src_slc = [slice(None)] * len(v.shape)
        src_slcR = [slice(None)] * len(v.shape)

        if arr.shape[axi] < 2 and invarient_dx:
            for j in range(nl):
                dest_slc[axi] = slice(nl - 1 - j, nl - 1 - j + 1)
                src_slc[axi] = slice(nl - j, nl - j + 1)
                v[tuple(dest_slc)] = v[tuple(src_slc)] - invarient_dx
            for j in range(nh):
                dest_slc[axi] = slice(ni + j, ni + j + 1)
                src_slc[axi] = slice(ni - 1 + j, ni - 1 + j + 1)
                v[tuple(dest_slc)] = v[tuple(src_slc)] + invarient_dx
        elif order == 0:
            if nl:
                dest_slc[axi] = slice(None, nl)
                src_slc[axi] = slice(nl, nl + 1)
                v[tuple(dest_slc)] = v[tuple(src_slc)]
            if nh:
                dest_slc[axi] = slice(ni, None)
                src_slc[axi] = slice(ni - 1, ni)
                v[tuple(dest_slc)] = v[tuple(src_slc)]
        elif arr.shape[axi] < 2:
            for j in range(nl):
                dest_slc[axi] = slice(nl - 1 - j, nl - 1 - j + 1)
                src_slc[axi] = slice(nl - j, nl - j + 1)
                v[tuple(dest_slc)] = v[tuple(src_slc)] - invarient_dx
            for j in range(nh):
                dest_slc[axi] = slice(ni + j, ni + j + 1)
                src_slc[axi] = slice(ni - 1 + j, ni - 1 + j + 1)
                v[tuple(dest_slc)] = v[tuple(src_slc)] + invarient_dx
        elif order == 1:
            for j in range(nl):
                dest_slc[axi] = slice(nl - j - 1, nl - j)
                src_slc[axi] = slice(nl - j, nl - j + 1)
                src_slcR[axi] = slice(nl - j + 1, nl - j + 2)
                v[tuple(dest_slc)] = 2 * v[tuple(src_slc)] - v[tuple(src_slcR)]
            for j in range(nh):
                dest_slc[axi] = slice(ni + j, ni + j + 1)
                src_slc[axi] = slice(ni + j - 1, ni + j)
                src_slcR[axi] = slice(ni + j -2, ni + j - 1)
                v[tuple(dest_slc)] = 2 * v[tuple(src_slc)] - v[tuple(src_slcR)]
            # src_slc[i] = slice(-1, None)
            # dest_slc[i] = slice(-nh, None)
            # v[dest_slc] = arr[src_slc]
        else:
            raise ValueError("extend_boundaries can only be 0th or 1st order")
    return v

def extend_boundaries(fld, nl=1, nh=1, axes='all', nr_comp=None, order=1,
                      crd_order=1, crds=None, invarient_dx=0.0,
                      crd_invarient_dx=1.0):
    """Extend and pad boundaries of field (leaves new corners @ 0.0)

    Args:
        fld (Field): Field to extend
        nl (int): extend this many cells in lower direction
        nh (int): extend this many cells in upper direction
        axes (str, list): something like 'xyz', ['x', 'theta'], etc.
        nr_comp (int, None): index of shape that corresponds to vector
            component dimension
        order (int): extrapolation order for data; 0 for repeating
            boundary values or 1 for linear extrapolation
        crd_order (int): extrapolation order for crds; 0 for repeating
            boundary values or 1 for linear extrapolation
        crds (Coordinates): Use these coordinates, no extrapolate them
        invarient_dx (float): if fld has a single cell in a given
            dimension, and order == 1, the field is extended with
            constant dx, using 0 here is synonymous with 0th order
        invarient_dx (float): if fld has a single cell in a given
            dimension, and crd_order == 1, the coordinates in that
            dimension are extended with constant dx, using 0 here is
            synonymous with crd_order = 0.

    Returns:
        ndarray: A new extended / padded ndarray

    Warning:
        When using crd_order=0 (0 order hold), coordinates will be
        extended by repeating boundary values, so min_dx will be 0.
        Keep this in mind if dividing by dx.
    """
    arr_axes = fld.crds.axes
    axis_lookup = dict()
    for i, ax in enumerate(arr_axes):
        axis_lookup[i] = i
        axis_lookup[ax] = i
    if axes == 'all':
        axes = list([i for i, s in enumerate(fld.sshape)])
    if not isinstance(axes, (list, tuple, viscid.string_types)):
        axes = list(axes)
    axes = [axis_lookup[ax] for ax in axes]

    try:
        nr_comp = fld.nr_comp
    except TypeError:
        nr_comp = None

    new_dat = extend_boundaries_ndarr(fld.data, nl=nl, nh=nh, axes=axes,
                                      nr_comp=nr_comp, order=order,
                                      invarient_dx=invarient_dx)

    if crds is not None:
        new_crds = crds
    else:
        new_clist = []
        crds_nc = fld.crds.get_crds_nc()
        crds_cc = fld.crds.get_crds_cc()
        for ax, nc, cc in zip(fld.crds.axes, crds_nc, crds_cc):
            if fld.crds.axes.index(ax) in axes:
                new_nc = extend_boundaries_ndarr(nc, nl=nl, nh=nh, axes=[0],
                                                 order=crd_order,
                                                 invarient_dx=crd_invarient_dx)
                new_cc = extend_boundaries_ndarr(cc, nl=nl, nh=nh, axes=[0],
                                                 order=crd_order,
                                                 invarient_dx=crd_invarient_dx)
            else:
                new_nc = nc
                new_cc = cc
            new_clist.append((ax, new_nc, new_cc))

        crdtype = fld.crds.crdtype
        if 'nonuniform' not in crdtype:
            crdtype = crdtype.replace('uniform', 'nonuniform')
        new_crds = viscid.wrap_crds(crdtype, new_clist)

    return fld.wrap(new_dat, context=dict(crds=new_crds))

##
## EOF
##
