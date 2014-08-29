#!/usr/bin/env python
""" This has a mechanism to dispatch to different backends. You can change the
order of preference of the backend by changing default_backends.
TODO: this dispatch mechanism is very simple, maybe something a little more
flexable would be useful down the line? """

from __future__ import print_function
try:
    from collections import OrderedDict
except ImportError:
    from viscid.compat import OrderedDict

import numpy as np

from viscid import logger
from viscid import verror

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
