#!/usr/bin/env python
# This has a mechanism to dispatch to different backends. You can change the
# order of preference of the backend by changing default_backends.
# TODO: this dispatch mechanism is very simple, maybe something a little more
# flexable would be useful down the line?

from __future__ import print_function
from warnings import warn

import numpy as np

from .. import field
from .. import vutil

try:
    from . import cycalc
    has_cython = True
except ImportError:
    has_cython = False

try:
    from . import necalc
    has_numexpr = True
except ImportError:
    has_numexpr = False

_installed_backends = {"cython": has_cython,
                       "numexpr": has_numexpr,
                       "numpy": True,
                       "native": True,  # now this is a silly one
                      }

# list of default backends to use in order of preference
# this can be changed, with the warning that it will all users of this module
default_backends = ["cython", "numexpr", "numpy"]

# TODO: this is kind of a silly mechanism
def _check_backend(preferred, implemented):
    """ preferred should be a list of backends in order of preference or None
    to use the default list. The first preferred backend that is in implemented
    AND is installed is returned. If none of the preferred are usable, the list
    of implemented backends is checked in order... so the calling function should
    think about the order of the implemented list too... for instance, try
    numexpr before numpy
    """
    if preferred == None:
        preferred = default_backends
    if not isinstance(preferred, (list, tuple)):
        preferred = [preferred]

    # go through preferred backends
    for backend in preferred:
        if backend in implemented:
            try:
                if _installed_backends[backend]:
                    return backend
            except KeyError:
                vutil.warn("Unknown backend: {0}".format(backend))

    for backend in implemented:
        try:
            if _installed_backends[backend]:
                vutil.warn("Preferred backends {0} not available, using {1} "
                           "instead.".format(preferred, backend))
                return backend
        except KeyError:
            warn("Unknown backend (Should not be "
                 "implemented?): {0}".format(backend))

    raise RuntimeError("No implemented backends are installed "
                       "for this function.")

def difference(fld_a, fld_b, sla=slice(None), slb=slice(None), backends=None):
    implemented_backends = ["numexpr", "numpy"]
    use_backend = _check_backend(backends, implemented_backends)

    if use_backend == "numexpr":
        diff = necalc.difference(fld_a, fld_b, sla, slb)
    elif use_backend == "numpy":
        a = fld_a.data[sla]
        b = fld_b.data[slb]
        diff = a - b

    return field.wrap_field(fld_a.TYPE, fld_a.name + " difference", fld_a.crds,
                            diff, center=fld_a.center, time=fld_a.time)

def relative_diff(fld_a, fld_b, sla=slice(None), slb=slice(None),
                  backends=None):
    implemented_backends = ["numexpr", "numpy"]
    use_backend = _check_backend(backends, implemented_backends)

    if use_backend == "numexpr":
        diff = necalc.relative_diff(fld_a, fld_b, sla, slb)
    elif use_backend == "numpy":
        a = fld_a.data[sla]
        b = fld_b.data[slb]
        diff = (a - b) / a

    return field.wrap_field(fld_a.TYPE, fld_a.name + " difference", fld_a.crds,
                            diff, center=fld_a.center, time=fld_a.time)

def abs_val(fld, backends=None):
    implemented_backends = ["numexpr", "numpy"]
    use_backend = _check_backend(backends, implemented_backends)

    if use_backend == "numexpr":
        absarr = necalc.abs_val(fld)
    elif use_backend == "numpy":
        absarr = np.abs(fld.data)

    return field.wrap_field(fld.TYPE, "abs " + fld.name, fld.crds,
                            absarr, center=fld.center, time=fld.time)

def magnitude(fld, backends=None):
    implemented_backends = ["numexpr", "cython", "numpy", "native"]
    use_backend = _check_backend(backends, implemented_backends)

    if use_backend == "cython":
        return cycalc.magnitude(fld)
    elif use_backend == "numexpr":
        mag = necalc.magnitude(fld)
    elif use_backend == "numpy":
        vx, vy, vz = fld.component_views()
        mag = np.sqrt((vx**2) + (vy**2) + (vz**2))
    elif use_backend == "native":
        vx, vy, vz = fld.component_views()
        mag = np.empty_like(vx)
        for i in range(mag.shape[0]):
            for j in range(mag.shape[1]):
                for k in range(mag.shape[2]):
                    mag[i, j, k] = np.sqrt(vx[i, j, k]**2 + vy[i, j, k]**2 + \
                                           vz[i, j, k]**2)

    #print("MMM ", np.max(mag - np.sqrt((vx**2) + (vy**2) + (vz**2))))
    #print(mag[:,128,128])
    return field.wrap_field("Scalar", fld.name + " magnitude", fld.crds,
                            mag, center=fld.center, time=fld.time)

def div(fld, backends=None):
    implemented_backends = ["numexpr", "cython"]
    use_backend = _check_backend(backends, implemented_backends)

    if use_backend == "cython":
        return cycalc.div(fld)
    elif use_backend == "numexpr":
        return necalc.div(fld)

def closest1d_ind(arr, val):
    #i = np.argmin(ne.evaluate("abs(arr - val)"))
    return np.argmin(np.abs(arr - val))

def closest1d_val(arr, val):
    #i = np.argmin(ne.evaluate("abs(arr - val)"))
    i = np.argmin(np.abs(arr - val))
    return arr[i]

def nearest_val(fld, point):
    """ find value of field closest to point """
    x, y, z = point
    xind = closest1d_ind(fld.crds['x'], x)
    yind = closest1d_ind(fld.crds['y'], y)
    zind = closest1d_ind(fld.crds['z'], z)
    return fld[zind, yind, xind]

##
## EOF
##
