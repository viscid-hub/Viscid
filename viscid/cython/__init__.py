"""Some speedy compiled functions for interpolation and streamlines

So, some comments on the cython code are probably warranted. There
are many places where you'll see the code going sideways in order
to accomodate Fused types and retain it's typedness. This is especially
true with the weird `def _py_*` functions. These are all done to keep
important function calls in C (important meaning say called once per
segment in a streamline). The end result is that the cython code is
almost on par with a fortran streamline tracer, but far more versatile.
In short, anything that makes you think "why'd he do that?" you can
probably assume it's the path of least resistance in the whole Cython
mess.
"""

__all__ = ["cyamr", "cycalc", "cyfield", "streamline", "integrate",
           "null_tools",
           "interp", "interp_nearest", "interp_linear", "interp_trilin",
           "calc_streamlines", "streamlines", "find_classified_nulls",
           "find_nulls"]

# these are duplicated so that the values are available even if the cython
# code is not built. In principle, I could parse streamline.pyx to pull
# out the values, but that seems less than optimal
_streamline_attrs = dict(EULER1=1,  # euler1 non-adaptive
                         RK2=2,  # rk2 non-adaptive
                         RK4=3,  # rk4 non-adaptive
                         EULER1A=4,  # euler 1st order adaptive (huen)
                         RK12=5,  # euler1 + rk2 adaptive (midpoint)
                         RK45=6,  # RK-Fehlberg 45 adaptive
                         DIR_FORWARD=1,
                         DIR_BACKWARD=2,
                         DIR_BOTH=3,
                         OUTPUT_STREAMLINES=1,
                         OUTPUT_TOPOLOGY=2,
                         OUTPUT_BOTH=3,
                         TOPOLOGY_MS_NONE=0,
                         TOPOLOGY_MS_CLOSED=1,
                         TOPOLOGY_MS_OPEN_NORTH=2,
                         TOPOLOGY_MS_OPEN_SOUTH=4,
                         TOPOLOGY_MS_SW=8,
                         TOPOLOGY_MS_SEPARATOR=15)

class CythonNotBuilt(Exception):
    pass

class _dummy(object):
    """CythonNotBuilt Proxy Object"""
    def __init__(self, msg="Dummy Object", **attrs):
        attrs["msg"] = msg
        for attrname, value in attrs.items():
            super(_dummy, self).__setattr__(attrname, value)
    def __getattr__(self, name):
        try:
            return name in super(_dummy, self).__getattr__(name)
        except AttributeError:
            raise CythonNotBuilt(self.msg)
    def __setattr__(self, name, value):
        raise CythonNotBuilt(self.msg)

try:
    from viscid.cython import cyamr
    from viscid.cython import cycalc
    from viscid.cython import cyfield
    from viscid.cython import streamline
    from viscid.cython import integrate
    from viscid.cython import null_tools

    from viscid.cython.cycalc import interp
    from viscid.cython.cycalc import interp_nearest
    from viscid.cython.cycalc import interp_linear
    from viscid.cython.cycalc import interp_trilin
    from viscid.cython.streamline import calc_streamlines
    streamlines = calc_streamlines
    from viscid.cython.null_tools import find_classified_nulls
    from viscid.cython.null_tools import find_nulls

    # Check to make sure the _streamline_attrs values are the same as
    # the actual cython values. Since this is fragile in that _streamline_attrs
    # is only used as a fallback if the cython code is not built, I want
    # a full stop, go-fix-this-now type error
    def _check_streamline_attrs():
        streamline_attrs_err_msg = ("FULL STOP! viscid/cython/__init__.py "
                                    "_streamline_attrs is out of sync with "
                                    "compiled values! Go fix this now, I will "
                                    "wait.")
        for attrname, value in _streamline_attrs.items():
            if getattr(streamline, attrname) != value:
                raise RuntimeError(streamline_attrs_err_msg)
    _check_streamline_attrs()

except ImportError as e:
    cython_msg = ("Cython module calculator.{0} not available. Cython code \n"
                  "must be rebuilt using Viscid/setup.py (Note: Cython is "
                  "not required for the build,\njust a c compiler)\n\n"
                  "ImportError:\n" + str(e).strip())

    def interp(*args, **kwargs):  # pylint: disable=unused-argument
        """CythonNotBuilt Proxy"""
        raise CythonNotBuilt(cython_msg.format("interp"))
    def interp_nearest(*args, **kwargs):  # pylint: disable=unused-argument
        """CythonNotBuilt Proxy"""
        raise CythonNotBuilt(cython_msg.format("interp_nearest"))
    def interp_linear(*args, **kwargs):  # pylint: disable=unused-argument
        """CythonNotBuilt Proxy"""
        raise CythonNotBuilt(cython_msg.format("interp_linear"))
    def interp_trilin(*args, **kwargs):  # pylint: disable=unused-argument
        """CythonNotBuilt Proxy"""
        raise CythonNotBuilt(cython_msg.format("interp_trilin"))
    def calc_streamlines(*args, **kwargs):  # pylint: disable=unused-argument
        """CythonNotBuilt Proxy"""
        raise CythonNotBuilt(cython_msg.format("calc_streamlines"))
    streamlines = calc_streamlines
    def find_classified_nulls(*args, **kwargs):  # pylint: disable=unused-argument
        """CythonNotBuilt Proxy"""
        raise CythonNotBuilt(cython_msg.format("find_classified_nulls"))
    def find_nulls(*args, **kwargs):  # pylint: disable=unused-argument
        """CythonNotBuilt Proxy"""
        raise CythonNotBuilt(cython_msg.format("find_nulls"))

    cyamr = _dummy(cython_msg.format("cyamr"))
    cycalc = _dummy(cython_msg.format("cycalc"))
    cyfield = _dummy(cython_msg.format("cyfield"))
    streamline = _dummy(cython_msg.format("streamline"), **_streamline_attrs)
    integrate = _dummy(cython_msg.format("integrate"))
    null_tools = _dummy(cython_msg.format("null_tools"))


# this is kinda silly, but lets us have access to TOPOLOGY_MS_* even
# when cython code is not built
for attr in dir(streamline):
    if attr[0] != '_' and attr.isupper():
        vars()[attr] = getattr(streamline, attr)
        __all__.append(attr)


##
## EOF
##
