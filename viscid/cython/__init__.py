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

__all__ = ["cyamr", "cycalc", "cyfield", "integrate", "streamline"]

# these are duplicated so that the values are available even if the cython
# code is not built. In principle, I could parse streamline.pyx to pull
# out the values, but that seems less than optimal
_streamline_attrs = dict(TOPOLOGY_MS_NONE=0,
                         TOPOLOGY_MS_CLOSED=1,
                         TOPOLOGY_MS_OPEN_NORTH=2,
                         TOPOLOGY_MS_OPEN_SOUTH=4,
                         TOPOLOGY_MS_SW=8,
                         TOPOLOGY_MS_SEPARATOR=15)

class CythonNotBuilt(Exception):
    pass

class _dummy(object):
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
    from viscid.cython import streamline

    from viscid.cython.cycalc import interp_nearest
    from viscid.cython.cycalc import interp_trilin
    from viscid.cython.streamline import calc_streamlines

    streamlines = calc_streamlines

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

except ImportError:
    cython_msg = ("Cython module calculator.{0} not available. Cython code "
                  "must be built using Viscid/setup.py (Note: Cython is "
                  "not required for the build, just a c compiler)")

    def interp_nearest(*args, **kwargs):  # pylint: disable=unused-argument
        raise CythonNotBuilt(cython_msg.format("interp_nearest"))
    def interp_trilin(*args, **kwargs):  # pylint: disable=unused-argument
        raise CythonNotBuilt(cython_msg.format("interp_trilin"))
    def streamlines(*args, **kwargs):  # pylint: disable=unused-argument
        raise CythonNotBuilt(cython_msg.format("streamlines"))
    def calc_streamlines(*args, **kwargs):  # pylint: disable=unused-argument
        raise CythonNotBuilt(cython_msg.format("calc_streamlines"))

    cyamr = _dummy(cython_msg.format("cyamr"))
    cycalc = _dummy(cython_msg.format("cycalc"))
    streamline = _dummy(cython_msg.format("streamline"), **_streamline_attrs)

##
## EOF
##
