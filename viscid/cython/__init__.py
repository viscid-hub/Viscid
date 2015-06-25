"""Some speedy compiled functions for interpolation and streamlines"""

__all__ = ["cyamr", "cycalc", "cyfield", "integrate", "streamline"]

class CythonNotBuilt(Exception):
    pass

class _dummy(object):
    def __init__(self, msg="Dummy Object"):
        self.msg = msg
    def __getattr__(self, name):
        raise CythonNotBuilt(self.msg)
    def __setattr__(self, name, value):
        if name != "msg":
            raise CythonNotBuilt(self.msg)
        else:
            super(_dummy, self).__setattr__(name, value)

try:
    from viscid.cython import cyamr
    from viscid.cython import cycalc
    from viscid.cython import streamline

    from viscid.cython.cycalc import interp_nearest
    from viscid.cython.cycalc import interp_trilin
    from viscid.cython.streamline import calc_streamlines
    streamlines = calc_streamlines
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
    streamline = _dummy(cython_msg.format("streamline"))

##
## EOF
##
