# -*- coding: utf-8 -*-
"""A set of python modules that aid in plotting scientific data

Plotting depends on matplotlib and/or mayavi and file reading uses h5py
and to read hdf5 / xdmf files.

Note:
    Modules in calculator and plot must be imported explicitly since
    they have side effects on import.

Attributes:
    logger (logging.Logger): a logging object whose verbosity can be
        set from the command line using
        :py:func`viscid.vutil.common_argparse`.
"""

from __future__ import print_function
import logging
import os
import re
import signal
import sys
import textwrap

import numpy

from viscid import _rc
from viscid.compat.vimportlib import import_module


__version__ = """1.0.0"""

__all__ = ['amr_field',
           'amr_grid',
           'bucket',
           'coordinate',
           'cotr',
           'dataset',
           'dipole',
           'extools',
           'field',
           'fluidtrace',
           'grid',
           'mapfield',
           'multiplot',
           'npdatetime',
           'parallel',
           'pyeval',
           'rotation',
           'seed',
           'sliceutil',
           'tree',
           'verror',
           'vjson',
           'vutil',
           'calculator',  # packages
           'compat',
           'cython',
           'plot',
           'readers',
          ]

# weird windows fortran build artifact that has to be in the path
extra_dll_dir = os.path.join(os.path.dirname(__file__), ".libs")
if os.path.isdir(extra_dll_dir):
    os.environ["PATH"] += os.pathsep + extra_dll_dir

# cute default value that's useful in some instances
class NOT_SPECIFIED(object):
    """default value; never instantiate and test with `is`"""
    pass
__all__.append('NOT_SPECIFIED')


#########################################
# setup logger for use throughout viscid
logger = logging.getLogger("viscid")
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(fmt="%(levelname)s: %(message)s"))
logger.addHandler(_handler)


class _CustomFilter(logging.Filter, object):
    def filter(self, record):
        if '\n' not in record.msg:
            record.msg = '\n'.join(textwrap.wrap(record.msg, width=65))
        spaces = ' ' * (len(record.levelname) + 2)
        record.msg = record.msg.replace('\n', '\n' + spaces)
        return super(_CustomFilter, self).filter(record)


logger.addFilter(_CustomFilter())
logger.propagate = False
del _handler


###################################################################
# this is thunder-hacky, but it's a really simple way to import
# everything in __all__ and also, if those module have an __all__,
# then bring that stuff into this namespace too
def _on_injected_import_error(name, exception, quiet=False):
    if not quiet:
        logger.error(str(exception))
        logger.error("Viscid tried to import {0}, but the import failed.\n"
                     "This module will not be available".format(name))

def import_injector(attr_list, namespace, package=None, quiet=False,
                    fatal=False):
    """import list of modules and consume their __all__ attrs"""
    additional = []
    for s in list(attr_list):
        try:
            m = import_module("." + s, package=package)
            namespace[s] = m
            # print(">", package, ">", s)
            # print(">", package, ">", s, "::", getattr(m, "__all__", None))
            if hasattr(m, "__all__"):
                all_subattrs = getattr(m, "__all__")
                additional += all_subattrs
                for sub in all_subattrs:
                    # print("    ", sub, "=", getattr(m, sub))
                    namespace[sub] = getattr(m, sub)
        except ImportError as e:
            if s not in namespace:
                _on_injected_import_error(s, e, quiet=quiet)
                attr_list.remove(s)
                if fatal:
                    raise
    attr_list += additional

import_injector(__all__, globals(), package="viscid")


##############################################################
# now add some other random things into the __all__ namespace
__all__.append("logger")

# set the sample_dir so that it always points to something useful
# - for installed distribution
sample_dir = os.path.join(os.path.dirname(__file__), 'sample')
# - for in-place distribution
if not os.path.isdir(sample_dir):
    sample_dir = os.path.join(os.path.dirname(__file__), '..', 'sample')
    sample_dir = os.path.abspath(sample_dir)
# - is there a 3rd option? this shouldn't happen
if not os.path.isdir(sample_dir):
    sample_dir = "SAMPLE-DIR-NOT-FOUND"

__all__.append("sample_dir")

# now this is just too cute to pass up :)
if sys.version_info[0] >= 3:
    # hide setting to a unicode variable name in an exec b/c otherwise
    # this file wouldn't parse in python2x
    exec("π = numpy.pi")  # pylint: disable=exec-used
    __all__ += ["π"]

# apply settings in the rc file
_rc.load_rc_file("~/.viscidrc")

def check_version():
    """Check status of viscid and associated libraries and modules"""
    print("Viscid located at:", __file__)
    print()
    print("Viscid version:", __version__)
    print()
    print("Python version:", sys.version)
    try:
        import matplotlib
        print("Matplotlib version:", matplotlib.__version__)
    except ImportError:
        print("Matplotlib not installed")
    try:
        import mayavi
        print("Mayavi version:", mayavi.__version__)
        try:
            import vtk
            print("VTK version:", vtk.VTK_VERSION)
        except ImportError:
            print("VTK python module not installed")
    except ImportError:
        print("Mayavi not installed")
    print()

    def print_err(*args, **kwargs):
        kwargs.pop('file', '')
        print(*args, file=sys.stderr, **kwargs)

    if isinstance(cyfield, cython._dummy):
        print_err("WARNING: cython modules (interpolation and streamlines) are not")
        print_err("         available. To use these functions, please ensure that you")
        print_err("         have a C compiler compatable with your version of")
        print_err("         Python / Numpy and reinstall (or rebulid) Viscid.")
        print_err()
    else:
        print("Cython modules are compiled.")

    try:
        from viscid.readers import _jrrle
        print("Fortran modules are compiled.")
    except ImportError:
        print_err("WARNING: jrrle reader is not available. If you need this")
        print_err("         functionality, please ensure that you have a working")
        print_err("         fortran compiler and reinstall (or rebulid) Viscid.")
        print_err()


__all__.append("check_version")

def check():
    """Runtime check compiled modules"""
    import os
    import sys

    import numpy as np
    import viscid

    ret = 0

    check_version()
    print()

    #####################################################
    # run streamline calculation (checks cython modules)
    try:
        cotr = viscid.Cotr(dip_tilt=15.0, dip_gsm=21.0)  # pylint: disable=not-callable
        m = cotr.get_dipole_moment(crd_system='gse')
        seeds = viscid.seed.Sphere((0.0, 0.0, 0.0), 2.0, pole=-m, ntheta=25,
                                   nphi=25, thetalim=(5, 90), philim=(5, 360),
                                   phi_endpoint=False)
        B = viscid.make_dipole(m=m, crd_system='gse', n=(32, 32, 32),
                               l=(-25, -25, -25), h=(25, 25, 25), dtype='f8')
        lines, _ = viscid.calc_streamlines(B, seeds, ibound=1.0)
        for line in lines:
            if np.any(np.isnan(line)):
                raise ValueError("NaN in line")
        print("Cython module ran successfully")
    except Exception as e:
        print("Cython module has runtime errors.")
        print(str(e))
        ret |= (1 << 0)
    print()

    ####################################
    # load a jrrle file (checks fortran)
    try:
        f3d = viscid.load_file(os.path.join(viscid.sample_dir,
                                            'sample_jrrle.3df.*'))
        _ = np.array(f3d['pp'].data)
        print("Fortran module ran successfully")
    except Exception as e:
        print("Fortran module has runtime errors.")
        print(str(e))
        ret |= (1 << 1)
    print()

    return ret

__all__.append("check")

if hasattr(signal, 'SIGINFO'):
    # this is useful for debugging, ie, immediately do a pdb.set_trace()
    # on the SIGINFO signal
    def _set_trace(seg, frame):  # pylint: disable=unused-argument
        import pdb
        pdb.set_trace()
    signal.signal(signal.SIGINFO, _set_trace)
    # print("Trigger pdb with SIGINFO (ctrl + T)")
