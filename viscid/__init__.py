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
import re
import signal
import sys
import textwrap

import numpy

from viscid import _rc
from viscid.compat.vimportlib import import_module


__version__ = """0.98.3 dev"""

__all__ = ['amr_field',
           'amr_grid',
           'bucket',
           'coordinate',
           'cotr',
           'dataset',
           'dipole',
           'field',
           'grid',
           'mapfield',
           'npdatetime',
           'parallel',
           'pyeval',
           'seed',
           'tree',
           'verror',
           'vjson',
           'vlab',
           'vutil',
           'calculator',  # packages
           'compat',
           'cython',
           'plot',
           'readers',
          ]


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

# now this is just too cute to pass up :)
if sys.version_info[0] >= 3:
    # hide setting to a unicode variable name in an exec b/c otherwise
    # this file wouldn't parse in python2x
    exec("π = numpy.pi")  # pylint: disable=exec-used
    __all__ += ["π"]

# apply settings in the rc file
_rc.load_rc_file("~/.viscidrc")

# this block is useful for debugging, ie, immediately do a pdb.set_trace()
# on the SIGUSR2 signal
def _set_trace(seg, frame):  # pylint: disable=unused-argument
    import pdb
    pdb.set_trace()
# import os
# print("Trigger pdb with: kill -SIGUSR2", os.getpid())
signal.signal(signal.SIGUSR2, _set_trace)
