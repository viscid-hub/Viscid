"""Used for loading ~/.viscidrc

An example rc file might contain the line:
readers.openggcm.GGCMFile.read_log_file: true
"""

from __future__ import print_function
import os
import importlib

import viscid
from viscid import vjson

class RCPathError(Exception):
    message = ""
    def __init__(self, message=""):
        self.message = message
        super(RCPathError, self).__init__(message)


def _get_obj(rt, path):
    if len(path) == 0:
        return rt
    try:
        if not hasattr(rt, path[0]):
            try:
                # if path[0] is not an attribute of rt, then try
                # importing it
                module_name = "{0}.{1}".format(rt.__name__, path[0])
                importlib.import_module(module_name)
            except ImportError:
                # nope, the attribute really doesn't exist
                raise AttributeError("root {0} has no attribute "
                                     "{1}".format(rt, path[0]))
        return _get_obj(getattr(rt, path[0]), path[1:])
    except AttributeError:
        # can't re-raise AttributeError since this is recursive
        # and we're catching AttributeError, so the info about
        # what part of the path DNE would be lost
        raise RCPathError("'{0}' has no attribute '{1}'"
                          "".format(rt.__name__, path[0]))

def set_attribute(path, value):
    # try:
    #     value = _parse_rc_value(value)
    # except RCValueError as e:
    #     print("WARNING: Skipping bad ~/.viscidrc value:: {0}".format(value))
    #     print("                                          {0}".format(e.message))
    #     return None

    p = path.split('.')
    obj = _get_obj(viscid, p[:-1])

    if not hasattr(obj, p[-1]):
        print("WARNING: from rc file, object '{0}' has no attribute '{1}'.\n"
              "         If this isn't a typeo then the functionality may "
              "have moved.".format(".".join(p[:-1]), p[-1]))
    setattr(obj, p[-1], value)

def load_rc_file(fname):
    try:
        with open(os.path.expanduser(os.path.expandvars(fname)), 'r') as f:
            json_obj = vjson.load(f)

        for path, val in json_obj.items():
            try:
                set_attribute(path, val)
            except RCPathError as e:
                print("WARNING: from rc file, {0}\n"
                      "         If this isn't a typeo then the "
                      "functionality may have moved.".format(e.message))
    except IOError:
        pass

##
## EOF
##
