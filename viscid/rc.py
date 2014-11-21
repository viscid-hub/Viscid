"""Used for loading ~/.viscidrc

An example rc file might contain the line:
readers.openggcm.GGCMFile.read_log_file: true
"""

from __future__ import print_function
import os
import importlib

import viscid

class RCPathError(Exception):
    message = ""
    def __init__(self, message=""):
        self.message = message
        super(RCPathError, self).__init__(message)


class RCValueError(Exception):
    message = ""
    def __init__(self, message=""):
        self.message = message
        super(RCValueError, self).__init__(message)


def _parse_rc_value(s):
    s = s.strip()

    ret = None

    if s.startswith('{') and s.endswith('}'):
        # parse a dict
        s = s[1:-1]
        ret = {}
        for item in s.split(','):
            try:
                key, value = item.split(":")
                ret[key.strip()] = _parse_rc_value(value)
            except ValueError:
                if len(item.strip()) == 0:
                    continue
                raise RCValueError("Malformed dictionary: '{0}'"
                                   "".format(item))
    elif s.startswith('[') and s.endswith(']'):
        # parse a list
        ret = [_parse_rc_value(item) for item in s.split(',')]
    else:
        # parse and int / float / bool in that order
        # if all else fails, just retorn the string we got in
        try:
            ret = int(s)
        except ValueError:
            try:
                ret = float(s)
            except ValueError:
                if s.lower() == "true":
                    ret = True
                elif s.lower() == "false":
                    ret = False
                else:
                    ret = s
    return ret

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
    p = path.split('.')
    try:
        value = _parse_rc_value(value)
    except RCValueError as e:
        print("WARNING: Skipping bad ~/.viscidrc value:: {0}".format(e.message))
        return None

    obj = _get_obj(viscid, p[:-1])

    if not hasattr(obj, p[-1]):
        print("WARNING: from rc file, object '{0}' has no attribute '{1}'.\n"
              "         If this isn't a typeo then the functionality may "
              "have moved.".format(".".join(p[:-1]), p[-1]))
    setattr(obj, p[-1], value)

def load_rc_file(fname):
    try:
        with open(os.path.expanduser(os.path.expandvars(fname)), 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                lst = line.split(":")
                path, val = lst[0].strip(), ":".join(lst[1:]).strip()
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
