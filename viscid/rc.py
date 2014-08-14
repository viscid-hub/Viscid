"""Used for loading ~/.viscidrc

An example rc file might contain the line:
readers.openggcm.GGCMFile.read_log_file: true
"""

from __future__ import print_function
import os

import viscid

class RCPathError(Exception):
    pass

def _parse_rc_value(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            if s.lower() == "true":
                return True
            elif s.lower() == "false":
                return False
            else:
                return s

def _get_obj(rt, path):
    if len(path) == 0:
        return rt
    try:
        return _get_obj(getattr(rt, path[0]), path[1:])
    except AttributeError:
        # can't re-raise AttributeError since this is recursive
        # and we're catching AttributeError, so the info about
        # what part of the path DNE would be lost
        raise RCPathError("'{0}' has no attribute '{1}'".format(
                          rt.__name__, path[0]))

def set_attribute(path, value):
    p = path.split('.')
    value = _parse_rc_value(value)
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
