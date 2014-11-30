"""Used for loading ~/.viscidrc

An example rc file might contain the line:
readers.openggcm.GGCMFile.read_log_file: true
"""

from __future__ import print_function
import os
import importlib
import ast

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

class _Transformer(ast.NodeTransformer):
    """Turn a string into python objects (strings, numbers, as basic types)"""
    ALLOWED_NODE_TYPES = set([
        'Expression', # a top node for an expression
        'Name',       # an identifier...
        'NameConstant',
        'Attribute',
        'Load',       # loads a value of a variable with given identifier
        'Str',        # a string literal
        'Num',        # allow numbers too
        'Tuple',      # makes a tuple
        'List',       # and list literals
        'Dict',       # and dicts...
    ])

    def visit_Name(self, node):
        if node.id.lower() == "true":
            node = ast.copy_location(ast.NameConstant(True), node)
        elif node.id.lower() == "false":
            node = ast.copy_location(ast.NameConstant(False), node)
        else:
            node = ast.copy_location(ast.Str(s=node.id), node)
        return self.generic_visit(node)

    # def generic_visit(self, node):
    #     # nodetype = type(node).__name__
    #     # if nodetype not in self.ALLOWED_NODE_TYPES:
    #     #     raise RCValueError("Invalid expression: %s not allowed" % nodetype)
    #     return ast.NodeTransformer.generic_visit(self, node)

def _parse_rc_value(s):
    try:
        tree = ast.parse(s.strip(), mode='eval')
        _Transformer().visit(tree)
        return ast.literal_eval(tree)
    except ValueError as e:
        raise RCValueError("ast parser vomited")

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
        print("WARNING: Skipping bad ~/.viscidrc value:: {0}".format(value))
        print("                                          {0}".format(e.message))
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
