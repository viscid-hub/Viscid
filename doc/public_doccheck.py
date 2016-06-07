"""Go through all public functions and make sure they're documented

This checks two things:
    1) Public functions have docstrings
    2) They are documented somewhere in Viscid/doc/functions.rst
"""

from __future__ import print_function
import os
import sys
import types  # from types import BufferType, ModuleType

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), "..")))

import viscid
# from viscid.plot import mpl
# from viscid.plot import mvi


def main():
    doc_fname = os.path.join(os.path.dirname(sys.argv[0]), "functions.rst")
    with open(doc_fname, 'r') as fin:
        doc = fin.read()

    without_sphinx = []
    without_docstr = []

    sphinx_instance_blklst = []
    sphinx_module_blklst = []
    docstr_instance_blklst = []
    docstr_module_blklst = []

    for attr_name in dir(viscid):
        if attr_name.startswith("_"):
            continue
        attr = getattr(viscid, attr_name)
        if callable(attr):
            if not hasattr(attr, "__name__"):
                setattr(attr, "__name__", attr_name)

            if "`viscid.{0}`".format(attr_name) not in doc:
                if any(isinstance(attr, t) for t in sphinx_instance_blklst):
                    pass
                elif any(m in attr.__module__ for m in sphinx_module_blklst):
                    pass
                else:
                    without_sphinx.append(attr)

            if not attr.__doc__:
                if any(isinstance(attr, t) for t in docstr_instance_blklst):
                    pass
                elif any(m in attr.__module__ for m in docstr_module_blklst):
                    pass
                else:
                    without_docstr.append(attr)

    N = 62

    if without_docstr or without_sphinx:
        print("*" * N, file=sys.stderr)
        print("         documentation issues...           ", file=sys.stderr)
    else:
        print("*" * N, file=sys.stderr)
        print("  all public functions are doumented       ", file=sys.stderr)
        print("*" * N, file=sys.stderr)


    if without_docstr:
        err_str = "*" * N + "\n"
        err_str += "The following public functions are missing docstrings\n"
        err_str += "-" * N
        for fn in without_docstr:
            err_str += "\n  - {0}.{1}".format(fn.__module__, fn.__name__)
        print(err_str, file=sys.stderr)

    if without_sphinx:
        err_str = "*" * N + "\n"
        err_str += "The following public functions are not present in\n"
        err_str += "`{0}`:\n".format(doc_fname)
        err_str += "-" * N
        for fn in without_sphinx:
            err_str += "\n  - viscid.{0}".format(fn.__name__)
        print(err_str, file=sys.stderr)

    if without_docstr or without_sphinx:
        print("*" * N, file=sys.stderr)

    if without_docstr and without_sphinx:
        ret = 4
    elif without_docstr:
        ret = 1
    elif without_sphinx:
        ret = 0  # temporarily allow functions that are not in functinos.rst
    else:
        ret = 0
    return ret

if __name__ == "__main__":
    sys.exit(main())

##
## EOF
##
