""" Compatability modules... at the moment, this only contains the ordered dict
backport that's posted to active state """

# Taken from the six module
# Copyright (c) 2010-2014 Benjamin Peterson
import sys
PY3 = sys.version_info[0] == 3

try:
    from itertools import izip  # pylint: disable=no-name-in-module
    from itertools import izip_longest  # pylint: disable=no-name-in-module
except ImportError:
    izip = zip
    from itertools import zip_longest as izip_longest

try:
    from collections import OrderedDict
except ImportError:
    from viscid.compat.ordered_dict_backport import OrderedDict

if PY3:
    string_types = str,
    unicode = str
else:
    string_types = basestring,  # pylint: disable=undefined-variable
    unicode = unicode

__all__ = ['PY3', 'OrderedDict', 'izip', 'izip_longest', 'string_types',
           'unicode']

##
## EOF
##
