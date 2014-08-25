""" Compatability modules... at the moment, this only contains the ordered dict
backport that's posted to active state """

try:
    from collections import OrderedDict
except ImportError:
    from viscid.compat.ordered_dict_backport import OrderedDict

# Taken from the six module
# Copyright (c) 2010-2014 Benjamin Peterson
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    string_types = str,
else:
    string_types = basestring,

##
## EOF
##
