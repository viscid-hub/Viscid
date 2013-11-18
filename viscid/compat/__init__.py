""" Compatability modules... at the moment, this only contains the ordered dict
backport that's posted to active state """

try:
    from collections import OrderedDict
except ImportError:
    from .ordered_dict_backport import OrderedDict

##
## EOF
##
