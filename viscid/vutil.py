#!/usr/bin/env python

from __future__ import print_function
import sys

def warn(message):
    sys.stderr.write("WARNING: {0}\n".format(message))

def subclass_spider(cls):
    """ return recursive list of subclasses of cls """
    sub_classes = cls.__subclasses__()
    lst = [cls]
    for c in sub_classes:
        lst += subclass_spider(c)
    return lst

##
## EOF
##
