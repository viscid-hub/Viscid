#!/usr/bin/env python

from __future__ import print_function
import sys
from time import time

from . import verror

def warn(message):
    sys.stderr.write("WARNING: {0}\n".format(message))

def subclass_spider(cls):
    """ return recursive list of subclasses of cls """
    sub_classes = cls.__subclasses__()
    lst = [cls]
    for c in sub_classes:
        lst += subclass_spider(c)
    return lst

def timereps(reps, func, *args, **kwargs):
    arr = [None] * reps
    for i in range(reps):
        start = time()
        func(*args, **kwargs)
        end = time()
        arr[i] = end - start
    return min(arr), max(arr), sum(arr) / reps

def find_field(vfile, fld_name_lst):
    """ convenience function to get a field that could be called many things
    returns the first fld_name in the list that is in the file """
    for fld_name in fld_name_lst:
        if fld_name in vfile:
            return vfile[fld_name]
    raise verror.FieldNotFound("file {0} contains none of "
                               "{1}".format(vfile, fld_name_lst))

##
## EOF
##
