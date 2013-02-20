#!/usr/bin/env python

from __future__ import print_function
import sys
from time import time

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

##
## EOF
##
