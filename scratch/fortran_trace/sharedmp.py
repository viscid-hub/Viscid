#!/usr/bin/env python

import ctypes
import multiprocessing as mp
from contextlib import closing
from itertools import islice, repeat, count
try:
    from itertools import izip
except ImportError:
    izip = zip

import numpy as np

def do_business_star(*args, **kwargs):
    # print "args", len(args), len(args[0])
    return business(*args[0], **kwargs)

def business(i):
    print "business"
    print hex(id(ndarr))
    return ndarr[i::3] * 2.0
    # return i**2

def main():
    # create shared array
    NX, NY, NZ = 512, 256, 256
    nr_procs = 3

    global ndarr
    ndarr = np.arange(NZ * NY * NX).reshape((NZ, NY, NX))
    p = mp.Pool(nr_procs)
    # stuff = izip(range(3), repeat(ndarr))

    print "before"
    ar = p.map_async(business, range(3))
    while(True):
        try:
            print("get")
            result = ar.get(2**20)
            break
        except mp.TimeoutError:
            continue

    print "r =", len(result)
    print "done"

if __name__ == '__main__':
    main()

##
## EOF
##
