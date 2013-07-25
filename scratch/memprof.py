#!/usr/bin/env python

from __future__ import print_function
import resource

import numpy as np
# from pympler import muppy
# from pympler import tracker
# from pympler import summary

from viscid import readers

def print_mem():
    inuse = resource.getrusage(resource.RUSAGE_SELF)
    print("{0} MB".format(inuse.ru_maxrss / 1024**2))  # max resident set size
    # print("{0} MB".format(inuse.ru_ixrss / 1024**2))  # shared mem size
    # print("{0} MB".format(inuse.ru_idrss / 1024**2))  # unshared mem size

if __name__ == "__main__":
    print("just before load")
    print_mem()
    f = readers.load("/Users/kmaynard/dev/work/t1/t1.3df.xdmf")

    print("just after load")
    print_mem()
    for fi in f.iter_times(":"):
        # fi = this_time
        # with this_time as fi:
        bx = fi["bx"]
        by = fi["by"]
        bz = fi["bz"]
        b = np.sqrt(bx**2 + by**2 + bz**2)
        print("Done with time step", fi.time)
        print_mem()
        # fi.unload()
        print("outside the with")
        print_mem()
    fi = bx = by = bz = b = None
    print("Done with time step")
    print_mem()

    print("emptied the bucket")
    print_mem()

##
## EOF
##
