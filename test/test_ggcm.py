#!/usr/bin/env python
""" test a ggcm grid wrapper """

from __future__ import print_function
import sys
import os
import argparse

from matplotlib import pyplot as plt

_viscid_root = os.path.realpath(os.path.dirname(__file__) + '/../viscid/')
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)

from viscid import vutil
from viscid import readers
from viscid.readers import openggcm
from viscid.plot import mpl

# These two class definitions are examples of how to override a
# built-in reader and implement some convenience getters for
# derived quantities not in the file
class MyGGCMGrid(openggcm.GGCMGrid):
    mhd_to_gse_on_read = True

    def _get_bcc(self):
        return self['b']


# So, you can make a class that derrives from an existing vFile type.
# That way you just define this class, and when it comes time to call
# load_file(file_name), you don't have to give it grid_type=...
# class MyGGCMFile(openggcm.GGCMFile):  # pylint: disable=W0223
#     # you can change the detector, or you can not and you will
#     # completely override the parent reader
#     _detector = r"^\s*.*\.(p[xyz]_[0-9]+|3d|3df|iof)" \
#                 r"(\.[0-9]{6})?\.(xmf|xdmf)\s*$"
#
#     # this is for injecting your convenience methods defined
#     # above
#     _grid_type = MyGGCMGrid


def timeit(f, *args, **kwargs):
    from timeit import default_timer as time
    t0 = time()
    ret = f(*args, **kwargs)
    t1 = time()

    print("Took {0:.03g} secs.".format(t1 - t0))
    return ret

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)

    f3d = readers.load_file(_viscid_root + '/../sample/sample.3df.xdmf',
                            grid_type=MyGGCMGrid)

    pp = f3d['pp']
    rr = f3d['rr']
    T = f3d['T']
    # bmag = timeit(lambda: f3d['bmag'])
    bmag = f3d["bmag"]

    plt.subplot(141)
    mpl.plot(pp, "y=0,x=-20:10", plot_opts="log", earth=True, show=False)
    plt.subplot(142)
    mpl.plot(rr, "y=0,x=-20:10", plot_opts="log", earth=True, show=False)
    plt.subplot(143)
    mpl.plot(T, "y=0,x=-20:10", plot_opts="log", earth=True, show=False)
    plt.subplot(144)
    mpl.plot(bmag, "y=0,x=-20:10", plot_opts="log", earth=True, show=False)

    if args.show:
        mpl.mplshow()

if __name__ == "__main__":
    main()

##
## EOF
##
