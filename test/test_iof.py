#!/usr/bin/env python
""" test a ggcm grid wrapper """

from __future__ import print_function, division
import sys
import os
import argparse

from matplotlib import pyplot as plt

_viscid_root = os.path.realpath(os.path.dirname(__file__) + '/../src/viscid/')
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)

import viscid
from viscid import vutil
from viscid.plot import mpl

def lon_fmt(lon):
    return "{0:g}".format(lon * 24.0 / 360.0)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)

    iono_file = viscid.load_file(_viscid_root + '/../../sample/cen2000.iof.xdmf')

    fac_tot = 1e9 * iono_file["fac_tot"]

    ax1 = plt.subplot(121)
    mpl.plot(fac_tot, ax=ax1, hemisphere="north", style="contourf",
             plot_opts="lin_-300_300", extend="both",
             levels=50, drawcoastlines=True)
    ax2 = plt.subplot(122)
    mpl.plot(fac_tot, ax=ax2, hemisphere="south", style="contourf",
             plot_opts="lin_-300_300", extend="both",
             levels=50, drawcoastlines=True)

    if args.show:
        mpl.mplshow()

if __name__ == "__main__":
    main()

##
## EOF
##
