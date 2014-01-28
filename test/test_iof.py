#!/usr/bin/env python
""" test a ggcm grid wrapper """

from __future__ import print_function
import sys
import os
import argparse

import numpy as np
try:
    from mpl_toolkits.basemap import Basemap, shiftgrid
    _has_basemap = True
except ImportError:
    _has_basemap = False
from matplotlib import pyplot as plt

_viscid_root = os.path.realpath(os.path.dirname(__file__) + '/../src/viscid/')
if not _viscid_root in sys.path:
    sys.path.append(_viscid_root)

from viscid import vutil
from viscid import readers
from viscid.plot import mpl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)

    fio = readers.load_file(_viscid_root + '/../../sample/cen2000.iof.xdmf')

    pot = fio["pot"]

    if _has_basemap:
        lon, lat = np.meshgrid(pot.crds["lon"], 90.0 - pot.crds["lat"])

        plt.subplot2grid((3, 2), (0, 0), colspan=2)
        mpl.plot(pot, plot_opts="lin_0")

        ax = plt.subplot2grid((3, 2), (1, 0), colspan=2)
        m = Basemap(projection='hammer', lon_0=0, resolution='c', ax=ax)
        m.contourf(lon, lat, pot.data, 20, latlon=True)
        m.drawcoastlines(linewidth=0.25)

        ax = plt.subplot2grid((3, 2), (2, 0))
        m = Basemap(projection='ortho', lat_0=90.0, lon_0=180.0,
                    resolution='c', ax=ax)
        m.contourf(lon, lat, pot.data, 20, latlon=True)
        m.drawcoastlines(linewidth=0.25)

        ax = plt.subplot2grid((3, 2), (2, 1))
        m = Basemap(projection='ortho', lat_0=-90.0, lon_0=0.0,
                    resolution='c', ax=ax)
        m.contourf(lon, lat, pot.data, 20, latlon=True)
        m.drawcoastlines(linewidth=0.25)
    else:
        mpl.plot(pot, plot_opts="lin_0")

    if args.show:
        mpl.mplshow()

if __name__ == "__main__":
    main()

##
## EOF
##
