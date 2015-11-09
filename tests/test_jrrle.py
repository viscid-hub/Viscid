#!/usr/bin/env python
""" test a ggcm grid wrapper """

from __future__ import print_function, division
import argparse

from viscid_test_common import sample_dir, next_plot_fname

import viscid
from viscid import vutil
from viscid.plot import mpl
from viscid.plot.mpl import plt

def lon_fmt(lon):
    return "{0:g}".format(lon * 24.0 / 360.0)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", "--plot", action="store_true")
    args = vutil.common_argparse(parser)

    iono_file = viscid.load_file(sample_dir + '/sample_jrrle.iof.*')

    fac_tot = 1e9 * iono_file["fac_tot"]

    plot_args = dict(projection="polar",
                     lin=[-4e3, 3e3],
                     bounding_lat=35.0,
                     drawcoastlines=True,  # for basemap only
                     title="Total FAC\n",
                     gridec='gray',
                     label_lat=True,
                     label_mlt=True,
                     colorbar=dict(pad=0.1)  # pad the colorbar away from the plot
                    )

    ax1 = plt.subplot(121, projection='polar')
    mpl.plot(fac_tot, ax=ax1, hemisphere='north', **plot_args)
    ax1.annotate('(a)', xy=(0, 0), textcoords="axes fraction",
                 xytext=(-0.1, 1.0), fontsize=18)

    ax2 = plt.subplot(122, projection='polar')
    plot_args['gridec'] = False
    mpl.plot(fac_tot, ax=ax2, hemisphere="south", style="contourf",
             levels=50, extend="both", **plot_args)
    ax2.annotate('(b)', xy=(0, 0), textcoords="axes fraction",
                 xytext=(-0.1, 1.0), fontsize=18)

    mpl.plt.savefig(next_plot_fname(__file__))
    if args.show:
        mpl.mplshow()

if __name__ == "__main__":
    main()

##
## EOF
##
