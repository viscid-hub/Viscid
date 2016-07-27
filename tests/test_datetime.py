#!/usr/bin/env python

from __future__ import division, print_function
import argparse
import sys

import numpy as np

from viscid_test_common import next_plot_fname

import viscid
from viscid.plot import mpl


def main():
    parser = argparse.ArgumentParser(description="Test divergence")
    parser.add_argument("--show", "--plot", action="store_true")
    args = viscid.vutil.common_argparse(parser)
    # args.show = True

    t = viscid.linspace_datetime64('2006-06-10 12:30:00.0',
                                   '2006-06-10 12:33:00.0', 16)
    tL = viscid.as_datetime64('2006-06-10 12:31:00.0')
    tR = viscid.as_datetime64('2006-06-10 12:32:00.0')
    y = np.linspace(2 * np.pi, 4 * np.pi, 12)

    ### plots with a datetime64 axis
    f0 = viscid.ones([t, y], crd_names='ty', center='node')
    T, Y = f0.get_crds(shaped=True)
    f0.data += np.arange(T.size).reshape(T.shape)
    f0.data += np.cos(Y)

    fig = mpl.plt.figure(figsize=(10, 5))
    # 1D plot
    mpl.subplot(121)
    mpl.plot(f0[tL:tR]['y=0'], marker='^')
    mpl.plt.xlim(*viscid.as_datetime(t[[0, -1]]).tolist())
    # 2D plot
    mpl.subplot(122)
    mpl.plot(f0, x=(t[0], t[-1]))

    mpl.plt.suptitle("datetime64")
    mpl.auto_adjust_subplots(subplot_params=dict(top=0.9))
    mpl.plt.savefig(next_plot_fname(__file__))
    if args.show:
        mpl.show()
    mpl.plt.close(fig)

    ### plots with a timedelta64 axis
    tL = tL - t[0]
    tR = tR - t[0]
    t = t - t[0]
    f0 = viscid.ones([t, y], crd_names='ty', center='node')
    T, Y = f0.get_crds(shaped=True)
    f0.data += np.arange(T.size).reshape(T.shape)
    f0.data += np.cos(Y)

    fig = mpl.plt.figure(figsize=(10, 5))
    # 1D plot
    mpl.subplot(121)
    mpl.plot(f0[tL:tR]['y=0'], marker='^')
    mpl.plt.xlim(*viscid.as_datetime(t[[0, -1]]).tolist())
    # 2D plot
    mpl.subplot(122)
    mpl.plot(f0, x=(t[0], t[-1]), y=(y[0], y[-1]))

    mpl.plt.suptitle("timedelta64")
    mpl.auto_adjust_subplots(subplot_params=dict(top=0.9))
    mpl.plt.savefig(next_plot_fname(__file__))
    if args.show:
        mpl.show()
    mpl.plt.close(fig)

    return 0

if __name__ == "__main__":
    sys.exit(main())

##
## EOF
##
