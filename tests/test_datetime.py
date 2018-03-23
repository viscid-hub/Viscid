#!/usr/bin/env python
"""Test some of the datetime64 parsing"""

from __future__ import division, print_function
import argparse
import sys

import numpy as np

from viscid_test_common import next_plot_fname

import viscid
from viscid.plot import vpyplot as vlt
import matplotlib.pyplot as plt


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
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

    fig = plt.figure(figsize=(10, 5))
    # 1D plot
    vlt.subplot(121)
    vlt.plot(f0[tL:tR]['y=0'], marker='^')
    plt.xlim(*viscid.as_datetime(t[[0, -1]]).tolist())
    # 2D plot
    vlt.subplot(122)
    vlt.plot(f0, x=(t[0], t[-1]))

    plt.suptitle("datetime64")
    vlt.auto_adjust_subplots(subplot_params=dict(top=0.9))
    plt.savefig(next_plot_fname(__file__))
    if args.show:
        vlt.show()
    plt.close(fig)

    ### plots with a timedelta64 axis
    tL = tL - t[0]
    tR = tR - t[0]
    t = t - t[0]
    f0 = viscid.ones([t, y], crd_names='ty', center='node')
    T, Y = f0.get_crds(shaped=True)
    f0.data += np.arange(T.size).reshape(T.shape)
    f0.data += np.cos(Y)

    fig = plt.figure(figsize=(10, 5))
    # 1D plot
    vlt.subplot(121)
    vlt.plot(f0[tL:tR]['y=0'], marker='^')
    plt.xlim(*viscid.as_datetime(t[[0, -1]]).tolist())
    # 2D plot
    vlt.subplot(122)
    vlt.plot(f0, x=(t[0], t[-1]), y=(y[0], y[-1]))

    plt.suptitle("timedelta64")
    vlt.auto_adjust_subplots(subplot_params=dict(top=0.9))
    plt.savefig(next_plot_fname(__file__))
    if args.show:
        vlt.show()
    plt.close(fig)

    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
