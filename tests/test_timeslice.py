#!/usr/bin/env python
"""Test a bunch of Field slice edge cases

Both cell and node centered fields are tested.
"""
# pylint: disable=invalid-slice-index

from __future__ import print_function
import argparse
import sys

import numpy as np

import viscid_test_common  # pylint: disable=unused-import

import viscid
from viscid import vutil


def test_slice(dset, slc, ref_arr, **kwargs):
    # let us also test slicing to a single value
    try:
        len(ref_arr)
    except TypeError:
        ref_arr = np.array([ref_arr])

    t = np.array([g.time for g in dset.iter_times(slc, **kwargs)])
    viscid.logger.debug("{0}: {1}".format(slc, t))
    if len(t) != len(ref_arr) or np.any(t != ref_arr):
        s = ("Slice doesn't match reference\n"
             "  SLICE:     {0}\n"
             "  RESULT:    {1}\n"
             "  REFERENCE: {2}\n".format(slc, t, ref_arr))
        raise RuntimeError(s)

def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = vutil.common_argparse(parser)  # pylint: disable=unused-variable

    # viscid.logger.setLevel(10)

    grids = [viscid.grid.Grid(time=t) for t in np.linspace(1.0, 10.0, 8)]
    dset = viscid.dataset.DatasetTemporal(*grids, basetime='1980-01-01')

    times = np.array([g.time for g in dset])

    ###################
    # slice by integer

    test_slice(dset, np.s_[2], times[2])
    # forward
    test_slice(dset, np.s_[4:], times[4:])
    test_slice(dset, np.s_[2::3], times[2::3])
    test_slice(dset, np.s_[:4], times[:4])
    test_slice(dset, np.s_[:5:2], times[:5:2])
    test_slice(dset, np.s_[2:5:2], times[2:5:2])
    test_slice(dset, np.s_[6:5:1], times[6:5:1])
    # backward
    test_slice(dset, np.s_[4::-1], times[4::-1])
    test_slice(dset, np.s_[:4:-1], times[:4:-1])
    test_slice(dset, np.s_[:4:-2], times[:4:-2])
    test_slice(dset, np.s_[2:5:-2], times[2:5:-2])
    test_slice(dset, np.s_[6:4:-1], times[6:4:-1])
    # forward
    test_slice(dset, '4:', times[4:])
    test_slice(dset, '2::3', times[2::3])
    test_slice(dset, ':4', times[:4])
    test_slice(dset, ':5:2', times[:5:2])
    test_slice(dset, '2:5:2', times[2:5:2])
    test_slice(dset, '6:5:1', times[6:5:1])
    # backward
    test_slice(dset, '4::-1', times[4::-1])
    test_slice(dset, ':4:-1', times[:4:-1])
    test_slice(dset, ':4:-2', times[:4:-2])
    test_slice(dset, '2:5:-2', times[2:5:-2])
    test_slice(dset, '6:4:-1', times[6:4:-1])

    #################
    # slice by float

    # Note: times = [  1.           2.28571429   3.57142857   4.85714286
    #                  6.14285714   7.42857143   8.71428571  10.        ]

    test_slice(dset, np.s_['4.0f'], times[2])
    test_slice(dset, np.s_['4.0f':], times[3:])
    test_slice(dset, np.s_['4.0f'::2], times[3::2])
    test_slice(dset, np.s_[:'4.0f':2], times[:3:2])
    test_slice(dset, np.s_['2.0f':'7.8f'], times[1:6])
    test_slice(dset, np.s_['2.0f':'7.8f':2], times[1:6:2])
    test_slice(dset, np.s_['7.8f':'2.0f':-1], times[5:0:-1])
    test_slice(dset, np.s_['7.8f':'2.0f':-1], times[5:1:-1], val_endpoint=False)
    test_slice(dset, np.s_['7.8f':'2.0f':-2], times[5:0:-2])
    test_slice(dset, np.s_['7.8f':'2.0f':-2], times[5:1:-2], val_endpoint=False)
    test_slice(dset, np.s_['3.4f':'7.3f'], times[2:5])
    test_slice(dset, np.s_['3.4f':'7.3f'], times[1:6], interior=True)
    test_slice(dset, np.s_['2.4f':'2.5f'], times[2:2])
    test_slice(dset, np.s_['2.1f':'2.5f'], times[1:2])
    test_slice(dset, np.s_['2.1f':'2.5f'], times[1:1], val_endpoint=False)
    test_slice(dset, np.s_['2.3f':'2.5f'], times[1:3], interior=True)

    ################
    # slice by imag
    test_slice(dset, np.s_[4.0j], times[2])
    test_slice(dset, np.s_[4.0j:], times[3:])
    test_slice(dset, np.s_[4.0j::2], times[3::2])
    test_slice(dset, np.s_[:4.0j:2], times[:3:2])
    test_slice(dset, np.s_[2.0j:7.8j], times[1:6])
    test_slice(dset, np.s_[2.0j:7.8j:2], times[1:6:2])
    test_slice(dset, np.s_[7.8j:2.0j:-1], times[5:0:-1])
    test_slice(dset, np.s_[7.8j:2.0j:-1], times[5:1:-1], val_endpoint=False)
    test_slice(dset, np.s_[7.8j:2.0j:-2], times[5:0:-2])
    test_slice(dset, np.s_[7.8j:2.0j:-2], times[5:1:-2], val_endpoint=False)
    test_slice(dset, np.s_[3.4j:7.3j], times[2:5])
    test_slice(dset, np.s_[3.4j:7.3j], times[1:6], interior=True)
    test_slice(dset, np.s_[2.4j:2.5j], times[2:2])
    test_slice(dset, np.s_[2.1j:2.5j], times[1:2])
    test_slice(dset, np.s_[2.1j:2.5j], times[1:1], val_endpoint=False)
    test_slice(dset, np.s_[2.3j:2.5j], times[1:3], interior=True)

    ####################
    # slice by imag str
    test_slice(dset, np.s_['4.0j'], times[2])
    test_slice(dset, np.s_['4.0j':], times[3:])
    test_slice(dset, np.s_['4.0j'::2], times[3::2])
    test_slice(dset, np.s_[:'4.0j':2], times[:3:2])
    test_slice(dset, np.s_['2.0j':'7.8j'], times[1:6])
    test_slice(dset, np.s_['2.0j':'7.8j':2], times[1:6:2])
    test_slice(dset, np.s_['7.8j':'2.0j':-1], times[5:0:-1])
    test_slice(dset, np.s_['7.8j':'2.0j':-1], times[5:1:-1], val_endpoint=False)
    test_slice(dset, np.s_['7.8j':'2.0j':-2], times[5:0:-2])
    test_slice(dset, np.s_['7.8j':'2.0j':-2], times[5:1:-2], val_endpoint=False)
    test_slice(dset, np.s_['3.4j':'7.3j'], times[2:5])
    test_slice(dset, np.s_['3.4j':'7.3j'], times[1:6], interior=True)
    test_slice(dset, np.s_['2.4j':'2.5j'], times[2:2])
    test_slice(dset, np.s_['2.1j':'2.5j'], times[1:2])
    test_slice(dset, np.s_['2.1j':'2.5j'], times[1:1], val_endpoint=False)
    test_slice(dset, np.s_['2.3j':'2.5j'], times[1:3], interior=True)

    ############################
    # slice by deprecated float
    viscid.logger.info("testing deprecated slice-by-location")
    test_slice(dset, np.s_['4.0'], times[2])
    test_slice(dset, np.s_['4.0':], times[3:])
    test_slice(dset, np.s_['4.0'::2], times[3::2])
    test_slice(dset, np.s_[:'4.0':2], times[:3:2])
    test_slice(dset, np.s_['2.0':'7.8'], times[1:6])
    test_slice(dset, np.s_['2.0':'7.8':2], times[1:6:2])
    test_slice(dset, np.s_['7.8':'2.0':-1], times[5:0:-1])
    test_slice(dset, np.s_['7.8':'2.0':-1], times[5:1:-1], val_endpoint=False)
    test_slice(dset, np.s_['7.8':'2.0':-2], times[5:0:-2])
    test_slice(dset, np.s_['7.8':'2.0':-2], times[5:1:-2], val_endpoint=False)
    test_slice(dset, np.s_['3.4':'7.3'], times[2:5])
    test_slice(dset, np.s_['3.4':'7.3'], times[1:6], interior=True)
    test_slice(dset, np.s_['2.4':'2.5'], times[2:2])
    test_slice(dset, np.s_['2.1':'2.5'], times[1:2])
    test_slice(dset, np.s_['2.1':'2.5'], times[1:1], val_endpoint=False)
    test_slice(dset, np.s_['2.3':'2.5'], times[1:3], interior=True)
    viscid.logger.info("done testing deprecated slice-by-location")

    ####################
    # slice by datetime
    test_slice(dset, np.s_['1980-01-01T00:00:03.0':'1980-01-01T00:00:07.8'],
               times[2:6])
    test_slice(dset, np.s_['UT1980-01-01T00:00:03.0:UT1980-01-01T00:00:07.8'],
               times[2:6])
    test_slice(dset, np.s_['T1980-01-01T00:00:03.0:T1980-01-01T00:00:07.8'],
               times[2:6])

    test_slice(dset, np.s_['1980-01-01T00:00:07.8':'1980-01-01T00:00:03.0':-2],
               times[5:1:-2])

    #####################
    # slice by timedelta
    test_slice(dset, np.s_['00:03.0':'00:07.8'], times[2:6])
    test_slice(dset, np.s_['T00:03.0:T00:07.8'], times[2:6])

    #############################
    # slice by a few mixed types
    test_slice(dset, np.s_['UT1980-01-01T00:00:03.0:T00:07.8'], times[2:6])
    test_slice(dset, np.s_['3f:T00:07.8'], times[2:6])
    test_slice(dset, np.s_['3:T00:07.8'], times[3:6])

    assert dset.tslc_range('2:5') == (3.5714285714285716,
                                      7.4285714285714288)
    assert dset.tslc_range('2.3f:2.5f') == (2.3, 2.5)
    assert dset.tslc_range('UT1980-01-01T00:00:03.0:'
                           'UT1980-01-01T00:00:07.8') == (3.0, 7.8)


    t = viscid.linspace_datetime64('2010-01-01T12:00:00',
                                   '2010-01-01T15:00:00', 8)
    x = np.linspace(-1, 1, 12)
    fld = viscid.zeros([t, x], crd_names='tx', center='node')
    assert fld[:'2010-01-01T13:30:00'].shape == (4, 12)
    fld = viscid.zeros([t, x], crd_names='tx', center='cell')
    assert fld[:'2010-01-01T13:30:00'].shape == (4, 11)

    t = viscid.linspace_datetime64('2010-01-01T12:00:00',
                                   '2010-01-01T15:00:00', 8)
    t = t - t[0]
    x = np.linspace(-1, 1, 12)
    fld = viscid.zeros([t, x], crd_names='tx', center='node')
    assert fld[:'01:30:00'].shape == (4, 12)
    fld = viscid.zeros([t, x], crd_names='tx', center='cell')
    assert fld[:'01:30:00'].shape == (4, 11)

    return 0


if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
