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


def test_slice(arr, sel, ref, **kwargs):
    # result = arr[viscid.to_slice(arr, slc, **kwargs)]
    std_sel = viscid.standardize_sel(sel)
    result = arr[viscid.std_sel2index(std_sel, arr, **kwargs)]

    if isinstance(ref, np.ndarray):
        failed = len(result) != len(ref) or np.any(result != ref)
    else:
        failed = result != ref

    viscid.logger.debug("{0}: {1}".format(sel, result))
    if failed:
        s = ("Slice doesn't match reference\n"
             "  SLICE:     {0}\n"
             "  RESULT:    {1}\n"
             "  REFERENCE: {2}\n".format(sel, result, ref))
        raise RuntimeError(s)

def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = vutil.common_argparse(parser)  # pylint: disable=unused-variable

    arr = np.linspace(1.0, 10.0, 8)

    ###################
    # slice by integer

    test_slice(arr, np.s_[2], arr[2])
    # forward
    test_slice(arr, np.s_[4:], arr[4:])
    test_slice(arr, np.s_[2::3], arr[2::3])
    test_slice(arr, np.s_[:4], arr[:4])
    test_slice(arr, np.s_[:5:2], arr[:5:2])
    test_slice(arr, np.s_[2:5:2], arr[2:5:2])
    test_slice(arr, np.s_[6:5:1], arr[6:5:1])
    # backward
    test_slice(arr, np.s_[4::-1], arr[4::-1])
    test_slice(arr, np.s_[:4:-1], arr[:4:-1])
    test_slice(arr, np.s_[:4:-2], arr[:4:-2])
    test_slice(arr, np.s_[2:5:-2], arr[2:5:-2])
    test_slice(arr, np.s_[6:4:-1], arr[6:4:-1])
    # forward
    test_slice(arr, '4:', arr[4:])
    test_slice(arr, '2::3', arr[2::3])
    test_slice(arr, ':4', arr[:4])
    test_slice(arr, ':5:2', arr[:5:2])
    test_slice(arr, '2:5:2', arr[2:5:2])
    test_slice(arr, '6:5:1', arr[6:5:1])
    # backward
    test_slice(arr, '4::-1', arr[4::-1])
    test_slice(arr, ':4:-1', arr[:4:-1])
    test_slice(arr, ':4:-2', arr[:4:-2])
    test_slice(arr, '2:5:-2', arr[2:5:-2])
    test_slice(arr, '6:4:-1', arr[6:4:-1])

    #################
    # slice by float

    # Note: arr = [  1.           2.28571429   3.57142857   4.85714286
    #                6.14285714   7.42857143   8.71428571  10.        ]

    test_slice(arr, np.s_['4.0f'], arr[2])
    test_slice(arr, np.s_['4.0f':], arr[3:])
    test_slice(arr, np.s_['4.0f'::2], arr[3::2])
    test_slice(arr, np.s_[:'4.0f':2], arr[:3:2])
    test_slice(arr, np.s_['2.0f':'7.8f'], arr[1:6])
    test_slice(arr, np.s_['2.0f':'7.8f':2], arr[1:6:2])
    test_slice(arr, np.s_['7.8f':'2.0f':-1], arr[5:0:-1])
    test_slice(arr, np.s_['7.8f':'2.0f':-1], arr[5:1:-1], val_endpoint=False)
    test_slice(arr, np.s_['7.8f':'2.0f':-2], arr[5:0:-2])
    test_slice(arr, np.s_['7.8f':'2.0f':-2], arr[5:1:-2], val_endpoint=False)
    test_slice(arr, np.s_['3.4f':'7.3f'], arr[2:5])
    test_slice(arr, np.s_['3.4f':'7.3f'], arr[1:6], interior=True)
    test_slice(arr, np.s_['2.4f':'2.5f'], arr[2:2])
    test_slice(arr, np.s_['2.1f':'2.5f'], arr[1:2])
    test_slice(arr, np.s_['2.1f':'2.5f'], arr[1:1], val_endpoint=False)
    test_slice(arr, np.s_['2.3f':'2.5f'], arr[1:3], interior=True)

    ################
    # slice by imag
    test_slice(arr, np.s_[4.0j], arr[2])
    test_slice(arr, np.s_[4.0j:], arr[3:])
    test_slice(arr, np.s_[4.0j::2], arr[3::2])
    test_slice(arr, np.s_[:4.0j:2], arr[:3:2])
    test_slice(arr, np.s_[2.0j:7.8j], arr[1:6])
    test_slice(arr, np.s_[2.0j:7.8j:2], arr[1:6:2])
    test_slice(arr, np.s_[7.8j:2.0j:-1], arr[5:0:-1])
    test_slice(arr, np.s_[7.8j:2.0j:-1], arr[5:1:-1], val_endpoint=False)
    test_slice(arr, np.s_[7.8j:2.0j:-2], arr[5:0:-2])
    test_slice(arr, np.s_[7.8j:2.0j:-2], arr[5:1:-2], val_endpoint=False)
    test_slice(arr, np.s_[3.4j:7.3j], arr[2:5])
    test_slice(arr, np.s_[3.4j:7.3j], arr[1:6], interior=True)
    test_slice(arr, np.s_[2.4j:2.5j], arr[2:2])
    test_slice(arr, np.s_[2.1j:2.5j], arr[1:2])
    test_slice(arr, np.s_[2.1j:2.5j], arr[1:1], val_endpoint=False)
    test_slice(arr, np.s_[2.3j:2.5j], arr[1:3], interior=True)

    ####################
    # slice by imag str
    test_slice(arr, np.s_['4.0j'], arr[2])
    test_slice(arr, np.s_['4.0j':], arr[3:])
    test_slice(arr, np.s_['4.0j'::2], arr[3::2])
    test_slice(arr, np.s_[:'4.0j':2], arr[:3:2])
    test_slice(arr, np.s_['2.0j':'7.8j'], arr[1:6])
    test_slice(arr, np.s_['2.0j':'7.8j':2], arr[1:6:2])
    test_slice(arr, np.s_['7.8j':'2.0j':-1], arr[5:0:-1])
    test_slice(arr, np.s_['7.8j':'2.0j':-1], arr[5:1:-1], val_endpoint=False)
    test_slice(arr, np.s_['7.8j':'2.0j':-2], arr[5:0:-2])
    test_slice(arr, np.s_['7.8j':'2.0j':-2], arr[5:1:-2], val_endpoint=False)
    test_slice(arr, np.s_['3.4j':'7.3j'], arr[2:5])
    test_slice(arr, np.s_['3.4j':'7.3j'], arr[1:6], interior=True)
    test_slice(arr, np.s_['2.4j':'2.5j'], arr[2:2])
    test_slice(arr, np.s_['2.1j':'2.5j'], arr[1:2])
    test_slice(arr, np.s_['2.1j':'2.5j'], arr[1:1], val_endpoint=False)
    test_slice(arr, np.s_['2.3j':'2.5j'], arr[1:3], interior=True)

    ############################
    # slice by deprecated float
    viscid.logger.info("testing deprecated slice-by-location")
    test_slice(arr, np.s_['4.0'], arr[2])
    test_slice(arr, np.s_['4.0':], arr[3:])
    test_slice(arr, np.s_['4.0'::2], arr[3::2])
    test_slice(arr, np.s_[:'4.0':2], arr[:3:2])
    test_slice(arr, np.s_['2.0':'7.8'], arr[1:6])
    test_slice(arr, np.s_['2.0':'7.8':2], arr[1:6:2])
    test_slice(arr, np.s_['7.8':'2.0':-1], arr[5:0:-1])
    test_slice(arr, np.s_['7.8':'2.0':-1], arr[5:1:-1], val_endpoint=False)
    test_slice(arr, np.s_['7.8':'2.0':-2], arr[5:0:-2])
    test_slice(arr, np.s_['7.8':'2.0':-2], arr[5:1:-2], val_endpoint=False)
    test_slice(arr, np.s_['3.4':'7.3'], arr[2:5])
    test_slice(arr, np.s_['3.4':'7.3'], arr[1:6], interior=True)
    test_slice(arr, np.s_['2.4':'2.5'], arr[2:2])
    test_slice(arr, np.s_['2.1':'2.5'], arr[1:2])
    test_slice(arr, np.s_['2.1':'2.5'], arr[1:1], val_endpoint=False)
    test_slice(arr, np.s_['2.3':'2.5'], arr[1:3], interior=True)
    viscid.logger.info("done testing deprecated slice-by-location")

    assert viscid.selection2values(arr, np.s_[2:5]) == (3.5714285714285716,
                                                        7.4285714285714288)
    assert viscid.selection2values(None, np.s_[2:5]) == (np.nan, np.nan)
    assert viscid.selection2values(None, np.s_[:5]) == (-np.inf, np.nan)
    assert viscid.selection2values(None, np.s_[:5:-1]) == (np.inf, np.nan)
    assert viscid.selection2values(arr, '2.3f:2.5f') == (2.3, 2.5)
    assert viscid.selection2values(None, '2.3f:2.5f') == (2.3, 2.5)

    return 0


if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
