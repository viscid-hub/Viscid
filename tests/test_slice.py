#!/usr/bin/env python
""" kick the tires on making matplotlib plots """

from __future__ import print_function
import argparse

import numpy as np

import viscid_test_common  # pylint: disable=unused-import

import viscid
from viscid import vutil

def test_slice(selection, fld, dat_shape, nc_shape=None, cc_shape=None,
               data=None):
    slced_fld = fld[selection]

    assert nc_shape is None or all(slced_fld.crds.shape_nc == nc_shape)
    assert cc_shape is None or all(slced_fld.crds.shape_cc == cc_shape)
    assert slced_fld.data.shape == dat_shape

    if data:
        assert np.all(slced_fld == data)

def main():
    parser = argparse.ArgumentParser(description="Test slice")
    args = vutil.common_argparse(parser)  # pylint: disable=unused-variable

    # CELL CENTERED TESTS
    shape = [30, 40, 50]
    center = 'cell'
    fld = viscid.dat2field(np.arange(np.prod(shape)).reshape(shape),
                           center=center)
    fld_f = viscid.dat2field(np.arange(np.prod([3] + shape)).reshape([3] + shape),
                             fldtype='vector', layout='flat', center=center)
    fld_i = viscid.dat2field(np.arange(np.prod(shape + [3])).reshape(shape + [3]),
                             fldtype='vector', layout='interlaced', center=center)

    #### SLICE 1
    selection = np.s_[None, 1, ..., 2]
    test_slice(selection, fld, (1, 40), nc_shape=(2, 41), cc_shape=(1, 40))
    test_slice(selection, fld_f, (1, 30, 40), nc_shape=(2, 31, 41), cc_shape=(1, 30, 40))
    test_slice(selection, fld_i, (1, 40, 50), nc_shape=(2, 41, 51), cc_shape=(1, 40, 50))

    selection = "None, 1, ..., 2"
    test_slice(selection, fld, (1, 40), nc_shape=(2, 41), cc_shape=(1, 40))
    test_slice(selection, fld_f, (1, 30, 40), nc_shape=(2, 31, 41), cc_shape=(1, 30, 40))
    test_slice(selection, fld_i, (1, 40, 50), nc_shape=(2, 41, 51), cc_shape=(1, 40, 50))

    #### SLICE 2
    selection = np.s_[1, ..., None, 2]
    test_slice(selection, fld, (40, 1), nc_shape=(41, 2), cc_shape=(40, 1))
    test_slice(selection, fld_f, (30, 40, 1), nc_shape=(31, 41, 2), cc_shape=(30, 40, 1))
    test_slice(selection, fld_i, (40, 50, 1), nc_shape=(41, 51, 2), cc_shape=(40, 50, 1))

    selection = "1, ..., None, 2"
    test_slice(selection, fld, (40, 1), nc_shape=(41, 2), cc_shape=(40, 1))
    test_slice(selection, fld_f, (30, 40, 1), nc_shape=(31, 41, 2), cc_shape=(30, 40, 1))
    test_slice(selection, fld_i, (40, 50, 1), nc_shape=(41, 51, 2), cc_shape=(40, 50, 1))

    #### SLICE 3
    selection = np.s_[None, ..., None, 1]
    test_slice(selection, fld, (1, 30, 40, 1), nc_shape=(2, 31, 41, 2),
               cc_shape=(1, 30, 40, 1))
    test_slice(selection, fld_f, (3, 1, 30, 40, 1), nc_shape=(2, 31, 41, 2),
               cc_shape=(1, 30, 40, 1))
    test_slice(selection, fld_i, (1, 30, 40, 50, 1), nc_shape=(2, 31, 41, 51, 2),
               cc_shape=(1, 30, 40, 50, 1))

    selection = "None, ..., None, 1"
    test_slice(selection, fld, (1, 30, 40, 1), nc_shape=(2, 31, 41, 2),
               cc_shape=(1, 30, 40, 1))
    test_slice(selection, fld_f, (3, 1, 30, 40, 1), nc_shape=(2, 31, 41, 2),
               cc_shape=(1, 30, 40, 1))
    test_slice(selection, fld_i, (1, 30, 40, 50, 1), nc_shape=(2, 31, 41, 51, 2),
               cc_shape=(1, 30, 40, 50, 1))

    #### SLICE 4
    selection = "x=0, ..., None, 2"
    test_slice(selection, fld, (40, 1), nc_shape=(41, 2), cc_shape=(40, 1))
    test_slice(selection, fld_f, (3, 40, 1), nc_shape=(41, 2), cc_shape=(40, 1))
    test_slice(selection, fld_i, (40, 50, 1), nc_shape=(41, 51, 2), cc_shape=(40, 50, 1))

    #### SLICE 4
    selection = "x=5f, ..., t=None, 2"
    test_slice(selection, fld, (40, 1), nc_shape=(41, 2), cc_shape=(40, 1))
    test_slice(selection, fld_f, (3, 40, 1), nc_shape=(41, 2), cc_shape=(40, 1))
    test_slice(selection, fld_i, (40, 50, 1), nc_shape=(41, 51, 2), cc_shape=(40, 50, 1))

    # NODE CENTERED TESTS
    shape = [31, 41, 51]
    center = 'node'
    fld = viscid.dat2field(np.arange(np.prod(shape)).reshape(shape),
                           center=center)
    fld_f = viscid.dat2field(np.arange(np.prod([3] + shape)).reshape([3] + shape),
                             fldtype='vector', layout='flat', center=center)
    fld_i = viscid.dat2field(np.arange(np.prod(shape + [3])).reshape(shape + [3]),
                             fldtype='vector', layout='interlaced', center=center)

    #### SLICE 1
    selection = np.s_[None, 1, ..., 2]
    test_slice(selection, fld, (1, 41), nc_shape=(1, 41), cc_shape=(1, 40))
    test_slice(selection, fld_f, (1, 31, 41), nc_shape=(1, 31, 41), cc_shape=(1, 30, 40))
    test_slice(selection, fld_i, (1, 41, 51), nc_shape=(1, 41, 51), cc_shape=(1, 40, 50))

    selection = "None, 1, ..., 2"
    test_slice(selection, fld, (1, 41), nc_shape=(1, 41), cc_shape=(1, 40))
    test_slice(selection, fld_f, (1, 31, 41), nc_shape=(1, 31, 41), cc_shape=(1, 30, 40))
    test_slice(selection, fld_i, (1, 41, 51), nc_shape=(1, 41, 51), cc_shape=(1, 40, 50))

    #### SLICE 2
    selection = np.s_[1, ..., None, 2]
    test_slice(selection, fld, (41, 1), nc_shape=(41, 1), cc_shape=(40, 1))
    test_slice(selection, fld_f, (31, 41, 1), nc_shape=(31, 41, 1), cc_shape=(30, 40, 1))
    test_slice(selection, fld_i, (41, 51, 1), nc_shape=(41, 51, 1), cc_shape=(40, 50, 1))

    selection = "1, ..., None, 2"
    test_slice(selection, fld, (41, 1), nc_shape=(41, 1), cc_shape=(40, 1))
    test_slice(selection, fld_f, (31, 41, 1), nc_shape=(31, 41, 1), cc_shape=(30, 40, 1))
    test_slice(selection, fld_i, (41, 51, 1), nc_shape=(41, 51, 1), cc_shape=(40, 50, 1))

    #### SLICE 3
    selection = np.s_[None, ..., None, 1]
    test_slice(selection, fld, (1, 31, 41, 1), nc_shape=(1, 31, 41, 1),
               cc_shape=(1, 30, 40, 1))
    test_slice(selection, fld_f, (3, 1, 31, 41, 1), nc_shape=(1, 31, 41, 1),
               cc_shape=(1, 30, 40, 1))
    test_slice(selection, fld_i, (1, 31, 41, 51, 1), nc_shape=(1, 31, 41, 51, 1),
               cc_shape=(1, 30, 40, 50, 1))

    selection = "None, ..., None, 1"
    test_slice(selection, fld, (1, 31, 41, 1), nc_shape=(1, 31, 41, 1),
               cc_shape=(1, 30, 40, 1))
    test_slice(selection, fld_f, (3, 1, 31, 41, 1), nc_shape=(1, 31, 41, 1),
               cc_shape=(1, 30, 40, 1))
    test_slice(selection, fld_i, (1, 31, 41, 51, 1), nc_shape=(1, 31, 41, 51, 1),
               cc_shape=(1, 30, 40, 50, 1))

    #### SLICE 4
    selection = "x=0, ..., None, 2"
    test_slice(selection, fld, (41, 1), nc_shape=(41, 1), cc_shape=(40, 1))
    test_slice(selection, fld_f, (3, 41, 1), nc_shape=(41, 1), cc_shape=(40, 1))
    test_slice(selection, fld_i, (41, 51, 1), nc_shape=(41, 51, 1), cc_shape=(40, 50, 1))

    #### SLICE 4
    selection = "x=5f, ..., t=None, 2"
    test_slice(selection, fld, (41, 1), nc_shape=(41, 1), cc_shape=(40, 1))
    test_slice(selection, fld_f, (3, 41, 1), nc_shape=(41, 1), cc_shape=(40, 1))
    test_slice(selection, fld_i, (41, 51, 1), nc_shape=(41, 51, 1), cc_shape=(40, 50, 1))


if __name__ == "__main__":
    main()

##
## EOF
##
