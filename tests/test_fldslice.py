#!/usr/bin/env python
"""Test a bunch of Field slice edge cases

Both cell and node centered fields are tested.
"""

from __future__ import print_function
import argparse
import sys

import numpy as np

import viscid_test_common  # pylint: disable=unused-import

import viscid
from viscid import vutil


def test_slice(selection, fld, dat_shape, nc_shape=None, cc_shape=None,
               data=None):
    slced_fld = fld[selection]

    assert nc_shape is None or tuple(slced_fld.crds.shape_nc) == nc_shape
    assert cc_shape is None or tuple(slced_fld.crds.shape_cc) == cc_shape
    assert slced_fld.data.shape == dat_shape

    if data:
        assert np.all(slced_fld == data)

def _main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    test_slice(selection, fld_f, (3, 1, 40), nc_shape=(2, 41), cc_shape=(1, 40))
    test_slice(selection, fld_i, (1, 40, 3), nc_shape=(2, 41), cc_shape=(1, 40))

    selection = "None, 1, ..., 2"
    test_slice(selection, fld, (1, 40), nc_shape=(2, 41), cc_shape=(1, 40))
    test_slice(selection, fld_f, (3, 1, 40), nc_shape=(2, 41), cc_shape=(1, 40))
    test_slice(selection, fld_i, (1, 40, 3), nc_shape=(2, 41), cc_shape=(1, 40))

    #### SLICE 2
    selection = np.s_[1, ..., None, 2]
    test_slice(selection, fld, (40, 1), nc_shape=(41, 2), cc_shape=(40, 1))
    test_slice(selection, fld_f, (3, 40, 1), nc_shape=(41, 2), cc_shape=(40, 1))
    test_slice(selection, fld_i, (40, 1, 3), nc_shape=(41, 2), cc_shape=(40, 1))

    selection = "1, ..., None, 2"
    test_slice(selection, fld, (40, 1), nc_shape=(41, 2), cc_shape=(40, 1))
    test_slice(selection, fld_f, (3, 40, 1), nc_shape=(41, 2), cc_shape=(40, 1))
    test_slice(selection, fld_i, (40, 1, 3), nc_shape=(41, 2), cc_shape=(40, 1))

    #### SLICE 3
    selection = np.s_[None, ..., None, 1]
    test_slice(selection, fld, (1, 30, 40, 1), nc_shape=(2, 31, 41, 2),
               cc_shape=(1, 30, 40, 1))
    test_slice(selection, fld_f, (3, 1, 30, 40, 1), nc_shape=(2, 31, 41, 2),
               cc_shape=(1, 30, 40, 1))
    test_slice(selection, fld_i, (1, 30, 40, 1, 3), nc_shape=(2, 31, 41, 2),
               cc_shape=(1, 30, 40, 1))

    selection = "None, ..., None, 1"
    test_slice(selection, fld, (1, 30, 40, 1), nc_shape=(2, 31, 41, 2),
               cc_shape=(1, 30, 40, 1))
    test_slice(selection, fld_f, (3, 1, 30, 40, 1), nc_shape=(2, 31, 41, 2),
               cc_shape=(1, 30, 40, 1))
    test_slice(selection, fld_i, (1, 30, 40, 1, 3), nc_shape=(2, 31, 41, 2),
               cc_shape=(1, 30, 40, 1))

    #### SLICE 4
    selection = np.s_[5j, ..., None, 2]
    test_slice(selection, fld, (40, 1), nc_shape=(41, 2), cc_shape=(40, 1))
    test_slice(selection, fld_f, (3, 40, 1), nc_shape=(41, 2), cc_shape=(40, 1))
    test_slice(selection, fld_i, (40, 1, 3), nc_shape=(41, 2), cc_shape=(40, 1))

    #### SLICE 4
    selection = "5j, ..., t=None, 2"
    test_slice(selection, fld, (40, 1), nc_shape=(41, 2), cc_shape=(40, 1))
    test_slice(selection, fld_f, (3, 40, 1), nc_shape=(41, 2), cc_shape=(40, 1))
    test_slice(selection, fld_i, (40, 1, 3), nc_shape=(41, 2), cc_shape=(40, 1))

    # with crd slice
    selection = np.s_[5j, ..., None, 2, 'x']
    test_slice(selection, fld_f, (40, 1), nc_shape=(41, 2), cc_shape=(40, 1))
    test_slice(selection, fld_i, (40, 1), nc_shape=(41, 2), cc_shape=(40, 1))

    selection = "5j, ..., t=None, 2, x"
    test_slice(selection, fld_f, (40, 1), nc_shape=(41, 2), cc_shape=(40, 1))
    test_slice(selection, fld_i, (40, 1), nc_shape=(41, 2), cc_shape=(40, 1))

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
    test_slice(selection, fld, (1, 41), nc_shape=(1, 41), cc_shape=(0, 40))
    test_slice(selection, fld_f, (3, 1, 41), nc_shape=(1, 41), cc_shape=(0, 40))
    test_slice(selection, fld_i, (1, 41, 3), nc_shape=(1, 41), cc_shape=(0, 40))

    selection = "None, 1, ..., 2"
    test_slice(selection, fld, (1, 41), nc_shape=(1, 41), cc_shape=(0, 40))
    test_slice(selection, fld_f, (3, 1, 41), nc_shape=(1, 41), cc_shape=(0, 40))
    test_slice(selection, fld_i, (1, 41, 3), nc_shape=(1, 41), cc_shape=(0, 40))

    #### SLICE 2
    selection = np.s_[1, ..., None, 2]
    test_slice(selection, fld, (41, 1), nc_shape=(41, 1), cc_shape=(40, 0))
    test_slice(selection, fld_f, (3, 41, 1), nc_shape=(41, 1), cc_shape=(40, 0))
    test_slice(selection, fld_i, (41, 1, 3), nc_shape=(41, 1), cc_shape=(40, 0))

    selection = "1, ..., None, 2"
    test_slice(selection, fld, (41, 1), nc_shape=(41, 1), cc_shape=(40, 0))
    test_slice(selection, fld_f, (3, 41, 1), nc_shape=(41, 1), cc_shape=(40, 0))
    test_slice(selection, fld_i, (41, 1, 3), nc_shape=(41, 1), cc_shape=(40, 0))

    #### SLICE 3
    selection = np.s_[None, ..., None, 1]
    test_slice(selection, fld, (1, 31, 41, 1), nc_shape=(1, 31, 41, 1),
               cc_shape=(0, 30, 40, 0))
    test_slice(selection, fld_f, (3, 1, 31, 41, 1), nc_shape=(1, 31, 41, 1),
               cc_shape=(0, 30, 40, 0))
    test_slice(selection, fld_i, (1, 31, 41, 1, 3), nc_shape=(1, 31, 41, 1),
               cc_shape=(0, 30, 40, 0))

    selection = "None, ..., None, 1"
    test_slice(selection, fld, (1, 31, 41, 1), nc_shape=(1, 31, 41, 1),
               cc_shape=(0, 30, 40, 0))
    test_slice(selection, fld_f, (3, 1, 31, 41, 1), nc_shape=(1, 31, 41, 1),
               cc_shape=(0, 30, 40, 0))
    test_slice(selection, fld_i, (1, 31, 41, 1, 3), nc_shape=(1, 31, 41, 1),
               cc_shape=(0, 30, 40, 0))

    #### SLICE 4
    selection = np.s_[5j, ..., None, 2]
    test_slice(selection, fld, (41, 1), nc_shape=(41, 1), cc_shape=(40, 0))
    test_slice(selection, fld_f, (3, 41, 1), nc_shape=(41, 1), cc_shape=(40, 0))
    test_slice(selection, fld_i, (41, 1, 3), nc_shape=(41, 1), cc_shape=(40, 0))

    #### SLICE 4
    selection = "5j, ..., t=None, 2"
    test_slice(selection, fld, (41, 1), nc_shape=(41, 1), cc_shape=(40, 0))
    test_slice(selection, fld_f, (3, 41, 1), nc_shape=(41, 1), cc_shape=(40, 0))
    test_slice(selection, fld_i, (41, 1, 3), nc_shape=(41, 1), cc_shape=(40, 0))

    # with crd slice
    selection = np.s_[5j, ..., None, 2, 'x']
    test_slice(selection, fld_f, (41, 1), nc_shape=(41, 1), cc_shape=(40, 0))
    test_slice(selection, fld_i, (41, 1), nc_shape=(41, 1), cc_shape=(40, 0))

    selection = "5j, ..., t=None, 2, x"
    test_slice(selection, fld_f, (41, 1), nc_shape=(41, 1), cc_shape=(40, 0))
    test_slice(selection, fld_i, (41, 1), nc_shape=(41, 1), cc_shape=(40, 0))

    ################################################################
    #### test newaxes with slice_and_keep dims (None == newaxis)
    fld_A = fld.slice_and_keep(np.s_[None, :, '0j', ..., None])
    assert fld_A.crds.axes == ['new-x0', 'x', 'y', 'z', 'new-x1']
    assert fld_A.shape == (1, 31, 1, 51, 1)

    fld_A = fld.slice_and_keep(np.s_[None, ..., '0j', None])
    assert fld_A.crds.axes == ['new-x0', 'x', 'y', 'z', 'new-x1']
    assert fld_A.shape == (1, 31, 41, 1, 1)

    fld_A = fld.slice_and_keep(np.s_[None, :, '0j', None])
    assert fld_A.crds.axes == ['new-x0', 'x', 'y', 'new-x1', 'z']
    assert fld_A.shape == (1, 31, 1, 1, 51)

    fld_A = fld.slice_and_keep(np.s_[None, :, ..., None])
    assert fld_A.crds.axes == ['new-x0', 'x', 'y', 'z', 'new-x1']
    assert fld_A.shape == (1, 31, 41, 51, 1)

    fld_A = fld.slice_and_keep(np.s_[None, :, None, ...])
    assert fld_A.crds.axes == ['new-x0', 'x', 'new-x1', 'y', 'z']
    assert fld_A.shape == (1, 31, 1, 41, 51)

    fld_A = fld.slice_and_keep(np.s_[None, :, '0j', None, ...])
    assert fld_A.crds.axes == ['new-x0', 'x', 'y', 'new-x1', 'z']
    assert fld_A.shape == (1, 31, 1, 1, 51)

    # -
    # assorted other quick tests
    def _quick_test1(_fld, selection, axes, shape, verb=False):
        slcfld = _fld[selection]
        if verb:
            print("axes:", slcfld.crds.axes)
            print("shape:", slcfld.shape, '\n')
        assert slcfld.crds.axes == axes
        assert slcfld.shape == shape

    _quick_test1(fld, ':', ['x', 'y', 'z'], (31, 41, 51))
    _quick_test1(fld, ':, w=newaxis, z=0.0j:0.3j', ['x', 'w', 'y', 'z'],
                 (31, 1, 41, 1))
    _quick_test1(fld, ':, w=newaxis, 0.0j, v=newaxis, z=0.0j:0.3j',
                 ['x', 'w', 'v', 'z'], (31, 1, 1, 1))
    _quick_test1(fld, ':, w=newaxis, ..., 0.0j:0.3j', ['x', 'w', 'y', 'z'],
                 (31, 1, 41, 1))
    _quick_test1(fld, '..., :, w=newaxis, 0.0j:0.3j', ['x', 'y', 'w', 'z'],
                 (31, 41, 1, 1))
    _quick_test1(fld, 'u=newaxis, ..., w=newaxis', ['u', 'x', 'y', 'z', 'w'],
                 (1, 31, 41, 51, 1))
    _quick_test1(fld, 'newaxis, ..., w=newaxis', ['new-x0', 'x', 'y', 'z', 'w'],
                 (1, 31, 41, 51, 1))
    _quick_test1(fld, 'u=newaxis, ..., newaxis', ['u', 'x', 'y', 'z', 'new-x0'],
                 (1, 31, 41, 51, 1))

    # -
    #       implied x

    fld_A = fld.slice_and_keep('u=newaxis, :, 0j, ..., w=newaxis')
    assert fld_A.crds.axes == ['u', 'x', 'y', 'z', 'w']
    assert fld_A.shape == (1, 31, 1, 51, 1)

    fld_A = fld.slice_and_keep('u=newaxis, :, 0j, w=newaxis')
    assert fld_A.crds.axes == ['u', 'x', 'y', 'w', 'z']
    assert fld_A.shape == (1, 31, 1, 1, 51)

    fld_A = fld.slice_and_keep('u=newaxis, :, 0j, w=newaxis, ...')
    assert fld_A.crds.axes == ['u', 'x', 'y', 'w', 'z']
    assert fld_A.shape == (1, 31, 1, 1, 51)

    # test slice_interp
    shape = [31, 41, 51]
    fld = viscid.dat2field(np.arange(np.prod(shape)).astype('f4').reshape(shape),
                           center='node')
    fld_A = fld.interpolated_slice(np.s_[:, 10.5])
    fld_A = fld.interpolated_slice('y=10.5')
    fld_A = fld.interpolated_slice('..., 10.5j')

    shape = [30, 40, 50]
    fld = viscid.dat2field(np.arange(np.prod(shape)).astype('f4').reshape(shape),
                           center='cell')
    fld_A = fld.interpolated_slice(np.s_[:, 10.5])
    fld_A = fld.interpolated_slice('y=10.5')
    fld_A = fld.interpolated_slice('..., 10.5j')

    return 0


if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
