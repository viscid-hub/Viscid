#!/usr/bin/env python

# pylint: disable=invalid-slice-index

from __future__ import division, print_function
import sys

import numpy as np
import viscid


def check_nd_sel(fld, sel, good_sel_list=None, good_sel_names=None,
                 good_sel_newmask=None, good_comp_sel=slice(None)):
    sel_list = viscid.raw_sel2sel_list(sel)
    sel_list, comp_sel = viscid.prune_comp_sel(sel_list,
                                               comp_names=fld.comp_names)
    r = viscid.fill_nd_sel_list(sel_list, fld.crds.axes)
    full_sel_list, full_names, full_newdim = r

    if tuple(full_sel_list) != good_sel_list:
        print("ERROR ERROR ERROR ERROR")
        print("-----------------------")
        print("sel:", sel)
        print("sel_list:", sel_list)
        print("full_sel_list:", full_sel_list)
        print("good_sel_list:", good_sel_list)
        raise RuntimeError('sel list')

    if full_names != good_sel_names:
        print("ERROR ERROR ERROR ERROR")
        print("-----------------------")
        print("sel:", sel)
        print("sel_list:", sel_list)
        print("full_names:", full_names)
        print("good_sel_names:", good_sel_names)
        raise RuntimeError('sel names')

    if full_newdim != good_sel_newmask:
        print("ERROR ERROR ERROR ERROR")
        print("-----------------------")
        print("sel:", sel)
        print("sel_list:", sel_list)
        print("full_newdim:", full_newdim)
        print("good_sel_newmask:", good_sel_newmask)
        raise RuntimeError('full_newdim')

    if comp_sel != good_comp_sel:
        print("ERROR ERROR ERROR ERROR")
        print("-----------------------")
        print("sel:", sel)
        print("sel_list:", sel_list)
        print("comp_sel:", comp_sel)
        print("good_comp_sel:", good_comp_sel)
        raise RuntimeError('comp_sel')

def check_slc_val(val, check_val):
    if isinstance(check_val, np.ndarray):
        assert np.all(viscid.standardize_sel(val) == check_val)
    else:
        assert viscid.standardize_sel(val) == check_val

def check_sbv(slc, arr, check_slc, val_endpoint=True, interior=False):
    ret = viscid.std_sel2index(slc, arr, val_endpoint=val_endpoint,
                               interior=interior)
    # print('??', ret)
    assert ret == check_slc

def check_crd_slice(crds, selection, xl, xh, shape, axes, variant=''):
    if variant.strip().lower() == 'keep':
        crds2 = crds.slice_and_keep(selection)
    elif variant.strip().lower() == 'reduce':
        crds2 = crds.slice_and_reduce(selection)
    elif variant.strip().lower() == '':
        crds2 = crds.slice(selection)
    else:
        raise ValueError(variant)

    assert np.allclose(crds2.xl_nc, xl)
    assert np.allclose(crds2.xh_nc, xh)
    assert np.all(crds2.shape_nc == shape)
    assert all(a == b for a, b in zip(axes, crds2.axes))

def check_fld_slice(fld, selection, xl, xh, shape, axes, variant=''):
    if variant.strip().lower() == 'keep':
        fld2 = fld.slice_and_keep(selection)
    elif variant.strip().lower() == 'reduce':
        fld2 = fld.slice_and_reduce(selection)
    elif variant.strip().lower() == '':
        fld2 = fld[selection]
    else:
        raise ValueError(variant)

    assert np.allclose(fld2.xl, xl)
    assert np.allclose(fld2.xh, xh)
    assert np.all(fld2.shape == shape)
    assert all(a == b for a, b in zip(axes, fld2.crds.axes))


def _main():
    fld = viscid.mfield[-4:4:24j, -5:5:16j, -6:6:20j]
    fld_f = viscid.scalar_fields_to_vector([fld, fld, fld], layout='flat')
    fld_i = fld_f.as_layout('interlaced')

    check_nd_sel(fld_f,
                 np.s_[...],
                 good_sel_list=np.s_[:, :, :],
                 good_sel_names=['x', 'y', 'z'],
                 good_sel_newmask=[False, False, False],
                 good_comp_sel=slice(None),
                 )

    check_nd_sel(fld_f,
                 np.s_[0],
                 good_sel_list=np.s_[0, :, :],
                 good_sel_names=['x', 'y', 'z'],
                 good_sel_newmask=[False, False, False],
                 good_comp_sel=slice(None),
                 )

    check_nd_sel(fld_f,
                 np.s_[..., :3j],
                 good_sel_list=np.s_[:, :, :3j],
                 good_sel_names=['x', 'y', 'z'],
                 good_sel_newmask=[False, False, False],
                 good_comp_sel=slice(None),
                 )

    check_nd_sel(fld_f,
                 np.s_[..., None, :3j],
                 good_sel_list=np.s_[:, :, np.newaxis, :3j],
                 good_sel_names=['x', 'y', 'new-x0', 'z'],
                 good_sel_newmask=[False, False, True, False],
                 good_comp_sel=slice(None),
                 )

    check_nd_sel(fld_f,
                 np.s_[..., :3j, None],
                 good_sel_list=np.s_[:, :, :3j, np.newaxis],
                 good_sel_names=['x', 'y', 'z', 'new-x0'],
                 good_sel_newmask=[False, False, False, True],
                 good_comp_sel=slice(None),
                 )

    check_nd_sel(fld_f,
                 np.s_[..., None, :3j, None],
                 good_sel_list=np.s_[:, :, np.newaxis, :3j, np.newaxis],
                 good_sel_names=['x', 'y', 'new-x0', 'z', 'new-x1'],
                 good_sel_newmask=[False, False, True, False, True],
                 good_comp_sel=slice(None),
                 )

    check_nd_sel(fld_f,
                 np.s_[None, 'u=None', ..., None, :3j, None],
                 good_sel_list=np.s_[np.newaxis, np.newaxis, :, :, np.newaxis,
                                     :3j, np.newaxis],
                 good_sel_names=['new-x0', 'u', 'x', 'y', 'new-x1', 'z', 'new-x2'],
                 good_sel_newmask=[True, True, False, False, True, False, True],
                 good_comp_sel=slice(None),
                 )

    check_nd_sel(fld_f,
                 np.s_[:3j, ...],
                 good_sel_list=np.s_[:3j, :, :],
                 good_sel_names=['x', 'y', 'z'],
                 good_sel_newmask=[False, False, False],
                 good_comp_sel=slice(None),
                 )

    check_nd_sel(fld_f,
                 np.s_[0, ..., :3j],
                 good_sel_list=np.s_[0, :, :3j],
                 good_sel_names=['x', 'y', 'z'],
                 good_sel_newmask=[False, False, False],
                 good_comp_sel=slice(None),
                 )

    check_nd_sel(fld_f,
                 np.s_[0, 'u=newaxis', ..., :3j],
                 good_sel_list=np.s_[0, np.newaxis, :, :3j],
                 good_sel_names=['x', 'u', 'y', 'z'],
                 good_sel_newmask=[False, True, False, False],
                 good_comp_sel=slice(None),
                 )

    check_nd_sel(fld_f,
                 np.s_[0, np.newaxis, ..., :3j],
                 good_sel_list=np.s_[0, np.newaxis, :, :3j],
                 good_sel_names=['x', 'new-x0', 'y', 'z'],
                 good_sel_newmask=[False, True, False, False],
                 good_comp_sel=slice(None),
                 )

    check_nd_sel(fld_f,
                 np.s_[0, ..., 'u=newaxis', :3j],
                 good_sel_list=np.s_[0, :, np.newaxis, :3j],
                 good_sel_names=['x', 'y', 'u', 'z'],
                 good_sel_newmask=[False, False, True, False],
                 good_comp_sel=slice(None),
                 )

    check_nd_sel(fld_f,
                 np.s_[0, ..., np.newaxis, :3j],
                 good_sel_list=np.s_[0, :, np.newaxis, :3j],
                 good_sel_names=['x', 'y', 'new-x0', 'z'],
                 good_sel_newmask=[False, False, True, False],
                 good_comp_sel=slice(None),
                 )

    check_nd_sel(fld_f,
                 np.s_['z=4:10, x=:3'],
                 good_sel_list=np.s_[':3', :, '4:10'],
                 good_sel_names=['x', 'y', 'z'],
                 good_sel_newmask=[False, False, False],
                 good_comp_sel=slice(None),
                 )

    check_nd_sel(fld_f,
                 np.s_['z', None, 'u=None', :, :, None, 'z=:3j', None],
                 good_sel_list=np.s_[np.newaxis, np.newaxis, :, :, np.newaxis,
                                     ':3j', np.newaxis],
                 good_sel_names=['new-x0', 'u', 'x', 'y', 'new-x1', 'z', 'new-x2'],
                 good_sel_newmask=[True, True, False, False, True, False, True],
                 good_comp_sel=2,
                 )

    check_nd_sel(fld_f,
                 np.s_['z=4:10, newaxis, x=:3'],
                 good_sel_list=np.s_[':3', :, '4:10', np.newaxis],
                 good_sel_names=['x', 'y', 'z', 'new-x0'],
                 good_sel_newmask=[False, False, False, True],
                 good_comp_sel=slice(None),
                 )

    check_nd_sel(fld_f,
                 np.s_['z=4:10, u=newaxis, x=:3'],
                 good_sel_list=np.s_[':3', :, '4:10', np.newaxis],
                 good_sel_names=['x', 'y', 'z', 'u'],
                 good_sel_newmask=[False, False, False, True],
                 good_comp_sel=slice(None),
                 )

    check_nd_sel(fld_f,
                 np.s_['u=newaxis, z=4:10, x=:3'],
                 good_sel_list=np.s_[np.newaxis, ':3', :, '4:10'],
                 good_sel_names=['u', 'x', 'y', 'z'],
                 good_sel_newmask=[True, False, False, False],
                 good_comp_sel=slice(None),
                 )

    check_nd_sel(fld_f,
                 np.s_['z=4:10, x=:3, u=newaxis'],
                 good_sel_list=np.s_[':3', :, '4:10', np.newaxis],
                 good_sel_names=['x', 'y', 'z', 'u'],
                 good_sel_newmask=[False, False, False, True],
                 good_comp_sel=slice(None),
                 )

    check_slc_val('UT2010-01-01', np.datetime64('2010-01-01'))
    check_slc_val('T2010-01-01:T60:12:01.0', slice(viscid.as_datetime64('2010-01-01'),
                                                   viscid.as_timedelta64('60:12:01.0')))
    check_slc_val('[2010-01-01, 2011-02-02]', viscid.as_datetime64(['2010-01-01',
                                                                    '2011-02-02']))
    check_slc_val('[T31, 12:20.1]', viscid.as_timedelta64(['31', '12:20.1']))
    check_slc_val('[T31, T12]', viscid.as_timedelta64(['31', '12']))
    check_slc_val('13j:12:3', np.s_[13j:12:3])
    check_slc_val(np.s_[13j:12:3], np.s_[13j:12:3])
    check_slc_val(np.s_[13j:'T12':3], np.s_[13j:viscid.as_timedelta64(12):3])
    check_slc_val(np.s_[13j:'2010-01-01':3],
                  np.s_[13j:viscid.as_datetime64('2010-01-01'):3])
    check_slc_val(np.s_[13j:'None':3], np.s_[13j:None:3])

    # ret = viscid.std_sel2index(np.s_[:3], x, np.s_[:3])
    x = np.array([-1, -0.5, 0, 0.5, 1])

    # start, step > 0
    check_sbv(np.s_[-1.0j:], x, np.s_[0:])
    check_sbv(np.s_[-0.9999999j:], x, np.s_[0:])
    check_sbv(np.s_[-0.9999j:], x, np.s_[1:])
    check_sbv(np.s_[-0.8j:], x, np.s_[1:])
    check_sbv(np.s_[-0.6j:], x, np.s_[1:])
    check_sbv(np.s_[0.5j:], x, np.s_[3:])
    # start, step < 0
    check_sbv(np.s_[-1.0j::-1], x, np.s_[0::-1])
    check_sbv(np.s_[-0.51j::-1], x, np.s_[0::-1])
    check_sbv(np.s_[-0.50000001j::-1], x, np.s_[1::-1])
    check_sbv(np.s_[-0.5j::-1], x, np.s_[1::-1])
    check_sbv(np.s_[-0.4j::-1], x, np.s_[1::-1])
    check_sbv(np.s_[-0.6j::-1], x, np.s_[0::-1])
    check_sbv(np.s_[0.5j::-1], x, np.s_[3::-1])
    # stop, step > 0
    check_sbv(np.s_[:-1.1j], x, np.s_[:0])
    check_sbv(np.s_[:-1j], x, np.s_[:1])
    check_sbv(np.s_[:-0.51j], x, np.s_[:1])
    check_sbv(np.s_[:-0.5000001j], x, np.s_[:2])
    check_sbv(np.s_[:-0.5j], x, np.s_[:2])
    check_sbv(np.s_[:-0.48j], x, np.s_[:2])
    # stop, step < 0
    check_sbv(np.s_[:-1.0j:-1], x, np.s_[::-1])
    check_sbv(np.s_[:-0.51j:-1], x, np.s_[:0:-1])
    check_sbv(np.s_[:-0.5j:-1], x, np.s_[:0:-1])
    check_sbv(np.s_[:-0.499999j:-1], x, np.s_[:0:-1])
    check_sbv(np.s_[:-0.49j:-1], x, np.s_[:1:-1])
    # length-zero slice
    check_sbv(np.s_[:-10j:1], x, np.s_[:0:1])
    check_sbv(np.s_[:10j:-1], x, np.s_[:5:-1])

    crds = fld.crds
    check_crd_slice(crds, np.s_[-2j:2j, 0j:, :0j],
                    (-1.91304348, 0.33333, -6.0),
                    (1.91304348, 5.0, -0.31578947),
                    (12, 8, 10),
                    ('x', 'y', 'z')
                    )

    check_crd_slice(crds, np.s_[-2j:2j, 0j, :0j],
                    (-1.91304348, -6.0),
                    (1.91304348, -0.31578947),
                    (12, 10),
                    ('x', 'z')
                    )

    check_crd_slice(crds, np.s_[-2j:2j, 0j, :0j],
                    (-1.91304348, 0.33333, -6.0),
                    (1.91304348, 0.33333, -0.31578947),
                    (12, 1, 10),
                    ('x', 'y', 'z'),
                    variant='keep'
                    )

    check_crd_slice(crds, np.s_[-2j:2j, 0j, 'w=newaxis', :0j],
                    (-1.91304348, 0.33333, 0.0, -6.0),
                    (1.91304348, 0.33333, 0.0, -0.31578947),
                    (12, 1, 1, 10),
                    ('x', 'y', 'w', 'z'),
                    variant='keep'
                    )

    check_crd_slice(crds.slice_and_keep(np.s_[0j]), np.s_[..., 0j],
                    (-5.0,), (5.0,), (16,), ('y'),
                    variant='reduce'
                    )

    check_fld_slice(fld, np.s_[-2j:2j, 0j:, :0j],
                    (-1.91304348, 0.33333, -6.0),
                    (1.91304348, 5.0, -0.31578947),
                    (12, 8, 10),
                    ('x', 'y', 'z')
                    )

    check_fld_slice(fld, np.s_[-2j:2j, 0j:, 0j],
                    (-1.91304348, 0.33333, -0.3157894),
                    (1.91304348, 5.0, -0.31578947),
                    (12, 8, 1),
                    ('x', 'y', 'z'),
                    variant='keep'
                    )

    check_fld_slice(fld.slice_and_keep(np.s_[:, 5j]),
                    np.s_[-2j:2j, :, 0j],
                    (-1.91304348,),
                    (1.91304348,),
                    (12,),
                    ('x',),
                    variant='reduce'
                    )

    # fld = viscid.mfield[-4:4:24j, -5:5:16j, -6:6:20j]
    x = fld.get_crd('x')

    # check slice-by-in-array
    xslc = [1, 3, 5, 7]

    assert np.allclose(fld[xslc].get_crd('x'), x[xslc])
    assert np.allclose(fld[np.array(xslc)].get_crd('x'), x[xslc])
    assert np.allclose(fld[str(xslc)].get_crd('x'), x[xslc])

    bool_arr = np.zeros((fld.shape[0]), dtype=np.bool_)
    bool_arr[xslc] = True
    assert np.allclose(fld[list(bool_arr)].get_crd('x'), x[xslc])
    assert np.allclose(fld[bool_arr].get_crd('x'), x[xslc])
    assert np.allclose(fld[str(bool_arr)].get_crd('x'), x[xslc])

    val_arr = 1j * x[xslc]
    assert np.allclose(fld[list(val_arr)].get_crd('x'), x[xslc])
    assert np.allclose(fld[val_arr].get_crd('x'), x[xslc])
    assert np.allclose(fld[str(val_arr)].get_crd('x'), x[xslc])

    assert isinstance(fld[list(val_arr)].crds, viscid.coordinate.UniformCrds)
    assert isinstance(fld[val_arr].crds, viscid.coordinate.UniformCrds)
    assert isinstance(fld[str(val_arr)].crds, viscid.coordinate.UniformCrds)

    # will turn uniform into non-uniform
    xslc = [1, 3, 5, 8]

    assert np.allclose(fld[xslc].get_crd('x'), x[xslc])
    assert np.allclose(fld[np.array(xslc)].get_crd('x'), x[xslc])
    assert np.allclose(fld[str(xslc)].get_crd('x'), x[xslc])

    bool_arr = np.zeros((fld.shape[0]), dtype=np.bool_)
    bool_arr[xslc] = True
    assert np.allclose(fld[list(bool_arr)].get_crd('x'), x[xslc])
    assert np.allclose(fld[bool_arr].get_crd('x'), x[xslc])
    assert np.allclose(fld[str(bool_arr)].get_crd('x'), x[xslc])

    val_arr = 1j * x[xslc]
    assert np.allclose(fld[list(val_arr)].get_crd('x'), x[xslc])
    assert np.allclose(fld[val_arr].get_crd('x'), x[xslc])
    assert np.allclose(fld[str(val_arr)].get_crd('x'), x[xslc])

    assert isinstance(fld[list(val_arr)].crds, viscid.coordinate.NonuniformCrds)
    assert isinstance(fld[val_arr].crds, viscid.coordinate.NonuniformCrds)
    assert isinstance(fld[str(val_arr)].crds, viscid.coordinate.NonuniformCrds)

    ########### cell centered....
    fld_cc = fld.as_centered('cell')

    x_cc = fld_cc.get_crd('x')

    # check slice-by-in-array
    xslc = [1, 3, 5, 7]

    assert np.allclose(fld_cc[xslc].get_crd('x'), x_cc[xslc])
    assert np.allclose(fld_cc[np.array(xslc)].get_crd('x'), x_cc[xslc])
    assert np.allclose(fld_cc[str(xslc)].get_crd('x'), x_cc[xslc])

    bool_arr = np.zeros((fld_cc.shape[0]), dtype=np.bool_)
    bool_arr[xslc] = True
    assert np.allclose(fld_cc[list(bool_arr)].get_crd('x'), x_cc[xslc])
    assert np.allclose(fld_cc[bool_arr].get_crd('x'), x_cc[xslc])
    assert np.allclose(fld_cc[str(bool_arr)].get_crd('x'), x_cc[xslc])

    val_arr = 1j * x_cc[xslc]
    assert np.allclose(fld_cc[list(val_arr)].get_crd('x'), x_cc[xslc])
    assert np.allclose(fld_cc[val_arr].get_crd('x'), x_cc[xslc])
    assert np.allclose(fld_cc[str(val_arr)].get_crd('x'), x_cc[xslc])

    assert isinstance(fld_cc[list(val_arr)].crds, viscid.coordinate.UniformCrds)
    assert isinstance(fld_cc[val_arr].crds, viscid.coordinate.UniformCrds)
    assert isinstance(fld_cc[str(val_arr)].crds, viscid.coordinate.UniformCrds)

    # will turn uniform into non-uniform
    xslc = [1, 3, 5, 8]

    assert np.allclose(fld_cc[xslc].get_crd('x'), x_cc[xslc])
    assert np.allclose(fld_cc[np.array(xslc)].get_crd('x'), x_cc[xslc])
    assert np.allclose(fld_cc[str(xslc)].get_crd('x'), x_cc[xslc])

    bool_arr = np.zeros((fld_cc.shape[0]), dtype=np.bool_)
    bool_arr[xslc] = True
    assert np.allclose(fld_cc[list(bool_arr)].get_crd('x'), x_cc[xslc])
    assert np.allclose(fld_cc[bool_arr].get_crd('x'), x_cc[xslc])
    assert np.allclose(fld_cc[str(bool_arr)].get_crd('x'), x_cc[xslc])

    val_arr = 1j * x_cc[xslc]
    assert np.allclose(fld_cc[list(val_arr)].get_crd('x'), x_cc[xslc])
    assert np.allclose(fld_cc[val_arr].get_crd('x'), x_cc[xslc])
    assert np.allclose(fld_cc[str(val_arr)].get_crd('x'), x_cc[xslc])

    assert isinstance(fld_cc[list(val_arr)].crds, viscid.coordinate.NonuniformCrds)
    assert isinstance(fld_cc[val_arr].crds, viscid.coordinate.NonuniformCrds)
    assert isinstance(fld_cc[str(val_arr)].crds, viscid.coordinate.NonuniformCrds)

    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
