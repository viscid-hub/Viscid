#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import numexpr as ne

import viscid
from viscid import field
from viscid import coordinate

# ne.set_num_threads(1)  # for performance checking


def _normalize_scalar_dtype(s, arrs):
    # cast python scalars to an appropriate numpy dtype
    if isinstance(s, (int, float, complex)):
        ndarrs = [_a for _a in arrs if hasattr(_a, 'dtype')]
        flt_arrs = [_a for _a in ndarrs if _a.dtype.kind in 'fc']
        int_arrs = [_a for _a in ndarrs if _a.dtype.kind in 'i']
        if flt_arrs and isinstance(s, (int, float, complex)):
            s = np.asarray(s).astype(np.common_type(*flt_arrs))
        elif int_arrs and isinstance(s, (int, )):
            s = np.asarray(s).astype(max([_a.dtype for _a in int_arrs]))
    return s


class VneCalc(object):
    expr = None
    local_dict = None
    slice_dict = None
    ret_type = None
    ret_context = None

    def __init__(self, expr, local_dict,
                 ret_type=None, ret_context=None):
        self.expr = expr
        self.local_dict = local_dict
        # self.slice_dict = slice_dict
        self.ret_type = ret_type
        self.ret_context = ret_context

    def __call__(self, *args):
        ldict = {}

        for name, val in self.local_dict.items():
            if isinstance(val, int):
                ldict[name] = _normalize_scalar_dtype(args[val], args)
            else:
                ldict[name] = val

        # apply slices if given
        # if self.slice_dict is not None:
        #     for name, slc in self.slice_dict.items():
        #         ldict[name] = ldict[name][slc].data

        # print(ldict)
        result = ne.evaluate(self.expr, local_dict=ldict)
        return args[0].wrap(result, context=self.ret_context,
                            fldtype=self.ret_type)

neg = VneCalc("-a", {'a': 0})
add = VneCalc("a + b", {'a': 0, 'b': 1})
diff = VneCalc("a - b", {'a': 0, 'b': 1})
mul = VneCalc("a * b", {'a': 0, 'b': 1})
relative_diff = VneCalc("(a - b) / a", {'a': 0, 'b': 1})
abs_diff = VneCalc("abs(a - b)", {'a': 0, 'b': 1})
abs_val = VneCalc("abs(a)", {'a': 0})

def scale(a, fld):
    a = np.asarray(a, dtype=fld.dtype)
    b = fld.data  # pylint: disable=unused-variable
    return fld.wrap(ne.evaluate("a * b"))

def axpby(a, x, b, y):
    a = _normalize_scalar_dtype(a, [x, y])
    b = _normalize_scalar_dtype(b, [x, y])
    return x.wrap(ne.evaluate("a * x + b * y"))

def abs_max(fld):
    a = fld.data  # pylint: disable=W0612
    return np.max(ne.evaluate("abs(a)"))

def abs_min(fld):
    a = fld.data  # pylint: disable=W0612
    return np.min(ne.evaluate("abs(a)"))

# def scalar_mul(s, fld):
#     sarr = np.array([s], dtype=fld.dtype)  # pylint: disable=W0612
#     fldarr = fld.data  # pylint: disable=W0612
#     result = ne.evaluate("sarr * fldarr")
#     return fld.wrap(result)

def magnitude(fld):
    vx, vy, vz = fld.component_views()  # pylint: disable=W0612
    mag = ne.evaluate("sqrt((vx**2) + (vy**2) + (vz**2))")
    return fld.wrap(mag, fldtype="Scalar")

def dot(fld_a, fld_b):
    """ 3d dot product of two vector fields """
    ax, ay, az = fld_a.component_views()  # pylint: disable=W0612
    bx, by, bz = fld_b.component_views()  # pylint: disable=W0612
    prod = ne.evaluate("(ax * bx) + (ay * by) + (az * bz)")
    return fld_a.wrap(prod, fldtype="Scalar")

def cross(fld_a, fld_b):
    """ cross product of two vector fields """
    ax, ay, az = fld_a.component_views()  # pylint: disable=W0612
    bx, by, bz = fld_b.component_views()  # pylint: disable=W0612
    prodx = ne.evaluate("ay * bz - az * by")
    prody = ne.evaluate("-ax * bz + az * bx")
    prodz = ne.evaluate("ax * by - ay * bx")
    return fld_a.wrap([prodx, prody, prodz])

def project(fld_a, fld_b):
    """ project a along b (a dot b / magnitude(b)) """
    # ax, ay, az = fld_a.component_views()  # pylint: disable=W0612
    # bx, by, bz = fld_b.component_views()  # pylint: disable=W0612
    # prod = ne.evaluate("(ax * bx) + (ay * by) + (az * bz)")
    # mag = ne.evaluate("sqrt(bx**2 + by**2 + bz**2)")
    # prod = prod / mag
    # return fld_a.wrap(prod, fldtype="Scalar")

    ax, ay, az = fld_a.component_views()  # pylint: disable=W0612
    bx, by, bz = fld_b.component_views()  # pylint: disable=W0612
    projection = ne.evaluate("((ax * bx) + (ay * by) + (az * bz)) / "
                             "sqrt((bx**2) + (by**2) + (bz**2))")
    return fld_a.wrap(projection, fldtype="Scalar")

def normalize(fld_a):
    """ normalize a vector field """
    a = fld_a.as_flat().data
    ax, ay, az = fld_a.component_views()  # pylint: disable=W0612
    normed = ne.evaluate("a / sqrt((ax**2) + (ay**2) + (az**2))")
    return fld_a.wrap(normed, fldtype="Vector")

def grad(fld, bnd=True):
    """2nd order centeral diff, 1st order @ boundaries if bnd"""
    # vx, vy, vz = fld.component_views()
    if bnd:
        fld = viscid.extend_boundaries(fld, order=0, crd_order=0)

    if fld.iscentered("Cell"):
        crdx, crdy, crdz = fld.get_crds_cc(shaped=True)
        # divcenter = "Cell"
        # divcrds = coordinate.NonuniformCartesianCrds(fld.crds.get_clist(np.s_[1:-1]))
        # divcrds = fld.crds.slice_keep(np.s_[1:-1, 1:-1, 1:-1])
    elif fld.iscentered("Node"):
        crdx, crdy, crdz = fld.get_crds_nc(shaped=True)
        # divcenter = "Node"
        # divcrds = coordinate.NonuniformCartesianCrds(fld.crds.get_clist(np.s_[1:-1]))
        # divcrds = fld.crds.slice_keep(np.s_[1:-1, 1:-1, 1:-1])
    else:
        raise NotImplementedError("Can only do cell and node centered gradients")

    v = fld.data
    g = viscid.zeros(fld['x=1:-1, y=1:-1, z=1:-1'].crds, nr_comps=3)

    xp, xm = crdx[2:,  :,  :], crdx[:-2, :  , :  ]  # pylint: disable=bad-whitespace
    yp, ym = crdy[ :, 2:,  :], crdy[:  , :-2, :  ]  # pylint: disable=bad-whitespace
    zp, zm = crdz[ :,  :, 2:], crdz[:  , :  , :-2]  # pylint: disable=bad-whitespace

    vxp, vxm = v[2:  , 1:-1, 1:-1], v[ :-2, 1:-1, 1:-1]  # pylint: disable=bad-whitespace
    vyp, vym = v[1:-1, 2:  , 1:-1], v[1:-1,  :-2, 1:-1]  # pylint: disable=bad-whitespace
    vzp, vzm = v[1:-1, 1:-1, 2:  ], v[1:-1, 1:-1,  :-2]  # pylint: disable=bad-whitespace

    g['x'].data[...] = ne.evaluate("(vxp-vxm)/(xp-xm)")
    g['y'].data[...] = ne.evaluate("(vyp-vym)/(yp-ym)")
    g['z'].data[...] = ne.evaluate("(vzp-vzm)/(zp-zm)")
    return g

def div(fld, bnd=True):
    """2nd order centeral diff, 1st order @ boundaries if bnd"""
    if fld.iscentered("Face"):
        # dispatch fc div immediately since that does its own pre-processing
        return viscid.div_fc(fld, bnd=bnd)

    if bnd:
        fld = viscid.extend_boundaries(fld, order=0, crd_order=0)

    vx, vy, vz = fld.component_views()

    if fld.iscentered("Cell"):
        crdx, crdy, crdz = fld.get_crds_cc(shaped=True)
        divcenter = "Cell"
        # divcrds = coordinate.NonuniformCartesianCrds(fld.crds.get_clist(np.s_[1:-1]))
        divcrds = fld.crds.slice_keep(np.s_[1:-1, 1:-1, 1:-1])
    elif fld.iscentered("Node"):
        crdx, crdy, crdz = fld.get_crds_nc(shaped=True)
        divcenter = "Node"
        # divcrds = coordinate.NonuniformCartesianCrds(fld.crds.get_clist(np.s_[1:-1]))
        divcrds = fld.crds.slice_keep(np.s_[1:-1, 1:-1, 1:-1])
    else:
        raise NotImplementedError("Can only do cell and node centered divs")

    xp, xm = crdx[2:,  :,  :], crdx[:-2, :  , :  ]  # pylint: disable=bad-whitespace
    yp, ym = crdy[ :, 2:,  :], crdy[:  , :-2, :  ]  # pylint: disable=bad-whitespace
    zp, zm = crdz[ :,  :, 2:], crdz[:  , :  , :-2]  # pylint: disable=bad-whitespace

    vxp, vxm = vx[2:  , 1:-1, 1:-1], vx[ :-2, 1:-1, 1:-1]  # pylint: disable=bad-whitespace
    vyp, vym = vy[1:-1, 2:  , 1:-1], vy[1:-1,  :-2, 1:-1]  # pylint: disable=bad-whitespace
    vzp, vzm = vz[1:-1, 1:-1, 2:  ], vz[1:-1, 1:-1,  :-2]  # pylint: disable=bad-whitespace

    div_arr = ne.evaluate("(vxp-vxm)/(xp-xm) + (vyp-vym)/(yp-ym) + "
                          "(vzp-vzm)/(zp-zm)")
    return field.wrap_field(div_arr, divcrds, name="div " + fld.name,
                            center=divcenter, time=fld.time, parents=[fld])

def curl(fld, bnd=True):
    """2nd order centeral diff, 1st order @ boundaries if bnd"""
    if bnd:
        fld = viscid.extend_boundaries(fld, order=0, crd_order=0)

    vx, vy, vz = fld.component_views()

    if fld.iscentered("Cell"):
        crdx, crdy, crdz = fld.get_crds_cc(shaped=True)
        curlcenter = "cell"
        # curlcrds = coordinate.NonuniformCartesianCrds(fld.crds.get_clist(np.s_[1:-1]))
        curlcrds = fld.crds.slice_keep(np.s_[1:-1, 1:-1, 1:-1])
    elif fld.iscentered("Node"):
        crdx, crdy, crdz = fld.get_crds_nc(shaped=True)
        curlcenter = "node"
        # curlcrds = coordinate.NonuniformCartesianCrds(fld.crds.get_clist(np.s_[1:-1]))
        curlcrds = fld.crds.slice_keep(np.s_[1:-1, 1:-1, 1:-1])
    else:
        raise NotImplementedError("Can only do cell and node centered divs")

    xp, xm = crdx[2:,  :,  :], crdx[:-2, :  , :  ]  # pylint: disable=C0326
    yp, ym = crdy[ :, 2:,  :], crdy[:  , :-2, :  ]  # pylint: disable=C0326
    zp, zm = crdz[ :,  :, 2:], crdz[:  , :  , :-2]  # pylint: disable=C0326

    vxpy, vxmy = vx[1:-1, 2:  , 1:-1], vx[1:-1,  :-2, 1:-1]  # pylint: disable=C0326
    vxpz, vxmz = vx[1:-1, 1:-1, 2:  ], vx[1:-1, 1:-1,  :-2]  # pylint: disable=C0326

    vypx, vymx = vy[2:  , 1:-1, 1:-1], vy[ :-2, 1:-1, 1:-1]  # pylint: disable=C0326
    vypz, vymz = vy[1:-1, 1:-1, 2:  ], vy[1:-1, 1:-1,  :-2]  # pylint: disable=C0326

    vzpx, vzmx = vz[2:  , 1:-1, 1:-1], vz[ :-2, 1:-1, 1:-1]  # pylint: disable=C0326
    vzpy, vzmy = vz[1:-1, 2:  , 1:-1], vz[1:-1,  :-2, 1:-1]  # pylint: disable=C0326

    curl_x = ne.evaluate("(vzpy-vzmy)/(yp-ym) - (vypz-vymz)/(zp-zm)")
    curl_y = ne.evaluate("-(vzpx-vzmx)/(xp-xm) + (vxpz-vxmz)/(zp-zm)")
    curl_z = ne.evaluate("(vypx-vymx)/(xp-xm) - (vxpy-vxmy)/(yp-ym)")

    return field.wrap_field([curl_x, curl_y, curl_z], curlcrds,
                            name="curl " + fld.name, fldtype="Vector",
                            center=curlcenter, time=fld.time,
                            parents=[fld])

##
## EOF
##
