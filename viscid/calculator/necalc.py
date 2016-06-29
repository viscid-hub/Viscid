#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import numexpr as ne

from viscid import field
from viscid import coordinate

# ne.set_num_threads(1)  # for performance checking

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
                ldict[name] = args[val]
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
    a = fld_a.data
    ax, ay, az = fld_a.component_views()  # pylint: disable=W0612
    normed = ne.evaluate("a / sqrt((ax**2) + (ay**2) + (az**2))")
    return fld_a.wrap(normed, fldtype="Vector")

def grad(fld):
    """ first order """
    pass

def div(fld):
    """ first order """
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

def curl(fld):
    """ first order """
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
