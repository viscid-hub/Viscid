#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import numexpr as ne

from .. import field
from .. import coordinate

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
        return args[0].wrap(result, context=self.ret_context, typ=self.ret_type)

add = VneCalc("a + b", {'a': 0, 'b': 1})
diff = VneCalc("a - b", {'a': 0, 'b': 1})
mul = VneCalc("a * b", {'a': 0, 'b': 1})
relative_diff = VneCalc("(a - b) / a", {'a': 0, 'b': 1})
abs_diff = VneCalc("abs(a - b)", {'a': 0, 'b': 1})
abs_val = VneCalc("abs(a)", {'a': 0})

def abs_max(fld):
    a = fld.data #pylint: disable=W0612
    return np.max(ne.evaluate("abs(a)"))

def abs_min(fld):
    a = fld.data #pylint: disable=W0612
    return np.min(ne.evaluate("abs(a)"))

# def scalar_mul(s, fld):
#     sarr = np.array([s], dtype=fld.dtype) #pylint: disable=W0612
#     fldarr = fld.data #pylint: disable=W0612
#     result = ne.evaluate("sarr * fldarr")
#     return fld.wrap(result)

def magnitude(fld):
    vx, vy, vz = fld.component_views() #pylint: disable=W0612
    mag = ne.evaluate("sqrt((vx**2) + (vy**2) + (vz**2))")
    return fld.wrap(mag, typ="Scalar")

def dot(fld_a, fld_b):
    """ 3d dot product of two vector fields """
    ax, ay, az = fld_a.component_views() #pylint: disable=W0612
    bx, by, bz = fld_b.component_views() #pylint: disable=W0612
    prod = ne.evaluate("(ax * bx) + (ay * by) + (az * bz)")
    return fld_a.wrap(prod, typ="Scalar")

def cross(fld_a, fld_b):
    """ cross product of two vector fields """
    ax, ay, az = fld_a.component_views() #pylint: disable=W0612
    bx, by, bz = fld_b.component_views() #pylint: disable=W0612
    prodx = ne.evaluate("ay * bz - az * by")
    prody = ne.evaluate("-ax * bz + az * bx")
    prodz = ne.evaluate("ax * by - ay * bx")
    return fld_a.wrap([prodx, prody, prodz])

def div(fld):
    """ first order """
    vx, vy, vz = fld.component_views()

    if fld.center == "Cell":
        crdz, crdy, crdx = fld.crds.get_crd(shaped=True, center="Cell")
        divcenter = "Cell"
        divcrds = coordinate.RectilinearCrds(fld.crds.get_clist(np.s_[1:-1]))
    elif fld.center == "Node":
        crdz, crdy, crdx = fld.crds.get_crd(shaped=True)
        divcenter = "Node"
        divcrds = coordinate.RectilinearCrds(fld.crds.get_clist(np.s_[1:-1]))
    else:
        raise NotImplementedError("Can only do cell and node centered divs")

    xp = crdx[:,:,2:]; xm = crdx[:,:,:-2] #pylint: disable=W0612,C0321,C0324
    yp = crdy[:,2:,:]; ym = crdy[:,:-2,:] #pylint: disable=W0612,C0321,C0324
    zp = crdz[2:,:,:]; zm = crdz[:-2,:,:] #pylint: disable=W0612,C0321,C0324

    vxp = vx[1:-1,1:-1,2:]; vxm = vx[1:-1,1:-1,:-2] #pylint: disable=W0612,C0321,C0324,C0301
    vyp = vy[1:-1,2:,1:-1]; vym = vy[1:-1,:-2,1:-1] #pylint: disable=W0612,C0321,C0324,C0301
    vzp = vz[2:,1:-1,1:-1]; vzm = vz[:-2,1:-1,1:-1] #pylint: disable=W0612,C0321,C0324,C0301

    # print(vxp.shape, vyp.shape, vzp.shape)
    # print(vxm.shape, vym.shape, vzm.shape)
    # print(xp.shape, yp.shape, zp.shape)
    # print(xm.shape, ym.shape, zm.shape)

    div_arr = ne.evaluate("(vxp-vxm)/(xp-xm) + (vyp-vym)/(yp-ym) + "
                          "(vzp-vzm)/(zp-zm)")

    return field.wrap_field("Scalar", "div " + fld.name, divcrds, div_arr,
                            center=divcenter, time=fld.time)

# This is not correct
# div = VneCalc("(vxp-vxm)/(xp-xm) + (vyp-vym)/(yp-ym) + (vzp-vzm)/(zp-zm)",
#               {"vxp": 0, "vxm": 0, "vyp": 1, "vym": 1, "vzp": 2, "vzm": 2,
#                "xp": 0, "xm": 0, "yp": 1, "ym": 1, "zp": 2, "zm": 2,
#               },
#               {"vxp": np.s_[1:-1,1:-1,2:], "vxm": np.s_[1:-1,1:-1,:-2],
#                "vyp": np.s_[1:-1,2:,1:-1], "vym": np.s_[1:-1,:-2,1:-1],
#                "vzp": np.s_[2:,1:-1,1:-1], "vzm": np.s_[:-2,1:-1,1:-1],
#                "xp": np.s_[:,:,2:], "xm": np.s_[:,:,:-2],
#                "yp": np.s_[:,2:,:], "ym": np.s_[:,:-2,:],
#                "zp": np.s_[2:,:,:], "zm": np.s_[:-2,:,:],
#               },
#               ret_type=field.ScalarField,
#              )

##
## EOF
##
