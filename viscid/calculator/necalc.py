#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import numexpr as ne

from .. import field
from .. import coordinate

# ne.set_num_threads(1)  # for performance checking

def difference(fld_a, fld_b, sla=slice(None), slb=slice(None)):
    a = fld_a.data[sla]  #pylint: disable=W0612
    b = fld_b.data[slb]  #pylint: disable=W0612
    diff = ne.evaluate("a - b")
    return diff

def relative_diff(fld_a, fld_b, sla=slice(None), slb=slice(None)):
    a = fld_a.data[sla]  #pylint: disable=W0612
    b = fld_b.data[slb]  #pylint: disable=W0612
    diff = ne.evaluate("(a - b) / a")
    return diff

def abs_diff(fld_a, fld_b, sla=slice(None), slb=slice(None)):
    a = fld_a.data[sla]  #pylint: disable=W0612
    b = fld_b.data[slb]  #pylint: disable=W0612
    diff = ne.evaluate("abs(a - b)")
    return diff

def abs_val(fld):
    a = fld.data  #pylint: disable=W0612
    absarr = ne.evaluate("abs(a)")
    return absarr

def magnitude(fld):
    vx, vy, vz = fld.component_views() #pylint: disable=W0612
    mag = ne.evaluate("sqrt((vx**2) + (vy**2) + (vz**2))")
    return mag

def div(fld):
    """ first order  """
    vx, vy, vz = fld.component_views()

    if fld.center == "Cell":
        crdz, crdy, crdx = fld.crds.get_cc(shaped=True)
        divcenter = "Cell"
        divcrds = coordinate.RectilinearCrds(fld.crds.get_clist(np.s_[1:-1]))
    elif fld.center == "Node":
        crdz, crdy, crdx = fld.crds.get_nc(shaped=True)
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

##
## EOF
##
