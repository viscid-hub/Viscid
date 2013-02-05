#!/usr/bin/env python

from __future__ import print_function
# import os
# from math import sqrt

import numpy as np
import numexpr as ne
# import pylab as pl

from .. import field
from .. import coordinate
from . import cycalc
# from weave_helper import wrap_inline

# ne.set_num_threads(1)

def difference(fld_a, fld_b, func="numexpr", sla=slice(None), slb=slice(None)):
    if func == "numexpr":
        a = fld_a.data[sla]  #pylint: disable=W0612
        b = fld_b.data[slb]  #pylint: disable=W0612
        diff = ne.evaluate("a-b")
    else:
        raise ValueError("Desired backend not found.")

    return field.wrap_field(fld_a.TYPE, fld_a.name + " difference", fld_a.crds,
                            diff, center=fld_a.center, time=fld_a.time)

def relative_diff(fld_a, fld_b, func="numexpr",
                  sla=slice(None), slb=slice(None)):
    if func == "numexpr":
        a = fld_a.data[sla]  #pylint: disable=W0612
        b = fld_b.data[slb]  #pylint: disable=W0612
        diff = ne.evaluate("(a - b) / a")
    else:
        raise ValueError("Desired backend not found.")

    return field.wrap_field(fld_a.TYPE, fld_a.name + " difference", fld_a.crds,
                            diff, center=fld_a.center, time=fld_a.time)

# def relative_diff(fld_a, fld_b, func="numexpr"):
#     if func == "numexpr":
#         a = fld_a.data  #pylint: disable=W0612
#         b = fld_b.data  #pylint: disable=W0612
#         diff = ne.evaluate("(a - b) / b")
#     else:
#         raise ValueError("Desired backend not found.")

#     return field.wrap_field(fld_a.TYPE, fld_a.name + " difference", fld_a.crds,
#                             diff, center=fld_a.center, time=fld_a.time)

def abs(fld, func="numexpr"): #pylint: disable=W0622
    if func == "numexpr":
        a = fld.data  #pylint: disable=W0612
        absarr = ne.evaluate("abs(a)")
    else:
        raise ValueError("Desired backend not found.")

    return field.wrap_field(fld.TYPE, "abs " + fld.name, fld.crds,
                            absarr, center=fld.center, time=fld.time)

def magnitude(fld, func="numexpr"):
    if func == "cython":
        return cycalc.magnitude(fld)
    elif func == "numexpr":
        vx, vy, vz = fld.component_views()
        mag = ne.evaluate("sqrt((vx**2) + (vy**2) + (vz**2))")
    elif func == "numpy":
        vx, vy, vz = fld.component_views()
        mag = np.sqrt((vx**2) + (vy**2) + (vz**2))
    elif func == "native":
        vx, vy, vz = fld.component_views()
        mag = np.empty_like(vx)
        for i in range(mag.shape[0]):
            for j in range(mag.shape[1]):
                for k in range(mag.shape[2]):
                    mag[i, j, k] = np.sqrt(vx[i, j, k]**2 + vy[i, j, k]**2 + \
                                           vz[i, j, k]**2)
    else:
        raise ValueError("Desired backend not found.")

    #print("MMM ", np.max(mag - np.sqrt((vx**2) + (vy**2) + (vz**2))))
    #print(mag[:,128,128])
    return field.wrap_field("Scalar", fld.name + " magnitude", fld.crds,
                            mag, center=fld.center, time=fld.time)

def div(fld, func="cython"):
    if func == "cython":
        return cycalc.div(fld)
    elif func == "numexpr":
        return divne(fld)
    else:
        raise ValueError("Desired backend not found.")

def divne(fld):
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

def closest1d_ind(arr, val):
    #i = np.argmin(ne.evaluate("abs(arr - val)"))
    return np.argmin(np.abs(arr - val))

def closest1d_val(arr, val):
    #i = np.argmin(ne.evaluate("abs(arr - val)"))
    i = np.argmin(np.abs(arr - val))
    return arr[i]

def nearest_val(fld, point):
    """ find value of field closest to point """
    x, y, z = point
    xind = closest1d_ind(fld.crds['x'], x)
    yind = closest1d_ind(fld.crds['y'], y)
    zind = closest1d_ind(fld.crds['z'], z)
    return fld[zind, yind, xind]

##
## EOF
##
