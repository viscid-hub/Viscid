# cython: boundscheck=False, wraparound=False, cdivision=True, profile=False
from viscid.cython.cyfield cimport *

import numpy as np

# should be 3.4028235e+38 ?
cdef float MAX_FLOAT = 1e37
# should be 1.7976931348623157e+308 ?
cdef double MAX_DOUBLE = 1e307


cdef inline int _c_int_max(int a, int b):
    if a >= b:
        return a
    else:
        return b

cdef CyField make_cyfield(vfield):
    vfield = vfield.as_interlaced(force_c_contiguous=True)
    fld_dtype = np.dtype(vfield.dtype)

    if fld_dtype == np.dtype('i4'):
        fld = _init_cyfield(Field_I4_Crd_F8(), vfield, 'i4', 'f8')
    elif fld_dtype == np.dtype('i8'):
        fld = _init_cyfield(Field_I8_Crd_F8(), vfield, 'i8', 'f8')
    if fld_dtype == np.dtype('f4'):
        fld = _init_cyfield(Field_F4_Crd_F4(), vfield, 'f4', 'f4')
    elif fld_dtype == np.dtype('f8'):
        fld = _init_cyfield(Field_F8_Crd_F8(), vfield, 'f8', 'f8')
    else:
        raise RuntimeError("Bad field dtype for cython code {0}"
                           "".format(fld_dtype))

    return fld

cdef FusedField _init_cyfield(FusedField fld, vfield, fld_dtype, crd_dtype):
    dat = vfield.data
    if len(dat.shape) < 4:
        dat = np.expand_dims(dat, axis=4)
    fld.data = dat

    fld.center = vfield.center

    x, y, z = vfield.get_crds("xyz")
    xnc, ync, znc = vfield.get_crds_nc("xyz")
    xcc, ycc, zcc = vfield.get_crds_cc("xyz")

    fld.x = x.astype(crd_dtype, copy=False)
    fld.y = y.astype(crd_dtype, copy=False)
    fld.z = z.astype(crd_dtype, copy=False)
    fld.xnc = xnc.astype(crd_dtype, copy=False)
    fld.ync = ync.astype(crd_dtype, copy=False)
    fld.znc = znc.astype(crd_dtype, copy=False)
    fld.xcc = xcc.astype(crd_dtype, copy=False)
    fld.ycc = ycc.astype(crd_dtype, copy=False)
    fld.zcc = zcc.astype(crd_dtype, copy=False)

    xl_nc = vfield.crds.xl_nc
    xh_nc = vfield.crds.xh_nc

    cdef int[3] sshape = vfield.sshape
    cdef int[3] sshape_nc = vfield.crds.shape_nc
    cdef int[3] sshape_cc = vfield.crds.shape_cc

    _crd_lst = [z, y, x]
    _crd_lst_nc = [znc, ync, xnc]
    _crd_lst_cc = [zcc, ycc, xcc]
    sshape_max = max(sshape)
    fld.crds = np.nan * np.empty((3, sshape_max), dtype=crd_dtype)
    sshape_nc_max = max(sshape_nc)
    fld.crds_nc = np.nan * np.empty((3, sshape_nc_max), dtype=crd_dtype)
    sshape_cc_max = max(sshape_cc)
    fld.crds_cc = np.nan * np.empty((3, sshape_cc_max), dtype=crd_dtype)

    fld.min_dx = np.min(vfield.crds.min_dx_nc)

    for i in range(3):
        fld.xlnc[i] = vfield.crds.xl_nc[i]
        fld.xhnc[i] = vfield.crds.xh_nc[i]
        fld.xlcc[i] = vfield.crds.xl_cc[i]
        fld.xhcc[i] = vfield.crds.xh_cc[i]

        fld.nr_nodes[i] = sshape_nc[i]  # len(fld.crds_nc[i])
        fld.nr_cells[i] = sshape_cc[i]  # len(fld.crds_cc[i])

        if sshape[i] == fld.nr_nodes[i]:
            fld.is_cc = 0
            fld.xl[i] = fld.xlnc[i]
            fld.xh[i] = fld.xhnc[i]
            fld.n[i] = fld.nr_nodes[i]  # len(fld.crds[i])
        else:
            fld.is_cc = 1
            fld.xl[i] = fld.xlcc[i]
            fld.xh[i] = fld.xhcc[i]
            fld.n[i] = fld.nr_cells[i]  # len(fld.crds[i])

        fld.L[i] = fld.xh[i] - fld.xl[i]
        fld.nm1[i] = fld.n[i] - 1
        fld.nm2[i] = fld.n[i] - 2

        for j in range(sshape[i]):
            fld.crds[i, j] = _crd_lst[i][j]
        for j in range(sshape_nc[i]):
            fld.crds_nc[i, j] = _crd_lst_nc[i][j]
        for j in range(sshape_cc[i]):
            fld.crds_cc[i, j] = _crd_lst_cc[i][j]

        fld.cached_ind[i] = 0
        fld.uniform_crds = vfield.crds._TYPE.startswith("uniform")
        if fld.uniform_crds:
            fld.dx[i] = (fld.xhnc[i] - fld.xlnc[i]) / fld.nr_nodes[i]
        else:
            fld.dx[i] = np.nan

        if fld.xh[i] < fld.xl[i]:
            raise RuntimeError("Forward crds only in cython code")

    # print("??", sshape_max)
    # for i in range(3):
    #     print("!", i)
    #     for j in range(sshape_max):
    #         print(">>", i, j, fld.crds[i, j])

    fld.fld_dtype = fld_dtype
    fld.crd_dtype = crd_dtype
    fld.vfield = vfield

    return fld

##
## EOF
##
