# cython: boundscheck=False, wraparound=False, cdivision=True, profile=False
# cython: emit_code_comments=False

from viscid.cython.cyfield cimport *
from viscid.cython.misc_inlines cimport int_max

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

cdef _init_cyfield(FusedField fld, vfield, fld_dtype, crd_dtype):
    dat = vfield.data
    while len(dat.shape) < 4:
        dat = np.expand_dims(dat, axis=4)
    fld.data = dat

    fld.center = vfield.center

    # x, y, z = vfield.get_crds()
    x, y, z = vfield.get_crds_vector()
    xnc, ync, znc = vfield.get_crds_nc()
    xcc, ycc, zcc = vfield.get_crds_cc()

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

    _crd_lst = [[_x, _y, _z] for _x, _y, _z in zip(x, y, z)]
    _crd_lst_nc = [xnc, ync, znc]
    _crd_lst_cc = [xcc, ycc, zcc]
    sshape_max = max(sshape)
    fld.crds = np.nan * np.ones((3, 3, sshape_max), dtype=crd_dtype)
    sshape_nc_max = max(sshape_nc)
    fld.crds_nc = np.nan * np.ones((3, sshape_nc_max), dtype=crd_dtype)
    sshape_cc_max = max(sshape_cc)
    fld.crds_cc = np.nan * np.ones((3, sshape_cc_max), dtype=crd_dtype)

    fld.xl = np.nan * np.ones((3, 3), dtype=crd_dtype)
    fld.xh = np.nan * np.ones((3, 3), dtype=crd_dtype)
    fld.L = np.nan * np.ones((3, 3), dtype=crd_dtype)

    for ic in range(3):
        for i in range(3):
            for j in range(sshape[i]):
                fld.crds[ic, i, j] = _crd_lst[ic][i][j]
            # fld.xl[ic, i] = fld.crds[ic, i, 0]
            # fld.xh[ic, i] = fld.crds[ic, i, sshape[i] - 1]

    fld.min_dx = np.min(vfield.crds.min_dx_nc)

    for i in range(3):
        if sshape[i] == fld.nr_nodes[i]:
            fld.is_cc = 0
        else:
            fld.is_cc = 1

        fld.xlnc[i] = vfield.crds.xl_nc[i]
        fld.xhnc[i] = vfield.crds.xh_nc[i]
        fld.xlcc[i] = vfield.crds.xl_cc[i]
        fld.xhcc[i] = vfield.crds.xh_cc[i]

        fld.nr_nodes[i] = sshape_nc[i]  # len(fld.crds_nc[i])
        fld.nr_cells[i] = sshape_cc[i]  # len(fld.crds_cc[i])
        fld.n[i] = sshape[i]  # len(fld.crds[i])
        fld.nm1[i] = fld.n[i] - 1
        fld.nm2[i] = fld.n[i] - 2

        for ic in range(3):
            fld.xl[ic, i] = fld.crds[ic, i, 0]
            fld.xh[ic, i] = fld.crds[ic, i, int_max(sshape[i] - 1, 0)]
            fld.L[ic, i] = fld.xh[ic, i] - fld.xl[ic, i]

        # fld.cached_ind[i] = 0  # dangerous for threads :(
        fld.uniform_crds = vfield.crds._TYPE.startswith("uniform")
        if fld.uniform_crds:
            fld.dx[i] = (fld.xhnc[i] - fld.xlnc[i]) / fld.nr_nodes[i]
        else:
            fld.dx[i] = np.nan

        for ic in range(3):
            if fld.xh[ic, i] < fld.xl[ic, i]:
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

cdef make_cyfield(vfield):
    vfield = vfield.as_interlaced(force_c_contiguous=True).atleast_3d()
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

##
## EOF
##
