cimport numpy as cnp

ctypedef fused real_t:
    cnp.float32_t
    cnp.float64_t

cdef float MAX_FLOAT
cdef double MAX_DOUBLE

cdef class CyField:
    # cdef bool uniform_crds
    cdef vfield
    cdef str fld_dtype
    cdef str crd_dtype
    cdef str center
    cdef int uniform_crds
    cdef int is_cc
    cdef int n[3]
    cdef int nm1[3]  # fld.n - 1
    cdef int nm2[3]  # fld.n - 2
    cdef int nr_nodes[3]
    cdef int nr_cells[3]
    # cdef int cached_ind[3]  # dangerous for threads :(
    cdef cnp.float64_t min_dx

cdef class Field_I4_Crd_F8(CyField):
    cdef cnp.int32_t[:,:,:,::1] data
    # cdef cnp.float64_t[:] x, y, z
    cdef cnp.float64_t[:] xnc, ync, znc
    cdef cnp.float64_t[:] xcc, ycc, zcc
    cdef cnp.float64_t[:, :, ::1] crds  # [x, y, z]  # first dim is for comp (ECFC)
    cdef cnp.float64_t[:, ::1] crds_nc  # [xnc, ync, znc]
    cdef cnp.float64_t[:, ::1] crds_cc  # [xcc, ycc, zcc]
    cdef cnp.float64_t[3] xlnc, xlcc
    cdef cnp.float64_t[3] xhnc, xhcc
    cdef cnp.float64_t[:, ::1] xl, xh, L  # first dim is for comp (ECFC)
    cdef cnp.float64_t[3] dx  # cell width, = NaN if uniform_crds == False

cdef class Field_I8_Crd_F8(CyField):
    cdef cnp.int64_t[:,:,:,::1] data
    # cdef cnp.float64_t[:] x, y, z
    cdef cnp.float64_t[:] xnc, ync, znc
    cdef cnp.float64_t[:] xcc, ycc, zcc
    cdef cnp.float64_t[:, :, ::1] crds  # [x, y, z]  # first dim is for comp (ECFC)
    cdef cnp.float64_t[:, ::1] crds_nc  # [xnc, ync, znc]
    cdef cnp.float64_t[:, ::1] crds_cc  # [xcc, ycc, zcc]
    cdef cnp.float64_t[3] xlnc, xlcc
    cdef cnp.float64_t[3] xhnc, xhcc
    cdef cnp.float64_t[:, ::1] xl, xh, L  # first dim is for comp (ECFC)
    cdef cnp.float64_t[3] dx  # cell width, = NaN if uniform_crds == False

cdef class Field_F4_Crd_F4(CyField):
    cdef cnp.float32_t[:,:,:,::1] data
    # cdef cnp.float32_t[:] x, y, z
    cdef cnp.float32_t[:] xnc, ync, znc
    cdef cnp.float32_t[:] xcc, ycc, zcc
    cdef cnp.float32_t[:, :, ::1] crds  # [x, y, z]  # first dim is for comp (ECFC)
    cdef cnp.float32_t[:, ::1] crds_nc  # [xnc, ync, znc]
    cdef cnp.float32_t[:, ::1] crds_cc  # [xcc, ycc, zcc]
    cdef cnp.float32_t[3] xlnc, xlcc
    cdef cnp.float32_t[3] xhnc, xhcc
    cdef cnp.float32_t[:, ::1] xl, xh, L  # first dim is for comp (ECFC)
    cdef cnp.float32_t[3] dx  # cell width, = NaN if uniform_crds == False

cdef class Field_F8_Crd_F8(CyField):
    cdef cnp.float64_t[:,:,:,::1] data
    # cdef cnp.float64_t[:] x, y, z
    cdef cnp.float64_t[:] xnc, ync, znc
    cdef cnp.float64_t[:] xcc, ycc, zcc
    cdef cnp.float64_t[:, :, ::1] crds  # [x, y, z]  # first dim is for comp (ECFC)
    cdef cnp.float64_t[:, ::1] crds_nc  # [xnc, ync, znc]
    cdef cnp.float64_t[:, ::1] crds_cc  # [xcc, ycc, zcc]
    cdef cnp.float64_t[3] xlnc, xlcc
    cdef cnp.float64_t[3] xhnc, xhcc
    cdef cnp.float64_t[:, ::1] xl, xh, L  # first dim is for comp (ECFC)
    cdef cnp.float64_t[3] dx  # cell width, = NaN if uniform_crds == False

ctypedef fused FusedField:
    Field_I4_Crd_F8
    Field_I8_Crd_F8
    Field_F4_Crd_F4
    Field_F8_Crd_F8

cdef make_cyfield(vfield)
