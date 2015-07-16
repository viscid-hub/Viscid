cimport numpy as cnp

from viscid.cython.cyfield cimport real_t, CyField
from viscid.cython.cyfield cimport Field_I4_Crd_F8, Field_I8_Crd_F8
from viscid.cython.cyfield cimport Field_F4_Crd_F4, Field_F8_Crd_F8

cdef class CyAMRField:
    cdef str crd_dtype
    cdef int nr_patches
    cdef int[:] nr_neighbors
    cdef int[:, ::1] neighbors
    cdef int[:, ::1] neighbor_mask
    cdef int active_patch_index
    cdef list patches  # maybe this can become a typed array?

cdef class AMRField_I4_Crd_F8(CyAMRField):
    cdef cnp.float64_t[:, ::1] xl, xm, xh
    cdef cnp.float64_t global_xl[3]
    cdef cnp.float64_t global_xh[3]
    cdef cnp.float64_t min_dx
    cdef Field_I4_Crd_F8 active_patch

cdef class AMRField_I8_Crd_F8(CyAMRField):
    cdef cnp.float64_t[:, ::1] xl, xm, xh
    cdef cnp.float64_t global_xl[3]
    cdef cnp.float64_t global_xh[3]
    cdef cnp.float64_t min_dx
    cdef Field_I8_Crd_F8 active_patch

cdef class AMRField_F4_Crd_F4(CyAMRField):
    cdef cnp.float32_t[:, ::1] xl, xm, xh
    cdef cnp.float32_t global_xl[3]
    cdef cnp.float32_t global_xh[3]
    cdef cnp.float32_t min_dx
    cdef Field_F4_Crd_F4 active_patch

cdef class AMRField_F8_Crd_F8(CyAMRField):
    cdef cnp.float64_t[:, ::1] xl, xm, xh
    cdef cnp.float64_t global_xl[3]
    cdef cnp.float64_t global_xh[3]
    cdef cnp.float64_t min_dx
    cdef Field_F8_Crd_F8 active_patch

ctypedef fused FusedAMRField:
    AMRField_I4_Crd_F8
    AMRField_I8_Crd_F8
    AMRField_F4_Crd_F4
    AMRField_F8_Crd_F8


cdef CyAMRField make_cyamrfield(vfield)
cdef CyField activate_patch(FusedAMRField amrfld, real_t x[3])

##
## EOF
##
