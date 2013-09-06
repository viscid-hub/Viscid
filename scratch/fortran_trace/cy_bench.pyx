# cython: boundscheck=False, wraparound=False, cdivision=True, profile=False

cimport numpy as np

ctypedef np.float32_t real_t

def cython_entry(real_t[:] arr_in, real_t[:] arr_out, int nfcalls):
    cdef real_t total
    cdef int i

    for i in range(nfcalls):
        total += worker(arr_in, arr_out)
        total += worker2(arr_in, arr_out)
    return total

cdef real_t worker(real_t[:] arr_in, real_t[:] arr_out):
    cdef real_t total
    cdef int i = 1

    # for i in range(arr_in.shape[0]):
    #     arr_out[i] = 2.0 * arr_in[i]
    #     total += arr_in[i]

    total = arr_in[i]

    return total

cdef real_t worker2(real_t[:] arr_in, real_t[:] arr_out):
    cdef real_t total
    cdef int i = 2

    # for i in range(arr_in.shape[0]):
    #     arr_out[i] = 2.0 * arr_in[i]
    #     total += arr_in[i]

    total = arr_in[i]

    return total
