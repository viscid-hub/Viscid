import cython
import numpy as np

if cython.compiled:
    print("Using cython compiled code :)")
else:
    print("Not using cython, may be slow")

int_arr3 = cython.int[:, :, :]
real_t_arr2 = cython.float[:, :]
real_t_arr3 = cython.float[:, :, :]

@cython.locals(a=cython.float[:, :, :], tmp=cython.float[:, :],
               i=cython.int, d=cython.int, N=cython.int,
               nr_dims=cython.int)
def reversearr(a, dims):
    nr_dims = len(a.shape)
    if isinstance(dims, int):
        dims = [dims]
    for d in dims:
        N = a.shape[d]
        fwdslcs = [slice(None)] * nr_dims
        bwdslcs = [slice(None)] * nr_dims
        for i in xrange(N // 2):
            fwdslcs[d] = i
            bwdslcs[d] = -1 - i
            tmp = np.array(a[fwdslcs])
            a[fwdslcs] = a[bwdslcs]
            a[bwdslcs] = tmp
    return a
