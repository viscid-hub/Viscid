#!/usr/bin/env python

from __future__ import print_function
from timeit import default_timer as time

import numpy as np
import h5py as h5

from viscid import readers

def timeit(f, *args, **kwargs):
    t0 = time()
    ret = f(*args, **kwargs)
    t1 = time()

    print("Took {0:.03g} secs.".format(t1 - t0))
    return ret

# def reversearr(a, dims):
#     if isinstance(dims, int):
#         dims = [dims]
#     ni, nj, nk = a.shape
#     nij = ni * nj
#     njk = nj * nk
#     nijk = ni * nj * nk

#     aflat = a.reshape(-1)

#     si = 0 in dims
#     sj = 1 in dims
#     sk = 2 in dims

#     # for ind in xrange(nijk):
#     #     i = ind // (njk)
#     #     j = (ind // nk) % nj
#     #     k = ind % nk
#     #     otheri = ni - 1 - i if si else i
#     #     otherj = nj - 1 - j if sj else j
#     #     otherk = nk - 1 - k if sk else k
#     #     otherind = otheri * njk + otherj * nk + otherk
#     #     tmp = aflat[ind]
#     #     aflat[ind] = aflat[otherind]
#     #     aflat[otherind] = tmp
#     for i in xrange(ni):
#         for j in xrange(nj):
#             for k in xrange(nk):
#                 otheri = ni - 1 - i if si else i
#                 otherj = nj - 1 - j if sj else j
#                 otherk = nk - 1 - k if sk else k
#                 tmp = a[i, j, k]
#                 a[i, j, k] = a[otheri, otherj, otherk]
#                 a[otheri, otherj, otherk] = tmp
#     return a

def reversearr(a, dims):
    nr_dims = len(a.shape)
    if isinstance(dims, int):
        dims = [dims]
    for d in dims:
        N = a.shape[d]
        fwdslcs = [slice(None)] * nr_dims
        bwdslcs = [slice(None)] * nr_dims
        tmpshape = list(a.shape)
        tmpshape.pop(d)
        tmp = np.empty(tmpshape, dtype=a.dtype)
        for i in xrange(N // 2):
            fwdslcs[d] = i
            bwdslcs[d] = -1 - i
            tmp[...] = a[fwdslcs]
            a[fwdslcs] = a[bwdslcs]
            a[bwdslcs] = tmp
    return a

f = h5.File("/Users/kmaynard/dev/work/tmp/cen2000.3d.004045_p000000.h5")
rr = f["/mrc_fld_rr-uid-0x5a40c8b0/rr/3d"]
pp = f["/mrc_fld_pp-uid-0x5a416d00/pp/3d"]
rr1_arr = timeit(np.array, rr)
rr2_arr = timeit(lambda x: np.array(np.array(x)[:,::-1,::-1]), rr)

print(rr1_arr.flags)
print("==")
print(rr2_arr.flags)

# # sizes = [(8, 8, 8), (16, 16, 16), (32, 32, 32), (64, 64, 64), (128, 128, 128)]
# sizes = [(256, 256, 512), (256, 256, 800), (512, 512, 1024)]
# for size in sizes:
#     print("problem size:", size)
#     arr = timeit(np.random.rand, *size)
#     # rev = timeit(lambda x: np.array(x[:,::-1,::-1]), arr)
#     arr = timeit(reversearr, arr, [1, 2])
#     print("same?", (arr == rev).all())
#     # print(rev.flags)
