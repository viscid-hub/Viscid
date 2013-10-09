
def sillytest(fld, seeds, switch=0):
    dtype=fld.data.dtype
    x1 = np.zeros((1,), dtype=dtype)
    val = sillyfunc(x1, dtype, seeds, switch)
    return val

# @cython.boundscheck(False)
# @cython.wraparound(False)
def sillyfunc(real_t[:] num, dtype, seeds, switch):
    cdef real_t[:] arr_c = np.empty((3,), dtype=dtype)
    cdef real_t[:,:] x0
    cdef real_t val = 0
    cdef unsigned int it = 0
    if switch == 0:
        if isinstance(seeds, seed.SeedGen):
            # recast the seed data type... this should be done better...
            x0 = seeds.points.astype(dtype)
        else:
            seeds_lst = list(seeds.iter_points())
            x0 = np.array(seeds_lst, dtype=dtype).reshape((-1, 3))
        for i from 0 <= i < x0.shape[0]:
            arr_c = x0[i]
            val += csillyfunc(x0[i])
            it += 1
    elif switch==1:
        for pt in seeds.iter_points():
            arr_c[0] = pt[0]
            arr_c[1] = pt[1]
            arr_c[2] = pt[2]
            val += csillyfunc(arr_c)
            it += 1
    elif switch==2:
        pts = seeds.iter_points()
        for ind from 0 <= ind < seeds.n_points():
            pt = pts.__next__()
            #for pt in seeds.iter_points():
            arr_c[0] = pt[0]
            arr_c[1] = pt[1]
            arr_c[2] = pt[2]
            val += csillyfunc(arr_c)
            it += 1
    print("total iterations: ", it)
    return val

cdef csillyfunc(real_t[:] arr):
    return arr[0] + arr[1] + arr[2]
