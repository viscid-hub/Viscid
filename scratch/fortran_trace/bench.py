#!/usr/bin/env python

from __future__ import print_function
from timeit import default_timer as time

import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs": [np.get_include()]})

import for_bench
import cy_bench

N = 1000
nfcalls = int(1e7)

def main():
    arr_in = np.array(np.linspace(0, 100, N), dtype="float32")
    arr_out = np.empty_like(arr_in)
    total = np.zeros((1,), dtype='float32')

    t0 = time()
    for_bench.fortran_entry(arr_in, arr_out, nfcalls, total)
    t1 = time()
    t_for = t1 - t0
    print("fortran: {0:.03e} s  {1:.03e} s/call".format(t_for, t_for / nfcalls))

    t0 = time()
    cy_bench.cython_entry(arr_in, arr_out, nfcalls)
    t1 = time()
    t_cy = t1 - t0
    print("cython: {0:.03e} s  {1:.03e} s/call".format(t_cy, t_cy / nfcalls))

    print("Fortran speedup {0}x".format(t_cy / t_for))

if __name__ == "__main__":
    main()
