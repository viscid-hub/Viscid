#from __future__ import print_function
cimport numpy as cnp

# NOTE: there may be a very small performance hit for using fused
# types like this
ctypedef fused real_t:
    cnp.float32_t
    cnp.float64_t

ctypedef fused fld_t:
    cnp.float32_t
    cnp.float64_t
    cnp.int32_t
    cnp.int64_t

# for debugging (faster compile) / making the C code more readable
# ctypedef np.float32_t real_t
