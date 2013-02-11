#from __future__ import print_function
cimport numpy as np

# NOTE: there may be a very small performance hit for using fused
# types like this
ctypedef fused real_t:
    np.float32_t
    np.float64_t
