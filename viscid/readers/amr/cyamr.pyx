# cython: boundscheck=False, wraparound=False, cdivision=True, profile=False
#
# Note: a _c_FUNCTION can only be called from another cdef-ed function, or
# a def-ed _py_FUNCTION function because of the use of the fused real_t
# to template both float32 and float64 versions

from __future__ import print_function

import numpy as np

# from cython.operator cimport dereference as deref
# from cython.view cimport array as cvarray
# from cython.parallel import prange

# from viscid import logger
# from viscid import field
# from viscid import coordinate
# from viscid.calculator import seed

###########
# cimports
cimport cython
cimport numpy as cnp

# from libc.math cimport sqrt

# from cycalc_util cimport *
# from cycalc cimport *


import numpy as np


# |-----|-----|-----|
# |     |     |     |
# |-----|-----|-----|
# |     | p0  |     |
# |-----|-----|-----|
# |     |     |     |
# |-----|-----|-----|

# there are 26 surface cubes on a rubix cube, 6 of which could be
# refined such that they become 4 neighbors, and 8 of which could be
# split to be 2 neighbors, so every cell has a max of 28 + 18 + 8 = 54
# neighbors

# def discover_neighbors(self):
#     d = dict(xm=[], xp=[], ym=[], yp=[], zm=[], zp=[])
#     self.neighbors_index = [d.copy() for _ in self.patches]
#     self.all_neighbors_index = [[] for _ in self.patches]

#     # call find_relationships (N * (N - 1)) / 2 times
#     for i, patch in enumerate(self.patches):
#         for j, other in enumerate(self.patches[:i]):
#             rels = patch.find_relationships(other)
#             # print("RELS:", i, j, rels)
#             for rel in rels:
#                 self.neighbors_index[i][rel[0]].append(j)
#                 self.neighbors_index[j][rel[1]].append(i)

#                 if j not in self.all_neighbors_index[i]:
#                     self.all_neighbors_index[i].append(j)
#                 if i not in self.all_neighbors_index[j]:
#                     self.all_neighbors_index[j].append(i)
#     self.tell_patches_about_neighbors()
