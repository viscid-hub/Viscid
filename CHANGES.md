# Changelog

## 0.50.1
Features:
  - Field slicing semantics are now the same as numpy in terms of when dimensions get reduced. To enforce specific reductions, use Field.slice_reduce() or Field.slice_keep()
  - crds accessable from grids / fields / crds using get_crd_[ncef]c or get_crds_[ncef]c for one or multile crds rspectively
  - rewrite fields to a cleaner, unified interface
  - stringy typed things should be case insensitive if compared with the provided methods (istype() and iscentered()), otherwise everything is lowercase
  - field properties are now even lazier and cachier
  - add reader for 1d gnuplot like files
  - add reader for numpy binary npz files (can also save a list of fields)
  - add super preliminary support for writing hdf5 files with companion xdmf file, still no direct hdf5 reading
  - add pretty_name kwarg to field constructor, this is the name that will appear in plot labels
Refactors:
  - filed.n_points -> filed.nr_points
  - filed.n_comps -> filed.nr_comps
  - grid / dataset n_times -> nr_times
  - trilin_interp -> interp_trilin
  - keyword arg cc_slice -> cc
Backward Incompatible Changes:
  - Coordinate.get_crd used to take a list, now that use case has to call Coordinate.get_crds
Deprecated:
  - readers.load() -> readers.load_file("...") or readers.load_files(["..."])

## 0.50.0 Release
Features:
  - Remove lxml dependancy
  - Remove Cython dependancy
  - Works with the older Python 2.6
