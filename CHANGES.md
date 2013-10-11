# Changelog

## 0.50.0 Release
Features:
  - Remove lxml dependancy
  - Remove Cython dependancy
  - Works with the older Python 2.6

## 0.50.1 dev
Features:
  - rewrite fields to a cleaner, unified interface
  - stringy typed things should be case insensitive if compared with the provided methods (istype() and iscentered()), otherwise everything is lowercase
  - field properties are now even lazier and cachier
Refactors:
  - filed.n_points -> filed.nr_points
  - filed.n_comps -> filed.nr_comps
  - grid / dataset n_times -> nr_times
Deprecated:
  - readers.load() -> readers.load_file("...") or readers.load_files(["..."])
