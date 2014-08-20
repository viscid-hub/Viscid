# Changelog

## 0.60.2 dev

Features:
  - The attribute `grid.Grid.longterm_field_caches` controls how long caches hang around

## 0.60.1

Bugfix:
  - Time slices were acting silly

## 0.60.0

Features:
  - Implement Ionosphere reading / plotting
  - Implement a Jrrle Openggcm Reader

Backward Incompatible Changes:
  - CHANGE FIELD SLICING SYNTAX! No more trailing 'i' to slice by index. Instead, use an integer to mean an index. For slicing by coordinate value, the decimal now needs to be explicit, so all code that slices like "y=0" now needs to be refactored to "y=0.0"

## 0.50.2

Features:
  - Support for custom grids (this allows for custom readers for GGCM / PSC / etc)
  - Grids can supply derived fields by defining _get_varname
  - Grids can supply generic transformations for fields / crds on load
  - GGCM reader, can translate MHD coordinate system to GSE using `viscid.readers.openggcm.GGCMGrid.mhd_to_gse_on_read = True`
  - PSC reader can calculate psi (flux function)

Refactors:
  - RectilinearCrds -> NonuniformCartesianCrds
  - "Rectilinear" -> "nonuniform_cartesian"

Backward Incompatible Changes:
  - kwargs to field constructors go to deep_meta dict if they start with a '_', else they go to the info dict

Other:
  - Precedence for auto-detecting classes is given to those more recently declared

## 0.50.1

Features:
  - Field slicing semantics are now the same as numpy in terms of when dimensions get reduced. To enforce specific reductions, use Field.slice_reduce() or Field.slice_keep()
  - crds accessible from grids / fields / crds using get_crd_[ncef]c or get_crds_[ncef]c for one or multiple crds respectively
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
  - Remove lxml dependency
  - Remove Cython dependency
  - Works with the older Python 2.6
