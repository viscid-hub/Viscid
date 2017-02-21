Changes
-------

0.98.2 dev
==========

Changes:
  - Interpolaton / Streamline code calls atleast_3d() on field if it thinks that might be necessary
  - interp now accepts 'linear' as well as 'trilin' etc.
  - add convenience glob2 function that wraps slice_globbed_filenames
  - mapfield now accepts 'n' and 's' for hemisphere
  - add a provisional convenience mpl function plot_iono (automatically annotates CPCP etc on ionosphere plots)
  - let user pass crd_type to easy-crd-constructors
  - re-cython .pyx files with Cython 0.25.2

Bugfixes:
  - extend_boundaries_ndarr was broken for flat vector arrays
  - array dimensions in Cotr are now standardized and documented properly
  - making vector masks (ie, of edge / face centered fields) now works
  - mapfields were funny when show = True
  - tests that use vtk 6.3 w/ Linux + MESA
  - round_datetime straight up didn't work
  - fix external call to meshlabserver

0.98.1
======

Bugfixes:
  - fix make_rotation_matrix new_x option

0.98.0
======

Changes:
  - Support for Edge / Face centered fields (interpolate, streamline, fill_dipole, div, fc2cc, ec2cc)
  - Spherical seeds now take crd_system argument that changes the meaning of phi=0 from the +x axis to midnight
  - New cotr module to facilitate geophysical coordinate transformations
  - dipole utilities use new cotr module
  - div, grad, curl now work with both numpy and numexpr
  - Allow an outer boundary radius for streamlines
  - let make_rotation_matrix roll to a specific x axis

Bugfixes:
  - fix cythonize errors for cython >= 0.24
  - misc. hdf5/xdmf bugs when saving fields
  - getting shaped coordinates

0.97.0
======

Changes:
  - Added Field.loc[...] for pandas-like slice by location
  - add 'viscid.as_mapfield', 'viscid.as_spherefield', 'as_polar_mapfield', and 'pts2polar_mapfield' for converting between different representations for spherical fields (theta/phi <-> lat/lon)
  - Spherical seed generators now plot on polar projection by default in MPL
  - Switch all internals from Datetime to numpy.datetime64 and numpy.timedelta64
  - Use of the new viscid.as_datetime64 and viscid.as_timedelta64 is HIGHLY recommended. This does the extra step of setting Zulu time if a tz is not specified for older versions of numpy (numpy >= 1.11 is tz agnostic).
  - Openggcm now correctly calls the simulation start time basetime and makes every attempt for this to be as precise as possible for hdf5/jrrle/etc and still be lazy.
  - Viscid-colorblind defaults to viridis cmap

Preliminary:
  - dipole tools
  - gradient

Bugfixes:
  - Sphere / Cap / Circle are now more standardized

Backward Incompatible Changes:
  - Spherical coordinates now use theta/phi or lat/lon always as axis names

0.96.1
======

Changes:
  - A lot of new Mayavi convenience wrappers in viscid.plot.mvi. Most things can now be done using viscid objects without much effort. See the improved Viscid/tests/test_mvi.py for examples of using the new wrappers.
  - When using the viscid.plot.mvi convenience wrappers, the colormaps automatically pick up the matplotlib defaults, including the `viscid.symmetric_cmap` default.

0.96.0
======

Changes:
  - Add developer's guide to the docs
  - Added `mpl_style` module to deprecate `mpl_extra` and `vseaborn`
  - Use nogil for streamlines and a threads option for lighterweight parallelism
  - Added simple functions to find separators in topology Fields. Separator tracing is still preliminary and the interface may change in the future. Usage examples are available on the Examples/Calculator page of the docs.
  - Added some untested minumum variance tools
  - Added some magnetopause specific tools

Deprecated:
  - setting `mpl_extra` attributes from the rc file
  - seaborn styling can be completely done in matplotlib >= 1.5.0

0.95.1
======

Changes:
  - Add some mpl wrappers (clf, subplot, subplot2grid, savefig, show)
  - Let jrrle files get current with 'jx', 'jy', 'jz'
  - Tests now exersize xdmf/jrrle files for 3d/2d/timeseries/iof the same way
  - Always remove old plots when running make check

Bugfixes:
  - Tests run on py3k again
  - fix custom grids on jrrle files
  - flip xjx, xjy when going mhd->gse

0.95.0
======

Changes
  - Travis-CI builds upload test summaries to Github Pages
  - Makefiles now have rules for all make / deploy steps
  - Wrap savefig with offscreen rendering on linux

0.94.0
======

Changes:
  - Change sample data
  - add continuous integration tests (travis-ci)
  - Makefiles can now be used for convenience

0.91.3
======

Changes:
  - add workaround comment for fortran build on OS X El Capitain

0.91.2
======

Changes:
  - bugfixes

0.91.0
======

Changes:
  - Use LinearLocator for linear colorbars by default
  - better error message when trying to mpl.plot vector fields
  - enable ionosphere plots in mayavi, see test_mvi.py
  - enable plotting meshes in mayavi, see test_mvi.py

Backward Incompatible Changes:
  - Seeds now use a unified interface. If you use seeds, check out test_seed.py to see how things work now
  - plot_lines (both matplotlib and mayavi) use a unified interface, check out test_streamline to see how things work now

Refactors:
  - blocks -> patches everywhere
  - block -> patch everywhere
  - vlab.multiplot argument nprocs -> nr_procs
  - lots of names in `viscid.plot.mvi` and `viscid.calculator.topology`

Bugfixes:
  - ibound topology detection (north/south accidently became dayside/nightside in version 0.90.0)

0.90.1
======

Changes:
  - xl and xh properties added to Fields and AMRFields
  - Fields now have get_slice_extent

Bugfixes:
  - XDMF reading uniform crds was wrong; ggcm is using zyx order for O+dxdydz since that seems to work for Paraview, but the xdmf docs call for xyz order, so something isn't kosher here but I guess this works for now
  - xyz ordering for scalar_fields_to_vector
  - TypeError on numpy reduction operations on ScalarFields
  - fix Volume genr_points for xyz ordering

0.90.0
======

Backward Incompatible Changes:
  - ALL indexing is now in the natural order (xyz as opposed to zyx). This means seeds, lines, fields... everything. Since this is such a big change that unavoidably breaks many things, this is the only feature of this release, so if you have code that depends on zyx ordering, stay at version 0.80.8.

0.80.9
======

Bugfixes:
  - vector -> scalar numpy reduction operations

0.80.8
======

Changes:
  - Files can now be unloaded (treating a file like a context manager will unload it on __exit__)
  - jrrle readers can handle files with ascii in them
  - hdf5 datawrappers can specify a component dimension / index
  - preliminary gkeyll reader
  - add `mpl.auto_adjust_subplots()` which is like tighen, but doesn't change axes limits on you.
  - All fields (regular + amr) now have an 'interpolated_slice' method
  - Cython Interpolate and Streamline now work on AMR Fields

Bugfixes:
  - calling ufuncs with AMR fields
  - numpy broadcasting with Fields of different type / shape
  - slicing by UT time was relative to 0 instead of the start of the simulation

Backward Incompatible Changes:
  - Slices by value are now a string of '[0-9\.]+f'. Slicing by floats prints a warning.

0.80.7
======

Backward Incompatible Changes:
  - Arguments to viscid.field.wrap_field have changed
  - typ arguments have been refactored to fldtype and crdtype
  - add Field methods real, imag, astype

Changes:
  - add support for calling `python -im viscid pylab`
  - add viscid.field.arrays2field and viscid.coordinates.arrays2crds
  - viscid.field now has empty, ones, zeros, empty_like, ones_like, zeros_like that act roughly the same as the numpy functions, except that these create fields

0.80.6
======

Changes:
  - Ascii dataset field names are now "c[0-9]+" where the number is the column number
  - Add a way to generate your own cubehelix colormaps
  - Sphinx has no more warnings

0.80.5
======

Features:
  - add amr_field.patch_indices to lookup patch index at a location

Bugfixes:
  - AMR slicing bugs

0.80.4
======

Bugfixes:
  - ./setup.py install wasn't installing parsers
  - command line utilities had bugs

0.80.3
======

Bugfixes:
  - Fix size of earth in AMR files
  - Fix matplotlib colorscale options

0.80.2
======

Features:
  - RC file and plot_opts can be given in Yaml syntax if user has PyYaml, otherwise the rc file has to be JSON and the plot_opts have to use the weird comma/space syntax

Bugfixes:
  - amr fields now work with command line utils

0.80.1
======

Bugfix:
  - vjson not found

0.80.0
======

Features:
  - preliminary amr support (1D and 2D matplotlib plots)

Backward Incompatible Changes:
  - rc file is now in JSON format (in the future, this should change to YAML, but JSON is a subset of YAML, so it shouldn't break in the future)

Changes:
  - 1:1 match between plot_opts and plot keyword arguments

Bugfixes:
  - Ionosphere files wouldn't plot with pcolormesh

0.60.3
======

Features:
  - thousands of jrrle files load fast over sshfs
  - when loading xdmf files, one can specify a root directory for hdf5 files
    so one can copy xdmf files locally and read the hdf5 files over sshfs

Changes:
  - Lots of little bugfixes

0.60.2
======

Features:
  - The attribute `grid.Grid.longterm_field_caches` controls how long caches hang around
  - Athena binary / ascii / hst readers (doesn't read SMR grids)
  - Can hyperslice into xdmf/HDF5 datasets so only data that is needed is read
  - Slicing in time can now take H:M:S styled times
  - New ways to print / get the time of a Dataset/Grid/Field
    - `format_time`
    - `time_as_datetime`

Changes:
  - Spatial slices completely rewritten
  - info dicts (for Datasets/Grids/Fields) are now private
    - `get_info(key)`: gets info from a specific object
    - `find_info(key)`: gets info from an object, or its parents
    - `set_info(key)`: sets info for an object
    - `update_info(key)`: updates info wherever in the tree it exists

Other:
  - Customization attributes have their own sphinx page for easy lookup
  - performance enhancements when globbing files on a remote server

0.60.1
======

Bugfix:
  - Time slices were acting silly

0.60.0
======

Features:
  - Implement Ionosphere reading / plotting
  - Implement a Jrrle Openggcm Reader

Backward Incompatible Changes:
  - CHANGE FIELD SLICING SYNTAX! No more trailing 'i' to slice by index. Instead, use an integer to mean an index. For slicing by coordinate value, the decimal now needs to be explicit, so all code that slices like "y=0" now needs to be refactored to "y=0.0"

0.50.2
======

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

0.50.1
======

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

0.50.0 Release
==============

Features:
  - Remove lxml dependency
  - Remove Cython dependency
  - Works with the older Python 2.6
