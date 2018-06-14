Useful Functions
================

.. contents::
  :local:

Utility Functions
-----------------

===================================  ===============================================
Function                             Description
===================================  ===============================================
:py:func:`viscid.load_file`          Load a single file, or glob of a single time
                                     series
:py:func:`viscid.load_files`         Load multiple files
:py:func:`viscid.unload_all_files`   Clear the cache of loaded files
:py:func:`viscid.interact`           Stop the program and get an interactive prompt
                                     at any point (uses IPython if it's installed).
===================================  ===============================================

Fields
------

For easy field creation.

.. cssclass:: table-striped

===================================  ===========================================================
Function                             Description
===================================  ===========================================================
:py:func:`viscid.arrays2field`       Field from arrays for data and coordinates
:py:func:`viscid.dat2field`          Field from data array
:py:func:`viscid.empty`              Make an empty field from coordinate arrays
:py:func:`viscid.full`               Make an initialized field from coordinate arrays
:py:func:`viscid.zeros`              Make an field of 0s from coordinate arrays
:py:func:`viscid.ones`               Make an field of 1s from coordinate arrays
:py:func:`viscid.empty_like`         Make an empty field like another
:py:func:`viscid.full_like`          Make an initialized field like another
:py:func:`viscid.zeros_like`         Make a field of 1s like another
:py:func:`viscid.ones_like`          Make a field of 0s like another
===================================  ===========================================================

Calculations
------------

Streamlines and Interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cssclass:: table-striped

========================================  ==================================================
Class                                     Description
========================================  ==================================================
:py:func:`viscid.calc_streamlines`        Calculate streamlines
:py:func:`viscid.interp`                  Interpolation, use `kind` kwarg for trilinear /
                                          nearest neighbor
:py:class:`viscid.Point`                  Collection of hand picked points
:py:class:`viscid.RectilinearMeshPoints`  Points that can be 2d plotted using [u, :, 0] and
                                          [v, 0, :] slices of pts as coordinate arrays
:py:class:`viscid.Line`                   A line between 2 points
:py:class:`viscid.Plane`                  A plane defined by an origin and a normal vector
:py:class:`viscid.Volume`                 A Volume of points on a uniform cartesian grid
:py:class:`viscid.Sphere`                 Points on the surface of a sphere
:py:class:`viscid.SphericalCap`           A cap of points around the pole of a sphere
:py:class:`viscid.Circle`                 Just a circle
:py:class:`viscid.SphericalPatch`         A rectangular patch on the surface of a sphere
========================================  ==================================================

Math
~~~~

These functions will by accelerated by Numexpr if it is installed. All functions below are also available from the `viscid` namespace.

.. cssclass:: table-striped

===================================================  ===========================================================
Function                                             Description
===================================================  ===========================================================
:py:func:`viscid.calculator.calc.add`                Add two fields
:py:func:`viscid.calculator.calc.diff`               Subtract a field from another
:py:func:`viscid.calculator.calc.mul`                Multiply two fields
:py:func:`viscid.calculator.calc.relative_diff`      Divide the difference by the magnitude
:py:func:`viscid.calculator.calc.abs_diff`           Absolute value of the difference
:py:func:`viscid.calculator.calc.abs_val`            Absolute value
:py:func:`viscid.calculator.calc.abs_max`            Max of the absolute value
:py:func:`viscid.calculator.calc.abs_min`            Min of the absolute value
:py:func:`viscid.calculator.calc.magnitude`          Magnitude of a :py:class:`viscid.field.VectorField`
:py:func:`viscid.calculator.calc.dot`                Dot product of two :py:class:`viscid.field.VectorField`
:py:func:`viscid.calculator.calc.cross`              Cross product of two :py:class:`viscid.field.VectorField`
:py:func:`viscid.calculator.calc.grad`               Gradient of a :py:class:`viscid.field.ScalarField`
:py:func:`viscid.calculator.calc.convective_deriv`   A dot grad B for vector field A and scalar/vector filed B
:py:func:`viscid.calculator.calc.div`                Divergence of a :py:class:`viscid.field.VectorField`
:py:func:`viscid.calculator.calc.curl`               Curl of a :py:class:`viscid.field.VectorField`
:py:func:`viscid.calculator.calc.normalize`          Divide a vector field by its magnitude
:py:func:`viscid.calculator.calc.project`            Project one :py:class:`viscid.field.VectorField` onto
                                                     another, i.e., `a dot b / |b|`
:py:func:`viscid.set_in_region`                      Set values in one field from another given a mask
:py:func:`viscid.calculator.calc.project_vector`     Project VectorField a onto b in the direction of b, i.e.,
                                                     `(a dot b / |b|) * (b / |b|)`
:py:func:`viscid.project_along_line`                 Project a Vector Field Parallel to a streamline.
:py:func:`viscid.resample_lines`                     Resample a list of lines to either more or fewer points.
                                                     With scipy, oversampling can be done with any type of
                                                     interpolation that :py:func:`scipy.interpolate.interp1d`
                                                     understands.
:py:func:`viscid.integrate_along_lines`              Integrate a field along streamlines
:py:func:`viscid.calc_psi`                           Calculate a 2D flux function
:py:func:`viscid.calc_beta`                          Calculate plasma beta
===================================================  ===========================================================

Geospace Tools
~~~~~~~~~~~~~~

These functions allow for transforming between geophysical coordinate systems, and adding a magnetic dipole to a field.

.. cssclass:: table-striped

================================================  ============================================================
Function                                          Description
================================================  ============================================================
:py:func:`viscid.Cotr`                            Object that facilitates geospace coordinate transformations
                                                  at a given UT time
:py:func:`viscid.cotr_transform`                  Transform a vector from one crd system to another at a
                                                  given UT time
:py:func:`viscid.get_dipole_moment`               Get Earth's dipole moment at a given time in any coordinate
                                                  system
:py:func:`viscid.get_dipole_moment_ang`           Get dipole moment given gsm-tilt and dipole-tilt angles in
                                                  gse or mhd crds
:py:func:`viscid.make_dipole`                     Create new dipole vector field to an existing field given
                                                  dipole moment vector
:py:func:`viscid.fill_dipole`                     Add dipole vector field to an existing field given dipole
                                                  moment vector (can be masked)
:py:func:`viscid.make_spherical_mask`             Make a spherically shaped mask (useful in conjunction with
                                                  :py:func:`viscid.fill_dipole`)
================================================  ============================================================

Magnetosphere Tools
~~~~~~~~~~~~~~~~~~~

Some tools for dealing with magnetospheric specific things. Refer to :doc:`../tutorial/magnetopause` for an example

.. cssclass:: table-striped

=============================================  ============================================================
Function                                       Description
=============================================  ============================================================
:py:func:`viscid.get_mp_info`                  Extract magnetopause info (possibly cached)
:py:func:`viscid.find_mp_edges`                Find edges of the magnetopause current sheet
=============================================  ============================================================

Magnetic Topology and Separator Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For using the separator tools, you may want to refer to :doc:`../tutorial/magnetic_topology`.

.. cssclass:: table-striped

=============================================  ============================================================
Function                                       Description
=============================================  ============================================================
:py:func:`viscid.topology2color`               Turn topology bitmask into colors
:py:func:`viscid.trace_separator`              **Still in testing** Trace a separator line using bisection
                                               algorithm
:py:func:`viscid.get_sep_pts_bisect`           **Still in testing** Use bisection algorithm to find one or
                                               more separators locations for a seed
:py:func:`viscid.get_sep_pts_bitor`            **Still in testing** Use bitwise-or algorithm to find one or
                                               more separators locations for a seed
:py:func:`viscid.topology_bitor_clusters`      Use bitwise-or algorithm to find one or more separators in a
                                               topology Field
=============================================  ============================================================

Matplotlib
----------

General Matplotlib Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These functions wrap Matplotlib with useful boilerplate type hacks.

.. cssclass:: table-striped

====================================================  ===========================================================
Function                                              Description
====================================================  ===========================================================
:py:func:`viscid.plot.vpyplot.auto_adjust_subplots`   Use Matplotlib's tight layout with some necessary hacks
:py:func:`viscid.plot.vpyplot.apply_labels`           Write labels onto a plot similar to a legend, but place
                                                      the labels next to the data. This adheres to the principle
                                                      that you shouldn't make the reader learn a key just to
                                                      read a single plot.
:py:func:`viscid.plot.vpyplot.despine`                Remove some spines to reduce chart clutter, this is the
                                                      exact same as seaborn's despine.
====================================================  ===========================================================

2D Matplotlib Plots
~~~~~~~~~~~~~~~~~~~

These functions wrap Matplotlib with useful boilerplate type hacks.

.. cssclass:: table-striped

================================================  =============================================================
Function                                          Description
================================================  =============================================================
:py:func:`viscid.plot.vpyplot.plot`               Meta function for plotting :py:class:`viscid.field.Field`
                                                  objects. This one will automatically delegate to
                                                  :py:func:`viscid.plot.vpyplot.plot1d_field`,
                                                  :py:func:`viscid.plot.vpyplot.plot2d_field`, or
                                                  :py:func:`viscid.plot.vpyplot.plot2d_mapfield`.
:py:func:`viscid.plot.vpyplot.plot1d_field`       Line plots of a 1D field.
:py:func:`viscid.plot.vpyplot.plot2d_field`       Colored plots (pcolormesh, contour, contourf) of 2D fields
:py:func:`viscid.plot.vpyplot.plot2d_mapfield`    Plots on the surface of a sphere (like ionosphere plots)
:py:func:`viscid.plot.vpyplot.plot_iono`          make annotated polar plots of ionosphere quantities, this
                                                  is just a wrapper for plot2d_mapfield that handles small
                                                  annyoances and annotations
:py:func:`viscid.plot.vpyplot.plot2d_lines`       Plot a list of colored lines parallel-projected into 2D
:py:func:`viscid.plot.vpyplot.plot2d_quiver`      Plot a :py:class:`viscid.field.VectorField` using
                                                  Matplotlib's quivers.
:py:func:`viscid.plot.vpyplot.streamplot`         Plot a :py:class:`viscid.field.VectorField` using
                                                  Matplotlib's streamplot.
:py:func:`viscid.plot.vpyplot.plot_earth`         Plot an Earth with black for nightside and white for dayside
================================================  =============================================================

3D Matplotlib Plots
~~~~~~~~~~~~~~~~~~~

These functions wrap Matplotlib in 3D with useful boilerplate type hacks.

.. cssclass:: table-striped

===============================================  =============================================================
Function                                         Description
===============================================  =============================================================
:py:func:`viscid.plot.vpyplot.plot3d_lines`      Plot a list of colored lines on 3D axes
:py:func:`viscid.plot.vpyplot.scatter_3d`        Plot a glyphs on 3D axes
===============================================  =============================================================

Mayavi
------

Mayavi is the preferred library for making 3D plots with Viscid. It's a little unwieldy, but for the moment, it's still the best Python interface to VTK. Mayavi has two ways to learn how to change details about the objects in a given scene (the documentation reads like somebody was shooting buckshot). The first is to make the change interactively while using the record feature. The other is to throw `import IPython; IPython.embed()` into your script and go spunking. Most Mayavi objects come from Traited VTK, which means they have a `print_traits()` method. This method will print out all the attributes that you may want to tweak.

.. include:: _mayavi_install_note.rst

Between the :doc:`example <tutorial/mayavi>` and the functions you see below, you should be able figure out most things without too much hassle.

Mayavi Wrappers
~~~~~~~~~~~~~~~

Chances are that you want to use these functions. They let you make most Mayavi objects from Viscid data structures (i.e., Fields and SeedGens). In addition, all of these functions allow you to specify a Matplotlib colormap for the data, and it picks up the default colormaps from Matplotlib's rcParams and viscidrc files. How cool is that?

.. cssclass:: table-striped

===============================================  =================================================================
Function                                         Description
===============================================  =================================================================
:py:func:`viscid.plot.vlab.plot_lines`           Plot colored lines in 3D
:py:func:`viscid.plot.vlab.scalar_cut_plane`     Make a scalar cut plane of a Field or existing Mayavi source
:py:func:`viscid.plot.vlab.vector_cut_plane`     Make a vector cut plane of a Field or existing Mayavi source
                                                 with optional scalar data.
:py:func:`viscid.plot.vlab.mesh`                 Make a mesh from a 2D array of vertices with optional scalar
                                                 data
:py:func:`viscid.plot.vlab.mesh_from_seeds`      Make a mesh from a Viscid SeedGen object with optional scalar
                                                 data. Useful for displaying the result of interpolating a field
                                                 onto a plane or sphere.
:py:func:`viscid.plot.vlab.streamline`           Use the interactive Mayavi (VTK) streamline tracer with optional
                                                 scalar data
:py:func:`viscid.plot.vlab.iso_surface`          Make volumetric contours of a Field or existing Mayavi source
:py:func:`viscid.plot.vlab.points3d`             Plot a list of points
:py:func:`viscid.plot.vlab.quiver3d`             Plot a list of vector arrows with optional scalar data
:py:func:`viscid.plot.vlab.colorbar`             Wrap `mayavi.mlab.colorbar`, then change the colormap if any
                                                 :py:func:`viscid.plot.vlab.apply_cmap` kwargs are provided
:py:func:`viscid.plot.vlab.scalarbar`            Wrap `mayavi.mlab.scalarbar`, then change the colormap if any
                                                 :py:func:`viscid.plot.vlab.apply_cmap` kwargs are provided
:py:func:`viscid.plot.vlab.vectorbar`            Wrap `mayavi.mlab.vectorbar`, then change the colormap if any
                                                 :py:func:`viscid.plot.vlab.apply_cmap` kwargs are provided
:py:func:`viscid.plot.vlab.fancy_axes`           Make axes with 3 shaded walls and a grid similar to what
                                                 matplotlib and paraview have
:py:func:`viscid.plot.vlab.axes`                 Wrap `mayavi.mlab.axes`
:py:func:`viscid.plot.vlab.xlabel`               Wrap `mayavi.mlab.xlabel`
:py:func:`viscid.plot.vlab.ylabel`               Wrap `mayavi.mlab.ylabel`
:py:func:`viscid.plot.vlab.zlabel`               Wrap `mayavi.mlab.zlabel`
:py:func:`viscid.plot.vlab.title`                Wrap `mayavi.mlab.title`
:py:func:`viscid.plot.vlab.outline`              Wrap `mayavi.mlab.outline`
:py:func:`viscid.plot.vlab.orientation_axes`     Wrap `mayavi.mlab.orientation_axes`, adds the little xyz arrows
:py:func:`viscid.plot.vlab.view`                 Wrap `mayavi.mlab.view`, adjusts the focal point, distance, and
                                                 various angles of the camera
===============================================  =================================================================

Mayavi Plots
~~~~~~~~~~~~

These functions make some commonly used objects.

.. cssclass:: table-striped

===============================================  =================================================================
Function                                         Description
===============================================  =================================================================
:py:func:`viscid.plot.vlab.plot_ionosphere`      Plot an ionospheric Field in 3D on the surface of a sphere
:py:func:`viscid.plot.vlab.plot_blue_marble`     Plot an Earth using the blue marble NASA image
:py:func:`viscid.plot.vlab.plot_earth_3d`        Plot an Earth with black for nightside and white for dayside
===============================================  =================================================================

Mayavi Workarounds
~~~~~~~~~~~~~~~~~~

Mayavi has various platform specific bugs. These will try to apply workarounds so that they always give the expected result. If you see an error at runtime about QT API versions, you may need to set an environment variable::

    export QT_API="pyside"

.. cssclass:: table-striped

===============================================  =================================================================
Function                                         Description
===============================================  =================================================================
:py:func:`viscid.plot.vlab.clf`                  Uses some hacks to clear a figure and make sure memory is freed
:py:func:`viscid.plot.vlab.remove_source`        Safely remove a specific vtk source (and its memory)
:py:func:`viscid.plot.vlab.resize`               default resize is unreliable on OS X / Linux
:py:func:`viscid.plot.vlab.savefig`              offscreen rendering hack
===============================================  =================================================================

Mayavi Pipeline
~~~~~~~~~~~~~~~

Here are some functions to quickly bring Viscid datastructures into the Mayavi pipeline.

.. cssclass:: table-striped

===============================================  =================================================================
Function                                         Description
===============================================  =================================================================
:py:func:`viscid.plot.vlab.add_source`           Given a VTKDataSource, add it to a figure
:py:func:`viscid.plot.vlab.add_lines`            Given a list of lines, add them to a figure as a data source
:py:func:`viscid.plot.vlab.add_field`            Given a :py:class:`viscid.field.Field`, add it to a figure as
                                                 a data source
:py:func:`viscid.plot.vlab.insert_filter`        Insert a filter above a module_manager.
:py:func:`viscid.plot.vlab.apply_cmap`           Apply a colormap to an existing Mayavi object. This also lets
                                                 you quickly rescale the limits on the colorbar, or switch to a
                                                 log scale.
===============================================  =================================================================
