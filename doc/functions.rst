Useful Functions
================

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
:py:func:`viscid.zeros`              Make an field of 0s from coordinate arrays
:py:func:`viscid.ones`               Make an field of 1s from coordinate arrays
:py:func:`viscid.empty_like`         Make an empty field like another
:py:func:`viscid.zeros_like`         Make a field of 1s like another
:py:func:`viscid.ones_like`          Make a field of 0s like another
===================================  ===========================================================

Calculations
------------

Streamlines and Interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cssclass:: table-striped

===================================  =================================================
Class                                Description
===================================  =================================================
:py:func:`viscid.calc_streamlines`   Calculate streamlines
:py:func:`viscid.interp`             Interpolation, use `kind` kwarg for trilinear /
                                     nearest neighbor
:py:class:`viscid.Point`             Collection of hand picked points
:py:class:`viscid.Line`              A line between 2 points
:py:class:`viscid.Plane`             A plane defined by an origin and a normal vector
:py:class:`viscid.Volume`            A Volume of points on a uniform cartesian grid
:py:class:`viscid.Sphere`            Points on the surface of a sphere
:py:class:`viscid.SphericalCap`      A cap of points around the pole of a sphere
:py:class:`viscid.Circle`            Just a circle
:py:class:`viscid.SphericalPatch`    A rectangular patch on the surface of a sphere
===================================  =================================================

Math
~~~~

These functions will by accelerated by Numexpr if it is installed.

.. cssclass:: table-striped

========================================  ===========================================================
Function                                  Description
========================================  ===========================================================
:py:func:`viscid.add`                     Add two fields
:py:func:`viscid.diff`                    Subtract a field from another
:py:func:`viscid.mul`                     Multiply two fields
:py:func:`viscid.relative_diff`           Divide the difference by the magnitude
:py:func:`viscid.abs_diff`                Absolute value of the difference
:py:func:`viscid.abs_val`                 Absolute value
:py:func:`viscid.abs_max`                 Max of the absolute value
:py:func:`viscid.abs_min`                 Min of the absolute value
:py:func:`viscid.magnitude`               Magnitude of a :py:class:`viscid.field.VectorField`
:py:func:`viscid.dot`                     Dot product of two :py:class:`viscid.field.VectorField`
:py:func:`viscid.cross`                   Cross product of two :py:class:`viscid.field.VectorField`
:py:func:`viscid.div`                     Divergence of a :py:class:`viscid.field.VectorField`
:py:func:`viscid.curl`                    Curl of a :py:class:`viscid.field.VectorField`
:py:func:`viscid.project`                 Project one :py:class:`viscid.field.VectorField` onto
                                          another
:py:func:`viscid.integrate_along_lines`   Integrate a field along streamlines
:py:func:`viscid.calc_psi`                Calculate a 2D flux function
:py:func:`viscid.calc_beta`               Calculate plasma beta
========================================  ===========================================================

Magnetic Topology and Separator Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cssclass:: table-striped

For using the separator tools, you may want to refer to :doc:`../examples/calc`.

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

Plotting
--------

General Matplotlib Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These functions wrap Matplotlib with useful boilerplate type hacks.

.. cssclass:: table-striped

================================================  ===========================================================
Function                                          Description
================================================  ===========================================================
:py:func:`viscid.plot.mpl.auto_adjust_subplots`   Use Matplotlib's tight layout with some necessary hacks
================================================  ===========================================================

2D Matplotlib Plots
~~~~~~~~~~~~~~~~~~~

These functions wrap Matplotlib with useful boilerplate type hacks.

.. cssclass:: table-striped

================================================  =============================================================
Function                                          Description
================================================  =============================================================
:py:func:`viscid.plot.mpl.plot`                   Meta function for plotting :py:class:`viscid.field.Field`
                                                  objects. This one will automatically delegate to
                                                  :py:func:`viscid.plot.mpl.plot1d_field`,
                                                  :py:func:`viscid.plot.mpl.plot2d_field`, or
                                                  :py:func:`viscid.plot.mpl.plot2d_mapfield`.
:py:func:`viscid.plot.mpl.plot1d_field`           Line plots of a 1D field.
:py:func:`viscid.plot.mpl.plot2d_field`           Colored plots (pcolormesh, contour, contourf) of 2D fields
:py:func:`viscid.plot.mpl.plot2d_mapfield`        Plots on the surface of a sphere (like ionosphere plots)
:py:func:`viscid.plot.mpl.plot2d_lines`           Plot a list of colored lines parallel-projected into 2D
:py:func:`viscid.plot.mpl.plot2d_quiver`          Plot a :py:class:`viscid.field.VectorField` using
                                                  Matplotlib's quivers.
:py:func:`viscid.plot.mpl.streamplot`             Plot a :py:class:`viscid.field.VectorField` using
                                                  Matplotlib's streamplot.
:py:func:`viscid.plot.mpl.plot_earth`             Plot an Earth with black for nightside and white for dayside
================================================  =============================================================

3D Matplotlib Plots
~~~~~~~~~~~~~~~~~~~

These functions wrap Matplotlib in 3D with useful boilerplate type hacks.

.. cssclass:: table-striped

===============================================  =============================================================
Function                                         Description
===============================================  =============================================================
:py:func:`viscid.plot.mpl.plot3d_lines`          Plot a list of colored lines on 3D axes
:py:func:`viscid.plot.mpl.scatter_3d`            Plot a glyphs on 3D axes
===============================================  =============================================================

3D Mayavi Plots
~~~~~~~~~~~~~~~

These functions wrap Mayavi with useful boilerplate type hacks.

.. cssclass:: table-striped

===============================================  =================================================================
Function                                         Description
===============================================  =================================================================
:py:func:`viscid.plot.mvi.add_source`            Given a VTKDataSource, add it to a figure
:py:func:`viscid.plot.mvi.add_lines`             Given a list of lines, add them to a figure as a data source
:py:func:`viscid.plot.mvi.add_lines`             Given a :py:class:`viscid.field.Field`, add it to a figure as
                                                 a data source
:py:func:`viscid.plot.mvi.plot_lines`            Plot colored lines in 3D
:py:func:`viscid.plot.mvi.plot_ionosphere`       Plot an ionospheric Field in 3D
:py:func:`viscid.plot.mvi.insert_filter`         Insert a filter above a module_manager.
:py:func:`viscid.plot.mvi.plot_blue_marble`      Plot an Earth using the blue marble NASA image
:py:func:`viscid.plot.mvi.plot_earth_3d`         Plot an Earth with black for nightside and white for dayside
:py:func:`viscid.plot.mvi.clf`                   Uses some hacks to clear a figure and make sure memory is freed
:py:func:`viscid.plot.mvi.resize`                Uses some hacks to resize a figure
===============================================  =================================================================
