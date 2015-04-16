Plot Options
============

Plot options are given to :py:func:`viscid.plot.mpl.plot` as either keyword arguments, or as a string called `plot_opts`. For the various plot types, what follows should be a comprehensive list of the available options.

.. note::

  For all options, supplying no arguments implies one argument with the value True.

.. note::

  Any options that Viscid does not understand are passed to the matplotlib function that actually plots the data (`pyplot.plot`, `pyplot.pcolormesh`, `pyplot.contourf`, etc).

Multiple arguments are given to keywords as a tuple, such as:

.. code-block:: python

  kwargs = {'lin': (-1.0, 1.0), 'gridec': 'k'}

or to the plot_opts string using underscores for separation:

.. code-block:: python

  plot_opts = "lin_-1.0_1.0, gridec_k"
  # also, any '=' will be substituted with a '_' to aid readability
  plot_opts = "lin=-1.0_1.0, gridec=k"

Here are some options that should work for all functions:

.. cssclass:: table-striped

==========  ===============   ==================================================
Option      Arguments         Description
==========  ===============   ==================================================
lin         [vmin, [vmax]]    Use a linear scale that goes from vmin to vmax.
                              This sets the color map in 2D, or the axis scaling
                              in 1D. If vmin or vmax are None, they are set using
                              the data.
log         [vmin, [vmax]]    Similar to lin, but with a logarithmic scale
loglog      [vmin, [vmax]]    Same as log, but also use a logscale for the
                              coordinates
logscale    bool              Use a logarithmic scale for the data
symetric    bool              Make the data scale symetric around 0
x           min, max          Set axis limits using :py:func:`pyplot.set_xlim`
y           min, max          Set axis limits using :py:func:`pyplot.set_ylim`
equalaxis   bool              Force 1:1 aspect ratio
scale       bool              Scale data by some scalar value
masknan     bool              Mask out NaN values in data
flipplot    bool              Alias for flip_plot
flip_plot   bool              Flip the horizontal and vertical axes
dolabels    bool              Alias for do_labels
do_labels   bool              Apply labels to the axes / colorbars
xlabel      str               Specific label for the x axis
xlabel      str               Specific label for the y axis
show        bool (optional)   Call :py:func:`pyplot.show` before returning
==========  ===============   ==================================================

.. note::
  When using lin, if vmin == 0 and vmax is not given, the scale will be symetric
  about 0. This is a shorthand for the symetric keyword argument.


2-D Plots
---------

.. cssclass:: table-striped

==========  ===============   ==================================================
Option      Arguments         Description
==========  ===============   ==================================================
style       str               One of (pcolormesh, pcolor, contour, contourf)
levels      number or list    Number of contours, or list of specific contour
                              values (contours only)
g           bool or color     Alias for gridec=k or gridec=color
gridec      color             Color for grid lines (pcolormesh only)
gridlw      number            Line width for grid lines (pcolormesh only)
patchaa     bool              Antialias grid lines (default: True)
p           bool or color     Alias for patchec=k or patchec=color
patchec     color             Color for patch boundaries
patchlw     number            Line width for patch boundaries
patchaa     bool              Antialias patch boundaries (default: False)
mod         modx, mody        Scale coordinates by some scalar value
colorbar    bool, dict        dict of keyword arguments for
                              :py:func:`pyplot.colorbar`
cbarlabel   str               Specific label for the color bar
earth       None              Plot a black and white circle for Earth
==========  ===============   ==================================================


2-D Map Plots
-------------

All options for normal 2-D plots work for map plots too.

.. cssclass:: table-striped

==============  ===============   ==================================================
Option          Arguments         Description
==============  ===============   ==================================================
projection      str               'polar' or Basemap projection to use
hemisphere      str               'north' or 'south'
drawcoastlines  bool              If projection is a basemap projection, then draw
                                  coastlines. Pretty cool, but not actually useful.
                                  Coastlines do NOT reflect UT time; London is
                                  always at midnight.
lon0            float             Center longitude (basemap projections only)
lat0            float             Center latitude (basemap projections only)
boundinglat     float             Bounding latitude in degrees from the nearest pole
                                  (not for all projections)
title           bool or str       Put a specific title on the plot, or if true, use
axgridec        color             Color for patch boundaries (use empty string or
                                  False to turn off axes)
axgridlw        number            Line width for patch boundaries
axgridls        str               Line style for patch boundaries
labellat        bool or str       Alias for label_lat
label_lat       bool or str       Label latitudes at 80, 70, 60 degrees
                                  with sign indicating northern / southern hemisphere.
                                  If label_lat is 'from_pole', then the labels are 10,
                                  20, 30 for both hemispheres. Note that basemap
                                  projections won't label latitudes unless they hit the
                                  edge of the plot.
label_mlt       bool              label magnetic local time
==============  ===============   ==================================================


1-D Color Plots
---------------

.. cssclass:: table-striped

==========  ===============   ==================================================
Option      Arguments         Description
==========  ===============   ==================================================
legend      loc               call :py:func:`pyplot.legend`
label       str               Label for the data series
mod         modx              Scale coordinates by some scalar value
==========  ===============   ==================================================
