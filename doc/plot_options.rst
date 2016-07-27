Plot Options
============

Plot options are given to :py:func:`viscid.plot.mpl.plot` as either keyword arguments, or as a string called `plot_opts`. For the various plot types, what follows should be a comprehensive list of the available options.

.. note::

  For all options, supplying no arguments implies one argument with the value True.

.. note::

  Any options that Viscid does not understand are passed to the matplotlib function that actually plots the data (`pyplot.plot`, `pyplot.pcolormesh`, `pyplot.contourf`, etc).

Multiple arguments are given to keywords as a tuple:

.. code-block:: python

  kwargs = {'lin': (-1.0, 1.0), 'gridec': 'k'}

All options can be passed as strings with one of two formats. First, the string can be a list of comma separated options with underscore separatod arguments:

.. code-block:: python

  plot_opts = "lin_-1.0_1.0, gridec_k"
  # also, any '=' will be substituted with a '_' to aid readability
  plot_opts = "lin=-1.0_1.0, gridec=k"

**If** PyYaml is available, plot options can be given between curly brackets using Yaml syntax. Note that the colon must be followed by a space in Yaml syntax.

.. code-block:: python

  plot_opts = "{lin: [-1.0, 1.0], gridec: k}"

Here are some options that should work for all functions:

.. cssclass:: table-striped

=============  ================  ==================================================
Option         Arguments         Description
=============  ================  ==================================================
clim           vmin, [vmax]      Set min and max values for the data scale
vmin           int               Minimum value for the data range
vmax           int               Maximum value for the data range
lin            [vmin, [vmax]]    Shorthand for clim and logscale=0. Min/Max values
                                 that aren't given are found from the data.
log            [vmin, [vmax]]    Shorthand for clim and logscale=1. Min/Max values
                                 that aren't given are found from the data.
loglog         [vmin, [vmax]]    Same as log, but also use a logscale for the
                                 coordinates
logscale       [bool]            Use a logarithmic scale for the data
symmetric      [bool]            Make the data scale symmetric around 0
norescale      [bool]            Do not set limit of the data axis explicitly
x              min, max          Set axis limits using :py:func:`pyplot.set_xlim`
y              min, max          Set axis limits using :py:func:`pyplot.set_ylim`
equalaxis      [bool]            Force 1:1 aspect ratio
scale          float             Scale data by some scalar value
masknan        [bool or color]   Mask out NaN values in data with a given color
                                 (default: 'y' for yellow)
flipplot       [bool]            Alias for flip_plot
flip_plot      [bool]            Flip the horizontal and vertical axes
nolabels       [bool]            Skip applying labels to x/y/cbar axes
xlabel         str               Specific label for the x axis
ylabel         str               Specific label for the y axis
majorfmt       ticker.Formatter  Formatter for major axes (x and y)
minorfmt       ticker.Formatter  Formatter for minor axes (x and y)
majorloc       ticker.Locator    Locator for major axes (x and y)
minorloc       ticker.Locator    Locator for minor axes (x and y)
datefmt        str               date format string in the datetime.strftime format
timefmt        str               time format string in the datetime.strftime format
                                 (used for timedeltas)
autofmt_xdate  [bool]            auto-rotate date labels on the x-axis
autofmtxdate   [bool]            alias for autofmt_xdate
show           [bool]            Call :py:func:`pyplot.show` before returning
=============  ================  ==================================================

.. note::
  When using lin, if vmin == 0 and vmax is not given, the scale will be symmetric
  about 0. This is a shorthand for the symmetric keyword argument.


2-D Plots
---------

.. cssclass:: table-striped

==========  ===============   ==================================================
Option      Arguments         Description
==========  ===============   ==================================================
style       str               One of (pcolormesh, pcolor, contour, contourf)
levels      [int or list]     Number of contours, or list of specific contour
                              values (contours only)
g           [bool or color]   Alias for gridec=k or gridec=color
gridec      color             Color for grid lines (pcolormesh only)
gridlw      number            Line width for grid lines (pcolormesh only)
gridaa      [bool]            Antialias grid lines (default: True)
p           [bool or color]   Alias for patchec=k or patchec=color
patchec     color             Color for patch boundaries
patchlw     number            Line width for patch boundaries
patchaa     [bool]            Antialias patch boundaries (default: True)
mod         modx, mody        Scale coordinates by some scalar value
colorbar    [bool or dict]    dict of keyword arguments for
                              :py:func:`pyplot.colorbar`
title       bool or str       Put a specific title on the plot, or if true, use
                              field's pretty_name (suppresses cbarlabel if both
                              would default to pretty_name)
cbarlabel   str               Specific label for the color bar
earth       [bool]            Plot a black and white circle for Earth
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
drawcoastlines  [bool]            If projection is a basemap projection, then draw
                                  coastlines. Pretty cool, but not actually useful.
                                  Coastlines do NOT reflect UT time; London is
                                  always at midnight.
lon0            float             Center longitude (basemap projections only)
lat0            float             Center latitude (basemap projections only)
boundinglat     float             Bounding latitude in degrees from the nearest pole
                                  (not for all projections)
title           bool or str       Put a specific title on the plot, or if true, use
                                  field's pretty_name
axgridec        color             Color for patch boundaries (use empty string or
                                  False to turn off axes)
axgridlw        number            Line width for patch boundaries
axgridls        str               Line style for patch boundaries
labellat        [bool or str]     Alias for label_lat
label_lat       [bool or str]     Label latitudes at 80, 70, 60 degrees
                                  with sign indicating northern / southern hemisphere.
                                  If label_lat is 'from_pole', then the labels are 10,
                                  20, 30 for both hemispheres. Note that basemap
                                  projections won't label latitudes unless they hit the
                                  edge of the plot.
label_mlt       [bool]            label magnetic local time
==============  ===============   ==================================================


1-D Color Plots
---------------

.. cssclass:: table-striped

==========  ===============   ==================================================
Option      Arguments         Description
==========  ===============   ==================================================
legend      [loc]             call :py:func:`pyplot.legend`
label       str               Label for the data series
mod         modx              Scale coordinates by some scalar value
==========  ===============   ==================================================
