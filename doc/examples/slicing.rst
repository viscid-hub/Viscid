Slicing Examples
================


Spatial Slices
--------------

Fields can be sliced just like Numpy ndarrays, but you can also use an extended syntax to ask for a physical location instead of an index in the array. This example snips off the fist 50 cells and last 30 cells in x and plots the y = 0.0 plane for z = [-10.0..10.0].

.. plot::
    :include-source:

    import viscid
    from viscid.plot import mpl

    f3d = viscid.load_file(_viscid_root + '/../../sample/sample.3df.xdmf')

    # notice how slices by index appear as integers, and slices by location
    # are done with floats... this means "y=0" is not the same as "y=0.0"
    pp = f3d["pp"]["x=50:-30,y=0.0,z=-10.0:10.0"]
    mpl.plot(pp, style="contourf", levels=50, plot_opts="log,earth")

Temporal Slices
---------------

Temporal Datasets can be sliced a number of ways too, here are some examples

.. doctest::

    >>>> import viscid
    >>>> f3d = viscid.load_file(_viscid_root + '/../../sample/sample.3df.xdmf')
    >>>>
    >>>> grids = f3d.get_times(slice("UT1967:01:01:00:10:00.0",
    >>>>                             "UT1967:01:01:00:20:00.0"))
    >>>> print([grid.time for grid in grids])
    [600.0, 1200.0]
    >>>>
    >>>> grids = f3d.get_times(1)
    >>>> print([grid.time for grid in grids])
    [600.0]
    >>>>
    >>>> grids = f3d.get_times(slice(None, 600.0))
    >>>> print([grid.time for grid in grids])
    [600.0]
    >>>> grids = f3d.get_times("T0:10:00.0:T0:20:00.0")
    >>>> print([grid.time for grid in grids])
    [600.0, 1200.0]
    >>>>
