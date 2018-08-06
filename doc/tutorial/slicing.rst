Slicing Fields
==============

See :doc:`../indexing` for more information on field indexing.

Spatial Slices
--------------

Fields can be sliced just like Numpy ndarrays, but you can also use an extended syntax to ask for a physical location instead of an index in the array. This extended syntax is to provide a string with a number followed by an 'f' in place of an integer that is usually found in slices. For even further convenience, you can specify the entire slice as a string and specify the slice dimension by component name.

The best way to show the utility of this extended syntax is by example, so here goes. This example snips off the fist 5 cells and last 25 cells in x and extracts the :code:`y = 0.0` plane. In the z-direction, the field is sliced every other grid cell between :code:`z = -8` and :code:`z = 10.0`. Here, x, y, and z are coordinate names, so in ionosphere grids, one would use 'lat' and 'lon'.

.. plot::
    :include-source:

    from os import path

    import viscid
    from viscid.plot import vpyplot as vlt


    f3d = viscid.load_file(path.join(viscid.sample_dir, 'sample_xdmf.3d.xdmf'))

    # snip off the first 5 and last 25 cells in x, and grab every other cell
    # in z between z = -8.0 and z = 10.0 (in space, not index).
    # Notice that slices by location are done by appending an 'f' to the
    # slice. This means "y=0" is not the same as "y=0j".
    pp = f3d["pp"]["x = 5:-25, y = 0.0j, z = -8.0j:10.0j:2"]
    vlt.plot(pp, style="contourf", levels=50, plot_opts="log,earth")

    vlt.show()

Temporal Slices
---------------

Temporal Datasets can be sliced a number of ways too, here are some examples

.. doctest::

    >>> from os import path
    >>>
    >>> import viscid
    >>>
    >>>
    >>> f3d = viscid.load_file(path.join(viscid.sample_dir, 'sample_xdmf.3d.xdmf'))
    >>>
    >>> grids = f3d.get_times(slice("UT1967:01:01:00:10:00.0",
    >>>                             "UT1967:01:01:00:20:00.0"))
    >>> print([grid.time for grid in grids])
    [600.0, 1200.0]
    >>>
    >>> grids = f3d.get_times(1)
    >>> print([grid.time for grid in grids])
    [600.0]
    >>>
    >>> grids = f3d.get_times(slice(None, 600.0))
    >>> print([grid.time for grid in grids])
    [600.0]
    >>> grids = f3d.get_times("T0:10:00.0:T0:20:00.0")
    >>> print([grid.time for grid in grids])
    [600.0, 1200.0]

Single Time Slice
~~~~~~~~~~~~~~~~~

.. plot::
    :include-source:

    from os import path

    import viscid
    from viscid.plot import vpyplot as vlt
    from matplotlib import pyplot as plt


    f3d = viscid.load_file(path.join(viscid.sample_dir, 'sample_xdmf.3d.xdmf'))

    _, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    f3d.activate_time(0)

    # notice y=0.0, this is different from y=0; y=0 is the 0th index in
    # y, which is this case will be y=-50.0
    vlt.plot(f3d["vz"]["x = -20.0j:20.0j, y = 0.0j, z = -10.0j:10.0j"],
             style="contourf", levels=50, plot_opts="lin_0,earth", ax=axes[0])
    plt.title(f3d.get_grid().format_time("UT"))

    # share axes so this plot pans/zooms with the first
    f3d.activate_time(-1)
    vlt.plot(f3d["vz"]["x = -20.0j:20.0j, y = 0.0j, z = -10.0j:10.0j"],
             style="contourf", levels=50, plot_opts="lin_0,earth", ax=axes[1])
    plt.title(f3d.get_grid().format_time("hms"))

    vlt.auto_adjust_subplots()
    vlt.show()

Iterating Over Time Slices
~~~~~~~~~~~~~~~~~~~~~~~~~~

Or, if you need to iterate over all time slices, you can do that too. The advantage of using the iterator here is that it's smart enough to kick the old time slice out of memory when you move to the next time.

.. plot::
    :include-source:

    from os import path

    import numpy as np
    import viscid
    from viscid.plot import vpyplot as vlt
    from matplotlib import pyplot as plt


    f2d = viscid.load_file(path.join(viscid.sample_dir, 'sample_xdmf.py_0.xdmf'))

    times = np.array([grid.time for grid in f2d.iter_times(":2")])
    nr_times = len(times)

    _, axes = plt.subplots(nr_times, 1)

    for i, grid in enumerate(f2d.iter_times(":2")):
        vlt.plot(grid["vz"]["x = -20.0j:20.0j, y = 0.0j, z = -10.0j:10.0j"],
                 plot_opts="lin_0,earth", ax=axes[i])
        plt.title(grid.format_time(".01f"))

    vlt.auto_adjust_subplots()
    vlt.show()
