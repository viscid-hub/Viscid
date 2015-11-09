Reading/Plotting Examples
=========================

The basic idea of Viscid is to convert data files to :class:`viscid.field.Field` objects and pass them to a convenience function for plotting. It is useful to note that Field objects have most of the same functionalty as Numpy's `ndarrays`. That is, you can say::

    field_momentum = field_density * field_velocity

to do simple math using Numpy.

Super Simple
------------

Here, we just open up an OpenGGCM xdmf file and plot the plasma pressure with a log scale. For a better idea of the different plotting options, refer to :meth:`viscid.plot.mpl.plot2d_field`.

.. plot::
    :include-source:

    import viscid
    from viscid.plot import mpl

    f3d = viscid.load_file(_viscid_root + '/../../sample/sample.3df.xdmf')
    # notice y=0.0, this is different from y=0; y=0 is the 0th index in
    # y, which is this case will be y=-50.0
    pp = f3d["pp"]["y=0.0f"]
    mpl.plot(pp, plot_opts="log")

Two Plots, One Figure
---------------------

.. plot::
    :include-source:

    from matplotlib import pyplot as plt

    import viscid
    from viscid.plot import mpl

    f3d = viscid.load_file(_viscid_root + '/../../sample/sample.3df.xdmf')

    ax1 = plt.subplot2grid((2, 1), (0, 0))

    # notice y=0.0, this is different from y=0; y=0 is the 0th index in
    # y, which is this case will be y=-50.0
    mpl.plot(f3d["pp"]["y=0.0f"], style="contourf", levels=20,
             plot_opts="log,earth")

    # share axes so this plot pans/zooms with the first
    plt.subplot2grid((2, 1), (1, 0), sharex=ax1, sharey=ax1)
    mpl.plot(f3d["vx"]["y=0.0f"], style="contourf", levels=20,
             plot_opts="earth")

    plt.xlim((-20, 20))
    plt.ylim((-10, 10))

Playing with Time
-----------------

To get access to a specific time slice, just ask the file object.

.. plot::
    :include-source:

    import viscid
    from viscid.plot import mpl

    f3d = viscid.load_file(_viscid_root + '/../../sample/sample.3df.xdmf')

    ax1 = mpl.plt.subplot2grid((2, 1), (0, 0))
    f3d.activate_time(0)

    # notice y=0.0, this is different from y=0; y=0 is the 0th index in
    # y, which is this case will be y=-50.0
    mpl.plot(f3d["vz"]["x = -20.0f:20.0f, y = 0.0f, z = -10.0f:10.0f"],
             style="contourf", levels=50, plot_opts="lin_0,earth")
    mpl.plt.title(f3d.get_grid().format_time("UT"))

    # share axes so this plot pans/zooms with the first
    mpl.plt.subplot2grid((2, 1), (1, 0), sharex=ax1, sharey=ax1)
    f3d.activate_time(-1)
    mpl.plot(f3d["vz"]["x = -20.0f:20.0f, y = 0.0f, z = -10.0f:10.0f"],
             style="contourf", levels=50, plot_opts="lin_0,earth")
    mpl.plt.title(f3d.get_grid().format_time("hms"))
    mpl.tighten()

Or, if you need to iterate over all time slices, you can do that too. The advantage of using the iterator here is that it's smart enough to kick the old time slice out of memory when you move to the next time.

.. plot::
    :include-source:

    import numpy as np
    from matplotlib import pyplot as plt

    import viscid
    from viscid.plot import mpl

    f3d = viscid.load_file(_viscid_root + '/../../sample/sample.3df.xdmf')

    times = np.array([grid.time for grid in f3d.iter_times()])
    nr_times = len(times)

    for i, grid in enumerate(f3d.iter_times()):
        plt.subplot2grid((nr_times, 1), (i, 0))
        mpl.plot(grid["vz"]["x = -20.0f:20.0f, y = 0.0f, z = -10.0f:10.0f"],
                 plot_opts="lin_0,earth")
        mpl.plt.title(grid.format_time(".01f"))
    mpl.tighten()
