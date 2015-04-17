OpenGGCM Examples
=================

GSE coordinates
---------------

This is just one way to customize the OpenGGCM reader. For more, check out :class:`viscid.readers.openggcm.GGCMGrid`.

.. plot::
    :include-source:

    import viscid
    from viscid.readers import openggcm
    from viscid.plot import mpl

    openggcm.GGCMGrid.mhd_to_gse_on_read = 'auto'

    f3d = viscid.load_file(_viscid_root + '/../../sample/sample.3df.xdmf')
    pp = f3d["pp"]["x=-20.0:20.0,y=0.0,z=-10.0:10.0"]
    mpl.plot(pp, plot_opts="log,x_-30_15", earth=True)
    mpl.plt.title(pp.format_time("UT"))

Time Series
-----------
.. plot::
    :include-source:

    import numpy as np
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates

    import viscid
    from viscid.readers import openggcm

    openggcm.GGCMGrid.mhd_to_gse_on_read = 'auto'

    f3d = viscid.load_file(_viscid_root + '/../../sample/sample.3df.xdmf')

    ntimes = f3d.nr_times()
    t = [None] * ntimes
    pressure = np.zeros((ntimes,), dtype='f4')

    for i, grid in enumerate(f3d.iter_times()):
        t[i] = grid.time_as_datetime()
        pressure[i] = grid['pp']['x=10.0, y=0.0, z=0.0']
    plt.plot(t, pressure)
    plt.ylabel('Pressure')

    dateFmt = mdates.DateFormatter('%H:%M:%S')
    # dateFmt = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
    plt.gca().xaxis.set_major_formatter(dateFmt)
    plt.gcf().autofmt_xdate()
    plt.gca().grid(True)

Ionosphere Files
----------------

.. plot::
    :include-source:

    from matplotlib import pyplot as plt

    import viscid
    from viscid.plot import mpl

    iono_file = viscid.load_file(_viscid_root + '/../../sample/cen2000.iof.xdmf')

    fac_tot = 1e9 * iono_file["fac_tot"]

    plot_args = dict(projection="polar",
                     lin=[-300, 300],
                     bounding_lat=35.0,
                     drawcoastlines=True,  # for basemap only, probably will never be used
                     title="Total FAC\n",  # make a title, or if a string, use the string as title
                     gridec='gray',
                     label_lat=True,
                     label_mlt=True,
                     colorbar=dict(pad=0.1)  # pad the colorbar away from the plot
                    )

    ax1 = plt.subplot(121, projection='polar')
    mpl.plot(fac_tot, ax=ax1, hemisphere='north', **plot_args)
    ax1.annotate('(a)', xy=(0, 0), textcoords="axes fraction",
                 xytext=(-0.1, 1.0), fontsize=18)

    ax2 = plt.subplot(122, projection='polar')
    mpl.plot(fac_tot, ax=ax2, hemisphere="south", style="contourf",
             levels=50, extend="both", **plot_args)
    ax2.annotate('(b)', xy=(0, 0), textcoords="axes fraction",
                 xytext=(-0.1, 1.0), fontsize=18)

    plt.gcf().set_size_inches(10, 5.0)
    mpl.tighten()
