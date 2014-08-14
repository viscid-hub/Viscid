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

    openggcm.GGCMGrid.mhd_to_gse_on_read = True

    f3d = viscid.load_file(_viscid_root + '/../sample/sample.3df.xdmf')
    pp = f3d["pp"]["x=-20.0:20.0,y=0.0,z=-10.0:10.0"]
    mpl.plot(pp, plot_opts="log,x_-30_15", earth=True)

Ionosphere Files
----------------

.. plot::
    :include-source:

    from matplotlib import pyplot as plt

    import viscid
    from viscid.plot import mpl

    iono_file = viscid.load_file(_viscid_root + '/../sample/cen2000.iof.xdmf')

    fac_tot = 1e9 * iono_file["fac_tot"]

    ax1 = plt.subplot(121)
    mpl.plot(fac_tot, ax=ax1, hemisphere="north", style="contourf",
             plot_opts="lin_-300_300", extend="both",
             levels=50, drawcoastlines=True)
    ax2 = plt.subplot(122)
    mpl.plot(fac_tot, ax=ax2, hemisphere="south", style="contourf",
             plot_opts="lin_-300_300", extend="both",
             levels=50, drawcoastlines=True)
    plt.gcf().set_size_inches(12, 4.5)
