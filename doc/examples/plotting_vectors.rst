Plotting Vector Quantities
==========================

Various shims for plotting fields using matplotlib are provided by :py:mod:`viscid.plot.mpl`. The functions are listed in :doc:`../functions`. For an enumeration of extra plotting keyword arguments, see :doc:`../plot_options`. Also, you can specify keyword arguments that are conusmed by the matplotlib function used to make the plot, i.e., `pyplot.streamline`, `pyplot.quiver`, etc.

Streamplot
----------

Plotting 2D streamlines. Note that using matplotlib to make streamlines from a 2D slice may make it appear that Div B ≠ 0, but that's just an illusion. If this is a problem, you can always resort to full 3D streamlines, as in :doc:`stream_and_interp`.

.. plot::
    :include-source:

    import viscid
    from viscid.plot import mpl

    f3d = viscid.load_file(_viscid_root + '/../../sample/sample_xdmf.3d.xdmf')

    mpl.plot(f3d['Pressure = pp']['z=0f'], logscale=True, earth=True)
    mpl.streamplot(f3d['v']['z=0f'], arrowsize=2, density=2)

    mpl.plt.xlim(-20, 20)
    mpl.plt.ylim(-10, 10)

Quivers
-------

Quivers are another way to show vector fields, but in many cases, it will be useful to downsample the field, or else there will be far too many arrows.

.. plot::
    :include-source:

    import viscid
    from viscid.plot import mpl

    f3d = viscid.load_file(_viscid_root + '/../../sample/sample_xdmf.3d.xdmf')

    mpl.plot(f3d['pp']['z=0f'], cbarlabel="Pressure", logscale=True, earth=True)
    mpl.plot2d_quiver(f3d['v']['x=::2, y=::2, z=0f'])

    mpl.plt.xlim(-20, 20)
    mpl.plt.ylim(-10, 10)

Uniform Quivers
---------------

If your data has a very nonuniform grid, it may be useful to interpolate your data onto a uniform grid first.

.. plot::
    :include-source:

    import viscid
    from viscid.plot import mpl

    f3d = viscid.load_file(_viscid_root + '/../../sample/sample_xdmf.3d.xdmf')

    mpl.plot(f3d['Pressure = pp']['z=0f'], logscale=True, earth=True)
    new_grid = viscid.Volume((-20, -20, 0), (20, 20, 0), n=(16, 16, 1))
    v = viscid.interp_trilin(f3d['v'], new_grid)
    mpl.plot2d_quiver(v['z=0f'])

    mpl.plt.xlim(-20, 20)
    mpl.plt.ylim(-10, 10)
