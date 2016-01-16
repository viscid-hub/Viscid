Calculator Examples
===================

Streamlines
-----------

.. plot::
    :include-source:

    import numpy as np

    import viscid
    from viscid.plot import mpl

    B = viscid.vlab.get_dipole(twod=True)
    obound0 = np.array([-4, -4, -4], dtype=B.data.dtype)
    obound1 = np.array([4, 4, 4], dtype=B.data.dtype)
    lines, topo = viscid.calc_streamlines(B,
                                          viscid.Line((0.2, 0.0, 0.0),
                                                      (1.0, 0.0, 0.0), 10),
                                          ds0=0.01, ibound=0.1, maxit=10000,
                                          obound0=obound0, obound1=obound1,
                                          method=viscid.EULER1,
                                          stream_dir=viscid.DIR_BOTH,
                                          output=viscid.OUTPUT_BOTH)
    topo_colors = viscid.topology2color(topo)
    mpl.plot2d_lines(lines, topo_colors, symdir='y')
    mpl.plt.ylim(-0.5, 0.5)

Magnetic Topology
-----------------

Magnetic topology can be determined by the end points of magnetic field lines. By default, the topology output from calc_streamlines is a bitmask of closed (1), open-north (2), open-south (4), and solar wind (8). Any value larger than 8 indicates something else happened with that field line, such as reaching maxit or max_length and ending without hitting a boundary. The topology output can be switched to the raw boundary bitmask by giving ``topo_style='generic'`` as an argument to calc_streamlines.

For more info on streamlines, check out :py:func:`viscid.calc_streamlines` and subclasses of :py:class:`viscid.SeedGen`

X-Point Location and Separator Tracing
--------------------------------------

Bisection algorithm
~~~~~~~~~~~~~~~~~~~

This algorithm takes a 2d map in the uv space of a seed generator and iteratively bisects it into 4 quadrants. If the perimeter of a quadrant contains all 4 global topologies, the bisection continues in that quadrant. This has the advantage of uniquely finding all separator points to a given precision that is determined by the maximum number of iterations. This method is superior in most ways than the bit-or algorithm, but it is less extensively tested.

.. plot::
    :include-source:

    import numpy as np

    import viscid
    from viscid.plot import mpl

    viscid.readers.openggcm.GGCMGrid.mhd_to_gse_on_read = 'auto'

    f3d = viscid.load_file(_viscid_root + '/../../sample/sample_xdmf.3d.xdmf')
    B = f3d['b']['x=-40f:15f, y=-20f:20f, z=-20f:20f']

    # for this method, seeds must be a SeedGen subclass, not a field subset
    seeds = viscid.Volume(xl=(-40, 0, -20), xh=(15, 0, 20), n=(64, 1, 128))

    trace_opts = dict(ibound=2.5)
    xpts = viscid.get_sep_pts_bisect(B, seeds, trace_opts=trace_opts)

    # all done, now just make some illustration
    log_bmag = np.log(viscid.magnitude(B))
    lines, topo = viscid.calc_streamlines(B, seeds, **trace_opts)

    mpl.plot(topo, cmap='afmhot')

    mpl.plot2d_lines(lines[::79], scalars=log_bmag, symdir='y')
    mpl.plt.plot(xpts[0], xpts[2], 'y*', ms=20,
                 markeredgecolor='k', markeredgewidth=1.0)
    mpl.plt.xlim(topo.xl[0], topo.xh[0])
    mpl.plt.ylim(topo.xl[2], topo.xh[2])

    # since seeds is a Field, we can use it to determine mhd|gse
    mpl.plot_earth(B['y=0f'])


Bit-or algorithm
~~~~~~~~~~~~~~~~

This algorithm takes a 2d map in the uv space of a seed generator and performs an iterative bitwise-or on neighbors until the intersection of all four global topologies meet. This algorithm is simple, but fragile. The more extended the reconnection region, the worse this algorithm will work. Also, do to the need for a clustering step, it is impossible to determine the accuracy of this method a priori.

.. plot::
    :include-source:

    import numpy as np

    import viscid
    from viscid.plot import mpl

    viscid.readers.openggcm.GGCMGrid.mhd_to_gse_on_read = 'auto'

    f3d = viscid.load_file(_viscid_root + '/../../sample/sample_xdmf.3d.xdmf')
    B = f3d['b']['x=-40f:15f, y=-20f:20f, z=-20f:20f']

    # Fields can be used as seeds to get one seed per grid point
    seeds = B.slice_keep('y=0f')
    lines, topo = viscid.calc_streamlines(B, seeds, ibound=2.5,
                                          output=viscid.OUTPUT_BOTH)
    xpts_night = viscid.topology_bitor_clusters(topo['x=:0f, y=0f'])

    # The dayside is done separately here because the sample data is at such
    # low resolution. Super-sampling the grid with the seeds can sometimes help
    # in these cases.
    day_seeds = viscid.Volume((7.0, 0.0, -5.0), (12.0, 0.0, 5.0), (16, 1, 16))
    _, day_topo = viscid.calc_streamlines(B, day_seeds, ibound=2.5,
                                          output=viscid.OUTPUT_TOPOLOGY)
    xpts_day = viscid.topology_bitor_clusters(day_topo)

    log_bmag = np.log(viscid.magnitude(B))

    clim = (min(np.min(day_topo), np.min(topo)),
            max(np.max(day_topo), np.max(topo)))
    mpl.plot(topo, cmap='afmhot', clim=clim)
    mpl.plot(day_topo, cmap='afmhot', clim=clim, colorbar=None)

    mpl.plot2d_lines(lines[::79], scalars=log_bmag, symdir='y')
    mpl.plt.plot(xpts_night[0], xpts_night[1], 'y*', ms=20,
                 markeredgecolor='k', markeredgewidth=1.0)
    mpl.plt.plot(xpts_day[0], xpts_day[1], 'y*', ms=20,
                 markeredgecolor='k', markeredgewidth=1.0)
    mpl.plt.xlim(topo.xl[0], topo.xh[0])
    mpl.plt.ylim(topo.xl[2], topo.xh[2])

    # since seeds is a Field, we can use it to determine mhd|gse
    mpl.plot_earth(seeds.slice_reduce(":"))
