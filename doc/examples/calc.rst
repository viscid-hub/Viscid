Calculator Examples
===================

Magnetic Topology
-----------------

Magnetic topology can be determined by the end points of magnetic field lines. By default, the topology output from calc_streamlines is a bitmask of closed (1), open-north (2), open-south (4), and solar wind (8). Any value larger than 8 indicates something else happened with that field line, such as reaching maxit or max_length and ending without hitting a boundary. The topology output can be switched to the raw boundary bitmask by giving ``topo_style='generic'`` as an argument to calc_streamlines.

For more info on streamlines, check out :py:func:`viscid.calc_streamlines` and subclasses of :py:class:`viscid.SeedGen`

.. plot::
    :include-source:

    import numpy as np

    import viscid
    from viscid.plot import mpl

    viscid.readers.openggcm.GGCMGrid.mhd_to_gse_on_read = 'auto'

    f3d = viscid.load_file(_viscid_root + '/../../sample/sample_xdmf.3d.xdmf')
    B = f3d['b']['x=-40f:15f, y=-20f:20f, z=-20f:20f']

    # # these seeds can be made a little more sparse than the actual grid if
    # # speed is more important than resolution
    # seeds = viscid.Volume((-30.0, 0.0, -10.0), (+10.0, 0.0, +10.0), (64, 1, 32))
    seeds = B.slice_and_keep('y=0f')
    lines, topo = viscid.calc_streamlines(B, seeds, ibound=2.5,
                                          output=viscid.OUTPUT_BOTH)

    # note that this requires so many iterations because the sample data is
    # at such low resolution and the dayside current sheet is so extended
    xpts_night = viscid.find_sep_points_cartesian(topo['x=:0f, y=0f'])

    # bz = B['x, y=0f']
    log_bmag = np.log(viscid.magnitude(B))

    mpl.plot(topo, cmap='afmhot')
    mpl.plot2d_lines(lines[::79], scalars=log_bmag, symdir='y')
    mpl.plt.plot(xpts_night[0], xpts_night[1], 'y*', ms=20,
                 markeredgecolor='k', markeredgewidth=1.0)
    mpl.plt.xlim(topo.xl[0], topo.xh[0])
    mpl.plt.ylim(topo.xl[2], topo.xh[2])


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
                                                      (1.0, 0.0, 0.0),
                                                      10),
                                          ds0=0.01, ibound=0.1, maxit=10000,
                                          obound0=obound0, obound1=obound1,
                                          method=viscid.EULER1,
                                          stream_dir=viscid.DIR_BOTH,
                                          output=viscid.OUTPUT_BOTH)
    topo_colors = viscid.topology2color(topo)
    mpl.plot2d_lines(lines, topo_colors, symdir='y')
    mpl.plt.ylim(-0.5, 0.5)
