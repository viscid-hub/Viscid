Calculator Examples
===================

Streamlines
-----------

For more info on streamlines, check out :py:func:`viscid.calculator.streamline.streamlines` and :py:mod:`viscid.calculator.seed`

.. plot::
    :include-source:

    import numpy as np

    import viscid
    from viscid.plot import mpl

    B = viscid.vlab.get_dipole(twod=True)
    obound0 = np.array([-4, -4, -4], dtype=B.data.dtype)
    obound1 = np.array([4, 4, 4], dtype=B.data.dtype)
    lines, topo = viscid.calc_streamlines(B,
                                          viscid.Line((0.0, 0.0, 0.2),
                                                      (0.0, 0.0, 1.0),
                                                      10),
                                          ds0=0.01, ibound=0.1, maxit=10000,
                                          obound0=obound0, obound1=obound1,
                                          method=viscid.EULER1,
                                          stream_dir=viscid.DIR_BOTH,
                                          output=viscid.OUTPUT_BOTH)
    topo_colors = viscid.topology2color(topo)
    mpl.plot2d_lines(lines, topo_colors, symdir='y')
