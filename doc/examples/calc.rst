Calculator Examples
===================

Streamlines
-----------

For more info on streamlines, check out :py:func:`viscid.calculator.streamline.streamlines` and :py:mod:`viscid.calculator.seed`

.. plot::
    :include-source:

    import numpy as np

    from viscid import vlab
    from viscid.plot import mpl
    from viscid.calculator import seed
    from viscid.calculator import streamline

    B = vlab.get_dipole(twod=True)
    obound0 = np.array([-4, -4, -4], dtype=B.data.dtype)
    obound1 = np.array([4, 4, 4], dtype=B.data.dtype)
    lines, topo = streamline.streamlines(B,
                                         seed.Line((0.0, 0.0, 0.2),
                                                   (0.0, 0.0, 1.0),
                                                   10),
                                         ds0=0.01, ibound=0.1, maxit=10000,
                                         obound0=obound0, obound1=obound1,
                                         method=streamline.EULER1,
                                         stream_dir=streamline.DIR_BOTH,
                                         output=streamline.OUTPUT_BOTH)
    mpl.plot_streamlines2d(lines, 'y', topo)
