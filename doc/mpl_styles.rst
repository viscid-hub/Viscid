Matplotlib Styles
=================

As of Matplotlib 1.5, styling plots is highly recomended, and very easy to do as `explained here <http://matplotlib.org/users/style_sheets.html>`_. One can browse the available style sheets `here <https://github.com/matplotlib/matplotlib/tree/master/lib/matplotlib/mpl-data/stylelib>`_. The collection of seaborn styles produce very nice plots; I recommend the `seaborn-talk` style. A default set of style sheets can be loaded by default using a viscidrc file as explained in :doc:`custom_behavior`.

Additionally, Viscid provides some style sheets that can be found in :py:mod:`viscid.plot.mpl_style`.

Viscid Styles
-------------

viscid-default
~~~~~~~~~~~~~~

By default, Viscid loads with the viridis color map for generic color plots, or the RdBu_r color map for plots symmetric around 0. Even though viscid-default is always loaded, you may need to put it later in the list of default styles since some of the styles set `image.cmap`, and precidence goes to the most recently applied styles.

.. plot::

    from viscid.plot import mpl

    with mpl.plt.style.context(('viscid-default')):
        mpl.show_cmap()
        mpl.plt.title("Default non-symmetric cmap")
        mpl.auto_adjust_subplots()

        mpl.show_cmap(mpl.matplotlib.rcParams['viscid.symmetric_cmap'])
        mpl.plt.title("Default symmetric cmap")
        mpl.auto_adjust_subplots()

viscid-colorblind
~~~~~~~~~~~~~~~~~

This style sets the following color palette as the colorcycle,

.. plot::

    from viscid.plot import mpl

    with mpl.plt.style.context(('viscid-colorblind')):
        mpl.show_colorcycle()
        mpl.plt.title("Default color cycle")
        mpl.auto_adjust_subplots()

viscid-steve
~~~~~~~~~~~~

Use Steve's colorbar tick formatter. This formatter does proper scientific notation , i.e., $a \times 10^b$, and has a more sane way of deciding what numbers should be in scientific notation.
