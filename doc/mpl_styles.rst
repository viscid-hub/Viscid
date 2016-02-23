Matplotlib Styles
=================

As of Matplotlib 1.5, styling plots is highly recomended, and very easy to do as `explained here <http://matplotlib.org/users/style_sheets.html>`_. One can browse the available style sheets `here <https://github.com/matplotlib/matplotlib/tree/master/lib/matplotlib/mpl-data/stylelib>`_. The collection of seaborn styles produce very nice plots; I recommend the `seaborn-talk` style. A default set of style sheets can be loaded by default using a viscidrc file as explained in :doc:`custom_behavior`.

Additionally, Viscid provides some style sheets that can be found in :py:mod:`viscid.plot.mpl_style`.

viscid-colorblind
-----------------

Provides an alternate color cycle that can be seen by those with red-green colorblindness,

.. plot::

    from viscid.plot import mpl
    with mpl.plt.style.context(('viscid-colorblind')):
        mpl.show_colorcycle()
        mpl.plt.title("Default color cycle")
        mpl.auto_adjust_subplots()

        mpl.show_cmap()
        mpl.plt.title("Default non-sequential cmap")
        mpl.auto_adjust_subplots()

        mpl.show_cmap(mpl.matplotlib.rcParams['viscid.symmetric_cmap'])
        mpl.plt.title("Default sequential cmap")
        mpl.auto_adjust_subplots()

viscid-steve
------------

Use Steve's colorbar tick formatter.
