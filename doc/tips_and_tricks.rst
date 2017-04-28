Tips & Tricks
=============

.. contents::
  :local:

Loading Files
-------------

Loading a Glob as a Single File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multiple files that make up a single time series can be loaded as a single file, here's how,

.. code-block:: python

    >>> import os
    >>> import viscid

    >>> os.listdir()
    ['run.3df.005', 'run.3df.010', 'run.3df.015']

    >>> f = viscid.load_file("run.3df.*")
    >>> [grid.time for grid in f.iter_times()]
    [5.0, 10.0, 15.0]


Loading a Slice of a Time Series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes you might have a single XDMF file that points to multiple XDMF files to make a time series. But, if you are loading a long time series, this might take a while just to parse XDMF files. Who has the time. To this end, you can load a slice of the time series by either index or value using the same conventions as slicing fields (bare int to slice by index, or number followed by 'f' to slice by location). This works using :py:func:`viscid.glob2`. Here are a couple examples,

.. code-block:: python

    >>> import os
    >>> import viscid

    >>> os.listdir()
    ['run.3d.005.xdmf', 'run.3d.010.xdmf', 'run.3d.015.xdmf', 'run.3d.020.xdmf']

    >>> # lets just load the first two files (by index)
    >>> f = viscid.load_file("*.3d.[:2].xdmf")
    >>> [grid.time for grid in f.iter_times()]
    [5.0, 10.0]

    >>> # what about just the tail...
    >>> f = viscid.load_file("*.3d.[-2:].xdmf")
    >>> [grid.time for grid in f.iter_times()]
    [15.0, 20.0]

    >>> # every other file?
    >>> f = viscid.load_file("*.3d.[::2].xdmf")
    >>> [grid.time for grid in f.iter_times()]
    [5.0, 15.0]

    >>> # let's load files after and including t = 10
    >>> f = viscid.load_file("*.3d.[10f:].xdmf")
    >>> [grid.time for grid in f.iter_times()]
    [10.0, 15.0, 20.0]

    >>> # every other file after and including t = 10
    >>> f = viscid.load_file("*.3d.[10f::2].xdmf")
    >>> [grid.time for grid in f.iter_times()]
    [10.0, 20.0]


Reloading Files
~~~~~~~~~~~~~~~

When debugging a simulation, it's sometimes convenient to keep Viscid open interactively in either IPython or a Jupyter Notebook. Unfortunately, Viscid doesn't appreciate if files are modified after they've been loaded. To fix this, you can just use,

.. code-block:: python

    viscid.load_file("...", force_reload=True)

If forcing a reload doesn't help, then you can always hammer-of-Thor the whole cache,

.. code-block:: python

    viscid.unload_all_files()
