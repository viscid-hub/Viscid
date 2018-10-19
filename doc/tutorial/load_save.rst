Loading Datasets
================

Automatic Type Discovery
------------------------

In most cases, Viscid can automatically detect the file type based on the filename. For these cases, loading a file is as easy as,

.. code-block:: python

   import os

   import viscid

   f = viscid.load_file(os.path.join(viscid.sample_dir, 'sample_xdmf.py_0.xdmf'))

Specifying File Type
--------------------

You can manually specify the file type using the ``file_type`` keyword argument to ``viscid.load_file``

.. code-block:: python

   import os

   import viscid

   f = viscid.load_file(os.path.join(viscid.sample_dir, 'sample_xdmf.py_0.xdmf'),
                        file_type=viscid.readers.ggcm_xdmf.GGCMFileXDMF)

You can also specify the class name as a string,

.. code-block:: python

   import os

   import viscid

   f = viscid.load_file(os.path.join(viscid.sample_dir, 'sample_xdmf.py_0.xdmf'),
                        file_type='ggcm-xdmf')

Saving Datasets
===============

HDF5 + XDMF
-----------

Grids (or lists of Fields) can be saved to an HDF5 + XDMF pair of files. Setting ``complevel`` greater than 0 enables gzip compression.

.. code-block:: python

   import os

   import viscid

   f = viscid.load_file(os.path.join(viscid.sample_dir, 'sample_xdmf.py_0.xdmf'))

   grid = f.get_grid(time=0)
   viscid.save_grid('example.h5', grid, complevel=9)

   # or, use ``viscid.save_fields('example.h5', [f['bx'], f['by']])``

   grid2 = viscid.load_file('example.xdmf')
   grid2.print_tree()

Numpy File
----------

Grids (or lists of Fields) can be saved to Numpy's binary data files.

.. code-block:: python

   import os

   import viscid

   f = viscid.load_file(os.path.join(viscid.sample_dir, 'sample_xdmf.py_0.xdmf'))
   viscid.save_grid('example.npz', f.get_grid())

   # or, use ``viscid.save_fields('example.npz', [f['bx'], f['by']])``

   f2 = viscid.load_file('example.npz')
   f2.print_tree()

Pandas
------

Datasets can be converted to / from Pandas dataframes. This enables the use of Pandas' extensive IO tools. The drawback to using pandas is that IO is not lazy. **Pandas IO should not be used if the whole file cannot comfortably fit in RAM**. Note that using Pandas to write HDF5 files requires PyTables to be installed.

.. code-block:: python

   import os

   import pandas as pd
   import viscid

   f = viscid.load_file(os.path.join(viscid.sample_dir, 'sample_xdmf.py_0.xdmf'))
   dataframe = f.to_dataframe()
   dataframe.to_hdf('example.h5', 'key', complevel=9)

   f2 = viscid.from_dataframe(pd.read_hdf('example.h5'))
   f2.print_tree()
