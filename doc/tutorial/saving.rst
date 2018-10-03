Saving Datasets
===============

Pandas
------

Datasets can be converted to / from Pandas dataframes. This enables the use of Pandas' extensive IO tools. Note that using Pandas to write HDF5 files requires PyTables to be installed.

.. code-block:: python

   import os

   import pandas as pd
   import viscid

   f = viscid.load_file(os.path.join(viscid.sample_dir, 'sample_xdmf.py_0.xdmf'))
   dataframe = f.to_dataframe()
   dataframe.to_hdf('example.h5', 'key', complevel=9)

   f2 = viscid.from_dataframe(pd.read_hdf('example.h5'))
   f2.print_tree()

Numpy File
----------

Grids (or lists of Fields) can be saved to Numpy's binary data files.

.. code-block:: python

   import os

   import viscid

   f = viscid.load_file(os.path.join(viscid.sample_dir, 'sample_xdmf.py_0.xdmf'))
   viscid.save_grid('example.npz', f.get_grid())

   f2 = viscid.load_file('example.npz')
   f2.print_tree()
