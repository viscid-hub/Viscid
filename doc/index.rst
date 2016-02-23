.. Viscid documentation master file, created by
   sphinx-quickstart on Tue Jul  2 01:44:29 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Viscid's documentation!
==================================

Viscid is a python framework to visualize scientific data on structured meshes. The following file types are understood,

+ XDMF + HDF5
+ OpenGGCM jrrle (3df, p[xyz], iof)
+ OpenGGCM binary (3df, p[xyz], iof)
+ Athena (bin, hst, tab)
+ ASCII

There is also preliminary support for reading and plotting AMR datasets from XDMF files.


Below are some simple examples to get you started with Viscid. There are some far more interesting examples in the ``Viscid/tests`` directory, but they are not always as straight forward or documented as the examples here.

Quick Start
-----------

I recommend using the `anaconda <https://store.continuum.io/cshop/anaconda/>`_ python distribution. It makes installing new python libraries almost enjoyable. Check :doc:`installation` for the list of dependancies. To build Viscid, I recommend running::

  ./setup.py build_ext -i
  viscid_dir=$(pwd)
  export PYTHONPATH=$PYTHONPATH:${viscid_dir}
  export PATH=$PATH:${viscid_dir}/scripts

Contents
--------

.. toctree::
  :maxdepth: 1

  ChangeLog<changes>

.. toctree::
  :maxdepth: 3

  installation
  philosophy
  functions
  plot_options
  mpl_styles
  custom_behavior
  command_line
  dev_guide
  extending_readers

  examples/index

  api/viscid

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
