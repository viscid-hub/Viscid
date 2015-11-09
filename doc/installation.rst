Installation
============

Dependencies
------------

+ Required

  + Python 2.6+ or 3.3+
  + Numpy

+ Highly Recommended

  + H5py (if reading hdf5 files)
  + Matplotlib (if you want to make 2d plots using viscid.plot.mpl)
  + Numexpr (for the calculator.necalc module)

+ Truly Optional

  + Seaborn
  + Mayavi2 (if you want to make 3d plots using viscid.plot.mvi)
  + PyYaml (rc file and plot options can parse using yaml)

+ Required only for developers

  + Cython > 0.17 (if you change pyx / pxd files)
  + Sphinx
  + sphinx_rtd_theme
  + sphinxcontrib-napoleon (if Sphinx is <= version 1.2)

The optional calculator modules (necalc and cycalc) are all dispatched through
calculator.calc, and it is intelligent enough not to use a library that is not
installed.

To get the dependancies squared away, I recommend using the `anaconda <https://store.continuum.io/cshop/anaconda/>`_ python distribution. It makes installing new python libraries almost enjoyable.

Standard Setup
--------------

The jrrle and fortbin readers depend on compiled Fortran code, and the interpolation and streamline functions depend on compiled Cython (C) code. To build Viscid, I recommend running::

    ./setup.py build_ext -i
    viscid_dir=$(pwd)
    export PYTHONPATH=$PYTHONPATH:${viscid_dir}
    export PATH=$PATH:${viscid_dir}/scripts

and adding the `Viscid` directory to your `PYTHONPATH` and `Viscid/scripts` to your `PATH`. This makes editing Viscid far easier.

However, the standard distutils commands also work if you're so inclined::

    ./setup.py build
    ./setup.py install
