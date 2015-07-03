Installation
============

Dependencies
------------

+ Python 2.6+ and 3.3+
+ numpy (required... for everything)
+ h5py (optional, if reading hdf5 files)
+ matplotlib (optional, if you import viscid.plot.mpl)
+ mayavi2 (optional, if you import viscid.plot.mvi)
+ numexpr (optional, for the calculator.necalc module)
+ cython > 0.17 (optional, only if you want to edit the cython code)
+ PyYaml (optional, rc file and plot options can parse using yaml)

The optional calculator modules (necalc and cycalc) are all dispatched through
calculator.calc, and it is intelligent enough not to use a library that is not
installed.

To get the dependancies squared away, I recommend using the `anaconda <https://store.continuum.io/cshop/anaconda/>`_ python distribution. It makes installing new python libraries almost enjoyable.

Standard Setup
--------------

The jrrle and fortbin readers depend on compiled Fortran code, and the interpolation and streamline functions depend on compiled Cython (C) code. To build Viscid, I recommend running::

  ./setup.py build_ext -i

and adding the `Viscid` directory to your `PYTHONPATH` and `Viscid/scripts` to your `PATH`. This makes editing Viscid far easier.

However the standard distutils commands also work if you're so inclined::

    ./setup.py build
    ./setup.py install
