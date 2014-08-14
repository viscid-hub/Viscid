Installation
============

Dependencies
------------

+ Python 2.6+
+ numpy (required... for everything)
+ h5py (required, for reading hdf5 files)
+ matplotlib (optional, if you import viscid.plot.mpl)
+ mayavi2 (optional, if you import viscid.plot.mvi)
+ numexpr (optional, for the calculator.necalc module)
+ cython > 0.17 (optional, only if you want to edit the cython code)

The optional calculator modules (necalc and cycalc) are all dispatched through
calculator.calc, and it is intelligent enough not to use a library that is not
installed.

Standard Setup
--------------

Just call the usual distutils commands. This compiles all the cython code (cython is not required for this).::

    ./setup.py build
    ./setup.py install

For Developers
--------------

For a better dev experience, I recommend adding Viscid to your PYTHONPATH,
viscid/scripts to your PATH, and building in-place with::

    ./setup.py build_ext -i --with-cython

If you want to ensure the cython generated code is up to date, you can use the
shortcut::

    ./setup.py dev

which is the same as above, but it ensures that cython is being used.
