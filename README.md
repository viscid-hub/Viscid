# Viscid #

Python framework to visualize scientific data on structured meshes. At the moment,
only rectilinear meshes defined in the xdmf format are supported, and support for
other mesh types will be added as needed.

File types:
+ XDMF + HDF5
+ ASCII (coming eventually)

## Install ##

Dependencies:
+ numpy (required... for everything)
+ lxml (required, for xdmf support)
+ h5py (required, for reading hdf5 files)
+ matplotlib (optional, if you import viscid.plot.mpl)
+ mayavi2 (optional, if you import viscid.plot.mvi)
+ numexpr (optional, for the calculator.necalc module)
+ cython > 0.17 (optional for calculator.cycalc module; in the future, I may
          check in the .c files so cycalc does not need cython to be available)

The optional calculator modules (necalc and cycalc) are all dispatched through
calculator.calc, and it is intelligent enough not to use a library that is not
installed.

Standard distutils
```./setup.py build
./setup.py install```

For a better dev experience, I recommend adding viscid/viscid to PYTHONPATH,
viscid/scripts to PATH, and building in-place with
```./setup.py build_ext -i```

## Developer Notes ##

I'm trying out the git flow model for this project, so the latest goodies
will be in the develop branch.
