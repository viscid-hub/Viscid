# Viscid #

Python framework to visualize gridded scientific data. Also 

File types:
+ XDMF + HDF5
+ ASCII (coming eventually)

## Install ##

Dependancies:
+ numpy (required)
+ lxml (required, for xdmf support)
+ h5py (required, for reading hdf5 files)
+ matplotlib (optional, if you import viscid.plot.mpl)
+ mayavi2 (optional, if you import viscid.plot.mvi)
+ numexpr (optional, for the calculator.calc module)
+ cython (optional for development, otherwise the .c files are already checked in)

Standard distutils
```./setup.py build
./setup.py install``` 

Far a better dev experience, I recommend adding source dir to PYTHONPATH and:
```./setup.py build_ext -i --with-cython```

