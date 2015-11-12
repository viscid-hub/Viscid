# Viscid #

Python framework to visualize scientific data on structured meshes. At the moment,
only rectilinear meshes are supported, and support for other mesh types will be added
as needed.

File types:
+ XDMF + HDF5
+ OpenGGCM jrrle (3df, p[xyz], iof)
+ OpenGGCM binary (3df, p[xyz], iof)
+ Athena (bin, hst, tab)
+ ASCII

There is also preliminary support for reading and plotting AMR datasets from XDMF files.

## Documentation ##

Both the master and dev branches should make every attempt to be usable (thanks to continuous integration), but the obvious caveats exist, i.e. the dev branch has more cool new features but it isn't _as_ tested.

Branch        | Docs                                                                      | Test Status
------------- | ------------------------------------------------------------------------- | -----------------------
master        | [html](http://kristoformaynard.github.io/Viscid/docs/master/index.html), [test summary](http://kristoformaynard.github.io/Viscid/summary/master-2.7-full/index.html)   | [![Build Status](https://travis-ci.org/KristoforMaynard/Viscid.svg?branch=master)](https://travis-ci.org/KristoforMaynard/Viscid)
dev           | [html](http://kristoformaynard.github.io/Viscid/docs/dev/index.html), [test summary](http://kristoformaynard.github.io/Viscid/summary/dev-2.7-full/index.html)      | [![Build Status](https://travis-ci.org/KristoforMaynard/Viscid.svg?branch=dev)](https://travis-ci.org/KristoforMaynard/Viscid)

## Install ##

Dependencies:

+ Required
  + Python 2.7+ or 3.3+
  + Python 2.6 + argparse
  + Numpy
+ Highly Recommended
  + H5py (if reading hdf5 files)
  + Matplotlib (if you want to make 2d plots using viscid.plot.mpl)
  + Numexpr (for the calculator.necalc module)
+ Truly Optional
  + Seaborn
  + Mayavi2 (if you want to make 3d plots using viscid.plot.mvi)
  + PyYaml (rc file and plot options can parse using yaml)
+ Optional for developers
  + Cython > 0.17 (if you change pyx / pxd files)
  + Sphinx
  + sphinx_rtd_theme
  + sphinxcontrib-napoleon (if Sphinx is <= version 1.2)

The optional calculator modules (necalc and cycalc) are all dispatched through
calculator.calc, and it is intelligent enough not to use a library that is not
installed.

The jrrle and fortbin readers depend on compiled Fortran code, and the interpolation and streamline functions depend on compiled Cython (C) code. To build Viscid, I recommend running::

    ./setup.py build_ext -i
    viscid_dir=$(pwd)
    export PYTHONPATH=$PYTHONPATH:${viscid_dir}
    export PATH=$PATH:${viscid_dir}/scripts

and adding the `Viscid` directory to your `PYTHONPATH` and `Viscid/scripts` to your `PATH`. This makes editing Viscid far easier.

However, the standard distutils commands also work if you're so inclined::

    ./setup.py build
    ./setup.py install

## Development ##

Please, if you edit the code, use [PEP 8](https://www.python.org/dev/peps/pep-0008/) style. Poor style is more than just aesthetic; it tends to lead to bugs that are difficult to spot. This is sorely true when it comes to whitespace (4 spaces per indent please).
