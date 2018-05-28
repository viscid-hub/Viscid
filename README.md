# Viscid #

Python framework to visualize scientific data on structured meshes. At the moment,
only rectilinear meshes are supported, and support for other mesh types will be added as needed.

File types:
+ XDMF + HDF5
+ OpenGGCM jrrle (3df, p[xyz], iof)
+ OpenGGCM binary (3df, p[xyz], iof)
+ Athena (bin, hst, tab)
+ VPIC
+ ASCII

There is also preliminary support for reading and plotting AMR datasets from XDMF files.

## Documentation ##

Both the master and dev branches should make every attempt to be usable (thanks to continuous integration), but the obvious caveats exist, i.e. the dev branch has more cool new features but it isn't _as_ tested.

Branch                                                      | Docs                                                                      | Test Status
------------- | ------------------------------------------------------------------------- | -----------------------
[master](https://github.com/KristoforMaynard/Viscid)        | [html](http://kristoformaynard.github.io/Viscid-docs/docs/master/index.html), [test summary](http://kristoformaynard.github.io/Viscid-docs/summary/master-2.7/index.html)   | [![Build Status](https://travis-ci.org/KristoforMaynard/Viscid.svg?branch=master)](https://travis-ci.org/KristoforMaynard/Viscid)
[dev](https://github.com/KristoforMaynard/Viscid/tree/dev)  | [html](http://kristoformaynard.github.io/Viscid-docs/docs/dev/index.html), [test summary](http://kristoformaynard.github.io/Viscid-docs/summary/dev-2.7/index.html)      | [![Build Status](https://travis-ci.org/KristoforMaynard/Viscid.svg?branch=dev)](https://travis-ci.org/KristoforMaynard/Viscid)

## Install ##

[![Anaconda-Server Badge](https://anaconda.org/kristoformaynard/viscid/badges/version.svg)](https://anaconda.org/kristoformaynard/viscid) [![Anaconda-Server Badge](https://anaconda.org/kristoformaynard/viscid/badges/platforms.svg)](https://anaconda.org/kristoformaynard/viscid)

[![PyPI Version](https://img.shields.io/pypi/v/Viscid.svg)](https://pypi.org/project/Viscid/)

Dependencies:

+ Required
  + Python 2.7+ or 3.3+
  + Python 2.6 + argparse
  + Numpy >= 1.9
+ Highly Recommended
  + Matplotlib >= 1.4 (if you want to make 2d plots using viscid.plot.vpyplot)
  + Scipy (gives Viscid special powers :))
  + Numexpr (for the calculator.necalc module)
  + H5py (if reading hdf5 files)
+ Truly Optional
  + Seaborn
  + Mayavi2 (if you want to make 3d plots using viscid.plot.vlab)
  + PyYaml (rc file and plot options can parse using yaml)
+ Optional for developers
  + Cython > 0.17 (if you change pyx / pxd files)
  + Sphinx
  + sphinx_rtd_theme
  + sphinxcontrib-napoleon (if Sphinx is <= version 1.2)

The optional calculator modules (necalc and cycalc) are all dispatched through
calculator.calc, and it is intelligent enough not to use a library that is not
installed.

The jrrle and fortbin readers depend on compiled Fortran code, and the interpolation and streamline functions depend on compiled Cython (C) code.

For explicit installation instructions, please refer to the [Quickstart Documentation](http://kristoformaynard.github.io/Viscid-docs/docs/master/installation.html).

## Development ##

Please, if you edit the code, use [PEP 8](https://www.python.org/dev/peps/pep-0008/) style. Poor style is more than just aesthetic; it tends to lead to bugs that are difficult to spot. Check out the documentation for a more complete developer's guide (inculding exceptions to PEP 8 that are ok).
