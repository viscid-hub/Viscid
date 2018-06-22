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
[master](https://github.com/viscid-hub/Viscid)        | [html](http://viscid-hub.github.io/Viscid-docs/docs/master/index.html), [test summary](http://viscid-hub.github.io/Viscid-docs/summary/master-2.7/index.html)   | [![Build Status](https://travis-ci.com/viscid-hub/Viscid.svg?branch=master)](https://travis-ci.com/viscid-hub/Viscid)
[dev](https://github.com/viscid-hub/Viscid/tree/dev)  | [html](http://viscid-hub.github.io/Viscid-docs/docs/dev/index.html), [test summary](http://viscid-hub.github.io/Viscid-docs/summary/dev-2.7/index.html)      | [![Build Status](https://travis-ci.com/viscid-hub/Viscid.svg?branch=dev)](https://travis-ci.com/viscid-hub/Viscid)

## Install ##

[![Anaconda-Server Badge](https://anaconda.org/viscid-hub/viscid/badges/version.svg)](https://anaconda.org/viscid-hub/viscid) [![Anaconda-Server Badge](https://anaconda.org/viscid-hub/viscid/badges/platforms.svg)](https://anaconda.org/viscid-hub/viscid)

[![PyPI Version](https://img.shields.io/pypi/v/Viscid.svg)](https://pypi.org/project/Viscid/)

Dependencies:

+ Required
  + Python 2.6, 2.7, or 3.3+
  + Numpy 1.9+
  + Argparse (Python 2.6 only)
+ Recommended
  + IPython *(better interactive interpreter)*
  + Matplotlib 1.4+ *(if you want to make 2d plots using viscid.plot.vpyplot)*
  + Scipy *(enables nonlinear interpolation and curve fitting)*
  + Numexpr *(for faster math on large grids)*
  + H5py *(enables hdf5 reader)*
+ Optional
  + Seaborn
  + Mayavi2 <sup id="a1">[[1]](#f1)</sup> *(if you want to make 3d plots using viscid.plot.vlab)*
  + PyYaml *(rc file and plot options can parse using yaml)*
+ Optional for developers
  + Cython 0.28+ *(if you change pyx / pxd files)*
  + Sphinx 1.3+

Detailed installation instructions are [available here](http://viscid-hub.github.io/Viscid-docs/docs/master/installation.html).

<a id="f1">[[1]](#a1)</a> Installing Mayavi can be tricky. Please [read this](http://viscid-hub.github.io/Viscid-docs/docs/master/installation.html#installing-mayavi) before you try to install it.

## Development ##

Please, if you edit the code, use [PEP 8](https://www.python.org/dev/peps/pep-0008/) style. Poor style is more than just aesthetic; it tends to lead to bugs that are difficult to spot. Check out the documentation for a more complete developer's guide (inculding exceptions to PEP 8 that are ok).
