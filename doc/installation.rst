Installation
============

.. raw:: html

    <style type="text/css">
    img {
        padding-top: 10pt;
        padding-bottom: 12pt;
    }
    </style>

Dependencies
------------

+ Required

  + Python 2.7+ or 3.3+
  + Python 2.6 and argparse
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

Installing Anaconda (optional but recommended)
----------------------------------------------

The `Anaconda Python Distribution <https://www.anaconda.com/distribution/>`_ makes managing dependencies and virtual environments wicked straight forward (and almost pleasant). You can download the full distribution, but it's laden with packages you probably don't need. Also, since ``conda install`` is so easy to use, I recommend the lightweight miniconda:

.. code-block:: bash

    if [ "$(uname -s)" == "Linux" ]; then
      wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    elif [ "$(uname -s)" == "Darwin" ]; then
      curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    fi
    bash miniconda.sh -b -p $HOME/local/anaconda
    rm miniconda.sh
    source ~/.bashrc
    hash -r
    conda config --set changeps1 no
    conda update -q conda

Installing Viscid
-----------------

You have a few choices for installing Viscid. Here is a quick breakdown of why you might choose one method over the another,

+ `Anaconda <http://anaconda.org>`_

  - **+**  Install with a single command
  - **+**  No compiler needed
  - **+**  Available for macOS, Linux, and Windows
  - **+**  Automatically installs dependancies
  - **-**  Lacks jrrle file support

+ PyPI (pip)

  - **+**  Install with a single command
  - **+**  No compiler needed for pure python functionality
  - **-**  Recommended dependancies must be installed by hand
  - **-**  Requires a C compiler for interpolation and streamline functions
  - **-**  Requires a Fortran compiler for jrrle file support

+ Source

  - **+**  Most flexable
  - **+**  Only way to use a development version
  - **-**  Requires a few commands and some knowledge about PATH and PYTHONPATH (but don't let this scare you, it's fairly straight forward)
  - **-**  Recommended dependancies must be installed by hand
  - **-**  Requires a C compiler for interpolation and streamline functions
  - **-**  Requires a Fortran compiler for jrrle file support

Choice 1: `Anaconda <http://anaconda.org>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://anaconda.org/kristoformaynard/viscid/badges/version.svg
  :target: https://anaconda.org/kristoformaynard/viscid
  :alt: Anaconda Version

.. image:: https://anaconda.org/kristoformaynard/viscid/badges/platforms.svg
  :target: https://anaconda.org/kristoformaynard/viscid
  :alt: Anaconda Platforms

If you have Anaconda, then installing Viscid and all the recommended dependancies happens with one command,

.. code-block:: bash

    conda install -c kristoformaynard viscid

You can check that the install succeeded by running,

.. code-block:: bash

    python -m viscid --check

.. attention::

    The binaries unloaded to anaconda lack support for reading jrrle files. If you need this, you will need to install from source (using pip or otherwise) and have a Fortran compiler.

Choice 2: PyPI (pip)
~~~~~~~~~~~~~~~~~~~~

.. image:: https://img.shields.io/pypi/v/Viscid.svg
  :target: https://pypi.org/project/Viscid/
  :alt: PyPI

You can install from source using pip with a single command, but the functionality depends on what compilers are available. Most of Viscid is pure python, but interpolation and streamline calculation requires a C compiler, and the jrrle reader requires a Fortran compiler. Also, Viscid only lists Numpy as a dependancy in pip so that installation will succeed on even the most bare-bones systems. This means you will want to install any of the recommended / optional dependancies yourself.

.. code-block:: bash

    pip install viscid

Compile errors will not cause Viscid's pip install to fail, and pip hides warning messages unless you use the ``-v`` flag. To check the functionality of your install, run

.. code-block:: bash

    python -m viscid --check

Choice 3: Source
~~~~~~~~~~~~~~~~

First, you'll have to clone the Viscit git repository. This should be done in whatever directory you want to store the Viscid source code. I use ``~/src`` myself.

.. code-block:: bash

    git clone https://github.com/KristoforMaynard/Viscid.git
    mkdir -p ~/.config/matplotlib
    cp Viscid/resources/viscidrc ~/.viscidrc

If you are using Anaconda to manage your dependencies, you can use the default Viscid environment to automatically install all Viscid's dependencies,

.. code-block:: bash

    conda env create -f Viscid/resources/viscid27.yml
    # for the more adventurous, you can try python 3.5
    conda env create -f Viscid/resources/viscid35mayavi.yml

Note that in order to use Viscid, you will need to activate the virtual environment that we just created (this needs to be done for each new terminal session),

.. code-block:: bash

    source activate viscid27  # or viscid35mayavi, etc.

An alternative to activating this environment for each session is to prepend PATH in your profile with

.. code-block:: bash

    profile="${HOME}/.bashrc"
    echo "export PATH=~/local/anaconda/envs/viscid27:"'${PATH}' >> ${profile}
    echo 'export CONDA_DEFAULT_ENV="$(basename "$(cd "$(dirname "$(which python)")/.."; pwd)")"' >> ${profile}
    echo 'export CONDA_PREFIX="$(cd "$(dirname "$(which python)")/.."; pwd)"' >> ${profile}
    source ${profile}

Now you have two choices about how you want to use Viscid. If you intend to edit viscid then I recommend using it inplace. Otherwise, it probably makes more sense to simply install viscid into your python distribution.

Choice 3a: installed
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    cd Viscid
    python setup.py install
    # the above line is synonymous with `make install`

    # or, if you don't have write permission, try
    # python setup.py install --user

To see if the install succeeded, try

.. code-block:: bash

    # kick the tires
    python -m viscid --check
    # run the full test suite
    make instcheck

To pull updates from github in the future, use

.. code-block:: bash

    git pull
    python setup.py install

Choice 3b: inplace
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    cd Viscid
    python setup.py build_ext -i
    # the above line is synonymous with `make inplace`

    # To set environment variables in Bash
    profile="${HOME}/.bashrc"
    echo 'export PATH="${PATH}:'"${PWD}/scripts\"" >> ${profile}
    echo 'export PYTHONPATH="${PYTHONPATH}:'"${PWD}\"" >> ${profile}
    source ${profile}

To see if the bulid succeeded, try

.. code-block:: bash

    # kick the tires
    python -m viscid --check
    # run the full test suite
    make check

To pull updates from github in the future, use

.. code-block:: bash

    git pull
    python setup.py build_ext -i

Verify Installation
-------------------

You can always run the following to check for any installation warnings. It is most helpful when verifying whether or not the C / Fortran modules compiled successfully.

.. code-block:: bash

    python -m viscid --check

Known Workarounds
-----------------

Ubuntu
~~~~~~

All Linux workarounds are currently incorporated in ``setup.py``.

OS X
~~~~

If you get an abort trap that says ``PyThreadState_Get: no current thread`` when trying to use mayavi, then this is probably yet another anaconda packaging issue. The solution is to roll back to a different sub-release of python. running this did the trick for me: ``conda install python=3.5.3 pyqt=4``.

If you see a link error that says `-lgcc_s.10.5` can't be found, try running::

    sudo su root -c "mkdir -p /usr/local/lib && ln -s /usr/lib/libSystem.B.dylib /usr/local/lib/libgcc_s.10.5.dylib"
