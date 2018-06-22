Installation
============

.. raw:: html

    <style type="text/css">
    img {
        padding-top: 10pt;
        padding-bottom: 12pt;
    }
    </style>

.. raw:: html

    <style type="text/css">
    div.topic {
        border-style: none;
    }
    </style>

.. contents::
  :local:

Dependencies
------------

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
  + Mayavi2 [#f1]_ *(if you want to make 3d plots using viscid.plot.vlab)*
  + PyYaml *(rc file and plot options can parse using yaml)*

+ Optional for developers

  + Cython 0.28+ *(if you change pyx / pxd files)*
  + Sphinx 1.3+

.. [#f1] Installing Mayavi can be tricky. Please :ref:`read this section <installing-mayavi>` before attempting to install it.

Installing Anaconda (optional)
------------------------------

The `Anaconda Python Distribution <https://www.anaconda.com/distribution/>`_ makes managing dependencies and virtual environments wicked straight forward (and almost pleasant). You can download the full distribution, but it's laden with packages you probably don't need. Also, since ``conda install`` is so easy to use, I recommend the lightweight miniconda:

.. code-block:: bash

    if [ "$(uname -s)" == "Linux" ]; then
      wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    elif [ "$(uname -s)" == "Darwin" ]; then
      curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    fi
    bash miniconda.sh -b -p $HOME/local/anaconda
    rm miniconda.sh
    source $HOME/local/anaconda/etc/profile.d/conda.sh
    conda config --set changeps1 no
    conda update -q conda
    conda activate

You can now add something like this to your bashrc or profile,

.. _conda_bashrc_blurb:

.. code-block:: bash

    export _CONDA_ROOT_PREFIX=${HOME}/local/anaconda  # <- point to anaconda
    export CONDA_DEFAULT_ENV=base  # <- edit this to taste
    # there is no need to edit the following lines directly
    source ${_CONDA_ROOT_PREFIX}/etc/profile.d/conda.sh
    export CONDA_SHLVL=1
    export CONDA_EXE=${_CONDA_ROOT_PREFIX}/bin/conda
    if [ "${CONDA_DEFAULT_ENV}" = "base" ]; then
      export CONDA_PREFIX="${_CONDA_ROOT_PREFIX}"
    else
      export CONDA_PREFIX="${_CONDA_ROOT_PREFIX}/envs/${CONDA_DEFAULT_ENV}"
    fi
    export CONDA_PROMPT_MODIFIER=""
    export CONDA_PYTHON_EXE="${_CONDA_ROOT_PREFIX}/bin/python"
    export PATH="${CONDA_PREFIX}/bin:${PATH}"

Installing Viscid
-----------------

You have a few choices for installing Viscid. Here is a quick breakdown of why you might choose one method over another.

+ :ref:`Anaconda <choice1-conda>`

  - **+**  Installs with a single command
  - **+**  No compiler needed
  - **+**  Available for macOS, Linux, and Windows
  - **+**  Automatically installs recommended dependencies
  - **+**  Optional dependencies are equally easy to install
  - **-**  You won't be able to edit Viscid's source code. You might naively edit the modules in site-packages, but this will confuse the conda package manager beyond repair.

+ :ref:`PyPI (pip) <choice2-pypi>`

  - **+**  Installs with a single command
  - **+**  No compiler needed for pure python functionality
  - **-**  Recommended dependencies must be explicitly installed
  - **-**  Requires a C compiler for interpolation and streamline support
  - **-**  Requires a Fortran compiler for jrrle file support

+ :ref:`Source <choice3-source>`

  - **+**  Only method that lets you edit Viscid's source code
  - **-**  Requires some knowledge about PATH and PYTHONPATH (but don't let this scare you, it's fairly straight forward)
  - **-**  Dependencies must be explicitly installed
  - **-**  Requires a C compiler for interpolation and streamline functions
  - **-**  Requires a Fortran compiler for jrrle file support

.. _choice1-conda:

Choice 1: `Anaconda <http://anaconda.org/viscid-hub/viscid>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://anaconda.org/viscid-hub/viscid/badges/version.svg
  :target: https://anaconda.org/viscid-hub/viscid
  :alt: Anaconda Version

.. image:: https://anaconda.org/viscid-hub/viscid/badges/platforms.svg
  :target: https://anaconda.org/viscid-hub/viscid
  :alt: Anaconda Platforms

If you have Anaconda, then installing Viscid and all the recommended dependencies happens with one command,

.. code-block:: bash

    conda install -c viscid-hub viscid

You can check that the install succeeded by running,

.. code-block:: bash

    python -m viscid --check

.. _choice2-pypi:

Choice 2: `PyPI <http://pypi.org/project/viscid/>`_ (pip)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://img.shields.io/pypi/v/Viscid.svg
  :target: https://pypi.org/project/Viscid/
  :alt: PyPI

You can install from source using pip, but the runtime functionality depends on which compilers are available. Most of Viscid is pure python, but interpolation and streamline calculation requires a C compiler, and the jrrle reader requires a Fortran compiler.

.. code-block:: bash

    pip install viscid

Compile errors will not cause Viscid's pip install to fail, and pip hides warning messages unless you use the ``-v`` flag. To check the functionality of your install, run

.. code-block:: bash

    python -m viscid --check

.. _choice3-source:

Choice 3: Source
~~~~~~~~~~~~~~~~

First, you'll have to clone the Viscid git repository. This should be done in whatever directory you want to store the Viscid source code. I use ``~/src`` myself.

.. code-block:: bash

    git clone https://github.com/viscid-hub/Viscid.git

    # Optional: set qt5 as the default matplotlib backend
    mkdir -p ~/.config/matplotlib
    echo "backend: Qt5Agg" >> ~/.config/matplotlib/matplotlibrc

    # Optional: copy the default viscidrc file
    cp Viscid/resources/viscidrc ~/.viscidrc

If you are using Anaconda to manage your dependencies, you can use the default Viscid environment to automatically install all Viscid's dependencies,

.. code-block:: bash

    conda env create -f Viscid/resources/viscid36mayavi.yml

    # if you need mayavi, but don't have OpenGL 3.2, you
    # will have to use python2.7 (Viscid/resources/viscid27.yml)

    # this activation must be done for each new command prompt
    conda activate viscid36mayavi  # or viscid27, etc.

:ref:`Read this <conda_bashrc_blurb>` if you need help editing your bashrc or profile to set the default Anaconda environment.

Viscid should work with native compiler toolchains, but in case you don't have access to native compilers, the Anaconda toolchains should work too. `Check here <https://conda.io/docs/user-guide/tasks/build-packages/compiler-tools.html>`_ to find the appropriate packages for your platform. On Windows, you should use the MSVC compiler and Anaconda's ``m2w64-gcc-fortran``. `Read this <https://wiki.python.org/moin/WindowsCompilers>`_ for more information about acquiring the correct Microsoft compiler. If you are using CPython 3.5 or 3.6, you probably need MSVC 14 compiler, `available here <https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017>`_. You do **not** need to install visual studio to get the build tools.

Now you have a choice about how you want to use Viscid. If you intend to edit viscid then I recommend building it inplace. Otherwise, it probably makes more sense to simply install viscid into your python distribution.

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

To see if the build succeeded, try

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

.. _installing-mayavi:

Installing Mayavi (optional)
----------------------------

.. warning::

    Do **not** install Mayavi using pip into an Anaconda environment. This will break your conda environment in a way that requires you to reinstall Anaconda. The issue is that pip happily clobbers some parts of pyqt that are hard linked to a cache in conda. You have been warned.

.. note::

    I do not recommend using the ``conda-forge`` channel for VTK or Mayavi. These binaries frequently have runtime problems caused by interactions with specific versions of vtk / pyqt.

Installing Mayavi can be a mine field of incompatible dependencies. To help, here is a table to help you choose your poison. If your environment is not in the table, then it is likely not supported by Mayavi / VTK.

.. cssclass:: table-striped

=============  ==============  ========================  =================================================
OS             Python Version  OpenGL / MESA             Installation Command
=============  ==============  ========================  =================================================
MacOS          3.5+            OpenGL 3.2+               ``conda install -c viscid-hub mayavi``
MacOS          2.7             OpenGL 3.2+               ``conda install -c viscid-hub mayavi``
MacOS          2.7             Any                       ``conda install mayavi vtk=6``
Linux          3.5+            OpenGL 3.2+, MESA 11.2+   ``conda install -c viscid-hub mayavi``
Linux          2.7             OpenGL 3.2+, MESA 11.2+   ``conda install -c viscid-hub mayavi``
Linux          2.7             Any                       ``conda install mayavi vtk=6``
Windows        3.5+            OpenGL 3.2+               ``conda install -c viscid-hub mayavi``
Windows        2.7             Any                       ``conda install mayavi vtk=6``
=============  ==============  ========================  =================================================

The Enthought Tool Suite requires some confusing environment variables. Effectively you can just cycle through these options until you find a combination that works.

.. code-block:: bash

    export ETS_TOOLKIT='qt'  # or "qt4" or "wx" for older versions of pyface
    export QT_API="pyqt5"  # or "pyqt" or "pyside"

Also, if you are using MESA to emulate the new OpenGL API for VTK 7+, you may need the following environment variables,

.. code-block:: bash

    export MESA_GL_VERSION_OVERRIDE=3.2
    export MESA_GLSL_VERSION_OVERRIDE=150

:ref:`Check here <functions-mayavi>` for a discussion of Viscid's wrapper functions and workarounds, :doc:`and here <tutorial/mayavi>` for an example of Mayavi in action.

Workarounds
-----------

Ubuntu
~~~~~~

All Linux workarounds are currently incorporated in ``setup.py``.

OS X
~~~~

If you see a link error that says ``-lgcc_s.10.5`` can't be found, try running::

    sudo su root -c "mkdir -p /usr/local/lib && ln -s /usr/lib/libSystem.B.dylib /usr/local/lib/libgcc_s.10.5.dylib"
