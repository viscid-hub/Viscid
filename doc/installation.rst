Installation
============

Dependencies
------------

+ Required

  + Python 2.7+ or 3.3+
  + Python 2.6 and argparse
  + Numpy >= 1.9

+ Highly Recommended

  + H5py (if reading hdf5 files)
  + Matplotlib >= 1.4 (if you want to make 2d plots using viscid.plot.vpyplot)
  + Scipy (gives Viscid special powers :))
  + Numexpr (for the calculator.necalc module)

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

Quickstart
----------

Installing Anaconda (optional but recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `Anaconda Python Distribution <https://store.continuum.io/cshop/anaconda/>`_ makes installing dependencies as easy as running one shell command.

.. code-block:: bash

    if [ "$(uname -s)" == "Linux" ]; then
      wget -O miniconda.sh https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
    elif [ "$(uname -s)" == "Darwin" ]; then
      curl -o miniconda.sh https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh
    fi
    bash miniconda.sh -b -p $HOME/local/anaconda
    rm miniconda.sh
    source ~/.bashrc
    hash -r
    conda config --set changeps1 no
    conda update -q conda

Downloading Viscid
~~~~~~~~~~~~~~~~~~

This should be done in whatever directory you want to store the Viscid source code. I use `~/src` myself.

.. code-block:: bash

    git clone https://github.com/KristoforMaynard/Viscid.git
    mkdir -p ~/.config/matplotlib
    cp Viscid/resources/viscidrc ~/.viscidrc

Installing Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

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


Building / Installing Viscid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now you have two choices about how you want to use Viscid. If you intend to edit viscid then I recommend using it inplace. Otherwise, it probably makes more sense to simply install viscid into your python distribution.

Choice 1 (installed)
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    cd Viscid
    python setup.py install
    # the above line is synonymous with `make install`

    # or, if you don't have write permission, try
    # python setup.py install --user

to kick the tires, use

.. code-block:: bash

    make instcheck

to pull updates from github in the future, use

.. code-block:: bash

    git pull
    python setup.py install

    # or, if you don't have write permission, try
    # python setup.py install --user

Choice 2 (inplace)
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

to kick the tires, use

.. code-block:: bash

    make check

to pull updates from github in the future, use

.. code-block:: bash

    git pull
    python setup.py build_ext -i

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
