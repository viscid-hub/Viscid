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
  + Matplotlib >= 1.4 (if you want to make 2d plots using viscid.plot.mpl)
  + Scipy (gives Viscid special powers :))
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

To get the dependancies squared away, I recommend using the `anaconda <https://store.continuum.io/cshop/anaconda/>`_ python distribution. It makes installing new python libraries almost enjoyable.

Quickstart
----------

If you want to **install Anaconda python** to manage your dependancies, start with

.. code-block:: bash

    if [ "$(uname -s)" == "Linux" ]; then
      wget -O miniconda.sh https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
    elif [ "$(uname -s)" == "Darwin" ]; then
      curl -O miniconda.sh https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh
    fi
    bash miniconda.sh -b -p $HOME/local/anaconda
    rm miniconda.sh
    source ~/.bashrc
    hash -r
    conda config --set changeps1 no
    conda update -q conda

**Getting and installing Viscid**; this should be done in whatever directory you want to store the Viscid source code. I use `~/src` myself.

.. code-block:: bash

    git clone git@github.com:KristoforMaynard/Viscid.git
    mkdir -p ~/.config/matplotlib
    cp Viscid/resources/viscidrc ~/.viscidrc

If you are using Anaconda to manage your dependancies, you can **use the default Viscid environment** to automatically install all Viscid's dependancies,

.. code-block:: bash

    conda env create -f Viscid/resources/viscid27.yml

Note that in order to use Viscid, you will need to run

.. code-block:: bash

    source activate viscid27

to activate the viscid environment for each new terminal session, or just prepend your PATH in your ~/.bashrc with

.. code-block:: bash

    export PATH="~/local/anaconda/envs/viscid27:${PATH}"

Now you have two choices about how you want to use Viscid. If you intend to edit viscid to your own liking then I recommend using it inplace. Otherwise, it probably makes more sense to simply install viscid into your python distribution.

Choice 1 (inplace)
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cd Viscid
    make inplace
    export PATH="${PATH}:${PWD}/Viscid/scripts"
    export PYTHONPATH="${PYTHONPATH}:${PWD}/Viscid"
    echo 'export PATH="${PATH}:'"${PWD}/scripts\"" >> ~/.bashrc
    echo 'export PYTHONPATH="${PYTHONPATH}:'"${PWD}\"" >> ~/.bashrc

to pull updates from github,

.. code-block:: bash

    git pull
    make inplace

Choice 2 (installed)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cd Viscid
    make
    make install

to pull updates from github,

.. code-block:: bash

    git pull
    make
    make install

Known Workarounds
-----------------

Ubuntu
~~~~~~

If you see an error that contains `GFORTRAN_1.4 not found`, you may need to preempt libgfortran with the system version. The solution is an environment variable that looks something like::

    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgfortran.so.3.0.0

OS X
~~~~

If you see a link error that says `-lgcc_s.10.5` can't be found, try running::

    sudo su root -c "mkdir -p /usr/local/lib && ln -s /usr/lib/libSystem.B.dylib /usr/local/lib/libgcc_s.10.5.dylib"
