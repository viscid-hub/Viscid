.. note::

    To install mayavi, I *strongly* suggest using the viscid-hub anaconda channel. This solves some vtk / pyqt issues with the current anaconda and conda-forge binaries. To install, just run ``conda install -c viscid-hub mayavi``. Also, note that mayavi on python3 requires vtk >= 7, which requires OpenGL 3.2 or MESA >= 11.2. If you have older drivers, you must install mayavi with python2.7 and vtk5.
