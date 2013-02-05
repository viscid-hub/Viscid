#!/usr/bin/env python

import sys
import os
import glob

from distutils import log
from distutils.core import setup
from distutils.extension import Extension
import numpy as np

from distutils.command.clean import clean
# from distutils.command.build import build
from distutils.command.build_ext import build_ext

with_cython = False

# listing the sources
cmdclass = {}
pkgs = ['viscid',
        'viscid.calculator',
        'viscid.plot',
        'viscid.readers',
        'viscid.tools',
       ]

scripts = glob.glob(os.path.join('scripts', '*'))

# list of extension objects
ext_mods = []
# cy_defs is [["path.to.module1", ["path/to/src1", "path/to/src2"], ["other"]],
#             ["path.to.module2", ["path/to/src1", "path/to/src2"], ["other"]]]
# note that sources should be without extension, .pyx will be
# appended if building with cython, and .c will be appended
# if using pre-generated c files
# "other" files are files with set extensions, like .pxd dependancies
cy_ccflags = ["-Wno-unused-function"]
cy_ldflags = []
cy_defs = []
cy_defs.append(["viscid.calculator.cycalc",
                ["viscid/calculator/cycalc"],
                ["viscid/calculator/cycalc_util.pxd"]])

############################################################################
# below this line shouldn't need to be changed except for version and stuff

# decide which extension to add to cython sources (pyx or c)
cy_ext = ".c" #  or ".cpp"?
try:
    sys.argv.remove("--with-cython")
    try:
        from Cython.Distutils import build_ext
        cy_ext = ".pyx"
        cmdclass["build_ext"] = build_ext
        with_cython = True
    except ImportError:
        raise ValueError()
except ValueError:
    pass  # cy_ext is already ".c"

# add extension to cython sources
for i, d in enumerate(cy_defs):
    for j, src in enumerate(d[1]):
        # if os.path.isfile():
        cy_defs[i][1][j] += cy_ext

# get clean to remove inplace files
class Clean(clean):
    def run(self):
        # distutils uses old-style classes???
        #super(Clean, self).run()
        clean.run(self)

        if self.all:
            # remove inplace extensions
            for ext in self.distribution.ext_modules:
                fn = os.path.join(*ext.name.split('.'))
                files = [fn + ".so", fn + ".pyd"]
                for fn in files:
                    if os.path.isfile(fn):
                        log.info("removing '{0}'".format(fn))
                        os.unlink(fn)

                # remove c files if cleaning --with-cython
                if with_cython:
                    for f in ext.sources:
                        if f[-4:] == ".pyx":
                            for rm_ext in ['.c', '.cpp']:
                                fn = f[:-4] + rm_ext
                                if os.path.isfile(fn):
                                    log.info("removing '{0}'".format(fn))
                                    os.unlink(fn)

cmdclass["clean"] = Clean

# make cython extension instances
for d in cy_defs:
    src_lst = d[1]
    if with_cython:
        src_lst += d[2]
    ext_mods += [Extension(d[0], src_lst, extra_compile_args=cy_ccflags,
                           extra_link_args=cy_ldflags)]

setup(name='viscid',
      version='0.011',
      description='Visualization in python',
      author='Kris Maynard',
      author_email='k.maynard@unh.edu',
      packages=pkgs,
      cmdclass = cmdclass,
      include_dirs = [np.get_include()],
      ext_modules = ext_mods,
      scripts=scripts,
     )

##
## EOF
##
