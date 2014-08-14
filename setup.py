#!/usr/bin/env python
# Notes: cython is auto-detected, so if it's not found, then the cython
# extensions are built / cleaned if the .c files are found, else, the
# extension is ignored gracefully

from __future__ import print_function
import sys
import os
import glob
from subprocess import Popen, CalledProcessError, PIPE
from distutils.command.clean import clean
from distutils import log
# from distutils.core import setup
# from distutils.extension import Extension
from distutils import sysconfig

import numpy as np
from numpy.distutils.core import setup
from numpy.distutils.extension import Extension as Extension
npExtension = Extension

from doc import ver

try:
    from Cython.Build import cythonize
    has_cython = True
except ImportError:
    has_cython = False

# listing the sources
cmdclass = {}
pkgs = ['viscid',
        'viscid.calculator',
        'viscid.plot',
        'viscid.readers',
       ]

scripts = glob.glob(os.path.join('scripts', '*'))

# list of extension objects
ext_mods = []
# cy_defs is [["path.to.module1", ["path/to/src1", "path/to/src2"], dict()],
#             ["path.to.module2", ["path/to/src1", "path/to/src2"], dict()]]
# note that sources should be without extension, .pyx will be
# appended if building with cython, and .c will be appended
# if using pre-generated c files
# dict are kwargs that go into the Extension() constructor
cy_ccflags = ["-Wno-unused-function"]
cy_ldflags = []
cy_defs = []
cy_defs.append(["viscid.calculator.cycalc",
                ["viscid/calculator/cycalc"],
                dict()
               ])
cy_defs.append(["viscid.calculator.integrate",
                ["viscid/calculator/integrate"],
                dict()
               ])
cy_defs.append(["viscid.calculator.streamline",
                ["viscid/calculator/streamline"],
                dict()
               ])

fort_fcflags = []
fort_ldflags = []
fort_defs = []
fort_defs.append(["viscid.readers._fortfile",
                  ["viscid/readers/_fortfile.F90"],
                  dict(define_macros=[("FSEEKABLE", 1), ("HAVE_STREAM", 1)])
                 ])
fort_defs.append(["viscid.readers._jrrle",
                  ["viscid/readers/_jrrle.f90"],
                  dict(define_macros=[("FSEEKABLE", 1), ("HAVE_STREAM", 1)])
                 ])

############################################################################
# below this line shouldn't need to be changed except for version and stuff

# FIXME: these should be distutils commands, but they're not
try:
    i = sys.argv.index("dev")
    sys.argv[i] = "build_ext"
    sys.argv.insert(i + 1, "-i")
    sys.argv.insert(i + 1, "--with-cython")
    if not has_cython:
        raise RuntimeError("dev builds imply you have cython, try just using "
                           "'build_ext -i --with-cython' for the most general "
                           "build (use cython if you have it)")
except ValueError:
    pass

try:
    i = sys.argv.index("devclean")
    sys.argv[i] = "clean"
    sys.argv.insert(i + 1, "-a")
    sys.argv.insert(i + 1, "--with-cython")
    if not has_cython:
        raise RuntimeError("devclean imply you have cython, try just using "
                           "'clean -a --with-cython' for the most general case "
                           "(clean .c files if you have cython)")
except ValueError:
    pass

try:
    sys.argv.remove("--with-cython")
    use_cython = True
except ValueError:
    use_cython = False

# check for multicore build
_nprocs = 1
for i, arg in enumerate(sys.argv):
    if arg.startswith("-j"):
        try:
            if arg.endswith("-j"):
                # get number from the next arg
                _nprocs = int(sys.argv.pop(i + 1))
            else:
                _nprocs = int(arg[2:])
            sys.argv.pop(i)
            break
        except IndexError:
            raise RuntimeError("Syntax for multiple process build is "
                               "-jN or -j N")
            sys.exit(2)
        except ValueError:
            raise RuntimeError("Number of build procs must be an integer")
            sys.exit(3)

# decide which extension to add to cython sources (pyx or c)
cy_ext = ".c"  # or ".cpp"?
if has_cython and use_cython:
    cy_ext = ".pyx"

# add extension to cython sources
for i, d in enumerate(cy_defs):
    for j, src in enumerate(d[1]):
        fname = cy_defs[i][1][j] + cy_ext
        if os.path.isfile(fname):
            cy_defs[i][1][j] = fname
        else:
            log.warn("{0} not found. Skipping extension: "
                     "{1}".format(fname, cy_defs[i][0]))
            print("To use this extension, please install cython",
                  file=sys.stderr)
            cy_defs[i] = None
            break

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

                # remove c files if has_cython
                if has_cython and use_cython:
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
    if d is None:
        continue

    src_lst = d[1]

    _ext = Extension(d[0], src_lst, extra_compile_args=cy_ccflags,
                     extra_link_args=cy_ldflags, **d[2])
    ext_mods += [_ext]

if has_cython and use_cython:
    ext_mods = cythonize(ext_mods, nthreads=_nprocs)

# make fortran extension instances
for d in fort_defs:
    if d is None:
        continue
    src_lst = d[1]
    ext_mods += [npExtension(d[0], src_lst, extra_compile_args=fort_fcflags,
                             extra_link_args=fort_ldflags, **d[2])]

# hack for OSX pythons that are compiled with gcc symlinked to llvm-gcc
if sys.platform == "darwin" and "-arch" in sysconfig.get_config_var("CFLAGS"):
    cc = sysconfig.get_config_var("CC")
    try:
        cc_version = Popen([cc, "--version"], stdout=PIPE,
                            stderr=PIPE).communicate()[0].decode()
        if "MacPorts" in cc_version:
            cc = "llvm-gcc"
            cc = Popen(["which", cc], stdout=PIPE).communicate()[0].strip()
            os.environ["CC"] = cc
            cc = cc.decode()
            print("switching compiler to", cc)
    except (CalledProcessError, OSError):
        print("I think there's a problem with your compiler ( CC =", cc,
              "), but I'll continue anyway...")

setup(name='viscid',
      version=ver.release,
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
