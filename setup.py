#!/usr/bin/env python
# Notes: cython is auto-detected, so if it's not found, then the cython
# extensions are built / cleaned if the .c files are found, else, the
# extension is ignored gracefully

from __future__ import print_function
from glob import glob
import io
import json
import os
import re
import shutil
from subprocess import Popen, CalledProcessError, PIPE
import sys

try:
    import setuptools
except ImportError:
    pass

from distutils.command.clean import clean
from distutils.errors import CompileError
from distutils.version import LooseVersion
from distutils import log
from distutils import sysconfig

import numpy as np
from numpy.distutils.command.build_ext import build_ext
from numpy.distutils.core import setup
from numpy.distutils.exec_command import exec_command
from numpy.distutils import fcompiler as _fcompiler
from numpy.distutils.extension import Extension as Extension
npExtension = Extension


INSTALL_MANIFEST = '.install_manifest.json'
RECORD_FNAME = '.temp_install_list.txt'


try:
    from Cython.Build import cythonize
    has_cython = True
except ImportError:
    has_cython = False

if sys.version_info >= (3, 0):
    PY3K = True
else:
    PY3K = False


try:
    FileNotFoundError
except NameError:
    class FileNotFoundError(Exception):
        pass


# listing the sources
cmdclass = {}
pkgs = ['viscid',
        'viscid.calculator',
        'viscid.cython',
        'viscid.compat',
        'viscid.compat.futures',
        'viscid.plot',
        'viscid.readers'
       ]

scripts = glob(os.path.join('scripts', '*'))


# list of extension objects
ext_mods = []
# cy_defs is [["path.to.module1", ["path/to/src1", "path/to/src2"], dict()],
#             ["path.to.module2", ["path/to/src1", "path/to/src2"], dict()]]
# note that sources should be without extension, .pyx will be
# appended if building with cython, and .c will be appended
# if using pre-generated c files
# dict are kwargs that go into the Extension() constructor
if sys.platform[:5] == 'linux' or sys.platform == 'darwin':
    cy_ccflags = ["-Wno-unused-function"]
else:
    cy_ccflags = [""]
cy_ldflags = []
cy_defs = []
cy_defs.append(["viscid.cython.cycalc",
                ["viscid/cython/cycalc"],
                dict()
               ])
cy_defs.append(["viscid.cython.integrate",
                ["viscid/cython/integrate"],
                dict()
               ])
cy_defs.append(["viscid.cython.streamline",
                ["viscid/cython/streamline"],
                dict()
               ])
cy_defs.append(["viscid.cython.null_tools",
                ["viscid/cython/null_tools"],
                dict()
               ])
cy_defs.append(["viscid.cython.cyfield",
                ["viscid/cython/cyfield"],
                dict()
               ])
cy_defs.append(["viscid.cython.cyamr",
                ["viscid/cython/cyamr"],
                dict()
               ])

fort_fcflags = []
fort_ldflags = os.environ.get('F_LDFLAGS', '').split()

fort_defs = []
# These fortran sources are compiled into the same python module since
# they pass an open file unit back and forth. This doesn't seem to work
# between python modules on windows.
fort_defs.append(["viscid.readers._jrrle",
                  ["viscid/readers/_fortfile.F90", "viscid/readers/_jrrle.f90"],
                  dict(define_macros=[("FSEEKABLE", 1), ("HAVE_STREAM", 1)])
                 ])

############################################################################
# below this line shouldn't need to be changed except for version and stuff

# hack for gfortran / conda-build on macOS
if sys.platform == 'darwin':
    if 'LDFLAGS' in os.environ or 'CONDA_BUILD_STATE' in os.environ:
        mandatory_flags = ['-undefined dynamic_lookup', '-bundle']
        for flag in mandatory_flags:
            if flag not in os.environ.get('LDFLAGS', ''):
                if flag not in fort_ldflags:
                    fort_ldflags.append(flag)

# fix fortran build on linux with conda's gfortran
if sys.platform[:5] == 'linux':
    if 'LDFLAGS' in os.environ or 'CONDA_BUILD_STATE' in os.environ:
        mandatory_flags = ['-shared']
        for flag in mandatory_flags:
            if flag not in os.environ.get('LDFLAGS', ''):
                if flag not in fort_ldflags:
                    fort_ldflags.append(flag)

# hack to use proper flags for python2.7 / numpy 1.11 windows conda-forge build
if (sys.platform[:3] == 'win' and sys.version_info[:2] == (2, 7)
    and 'CONDA_BUILD_STATE' in os.environ):
    mandatory_fcflags = ['-O3', '-funroll-loops']
    for flag in mandatory_fcflags:
        fort_fcflags.append(flag)
    mandatory_fflags = ['-Wl,--allow-multiple-definition',
                        '-Wl,--export-all-symbols',
                        '-static', '-mlong-double-64']
    for flag in mandatory_fflags:
        fort_ldflags.append(flag)

# hack to remove bad path for python2.7 / numpy 1.11 linux conda-forge build
if sys.platform[:5] == 'linux' and 'CONDA_BUILD_STATE' in os.environ:
    bad_path = os.path.join('opt', 'rh', 'devtoolset-2', 'root', 'usr', 'bin')
    split_path = os.environ.get("PATH", "").split(os.pathsep)
    split_path = [pth for pth in split_path if bad_path not in pth]
    os.environ["PATH"] = os.pathsep.join(split_path)

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

# prepare a cute hack to get an `uninstall`
desired_record_fname = ''
if 'install' in sys.argv:
    try:
        i = sys.argv.index('--record')
        sys.argv.pop(i)
        desired_record_fname = sys.argv[i]
        sys.argv.pop(i)
    except ValueError:
        pass
    except IndexError:
        print("error: --record must be followed by a filename")
        sys.exit(4)
    sys.argv += ['--record', RECORD_FNAME]

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
            log.warning("{0} not found. Skipping extension: "
                        "{1}".format(fname, cy_defs[i][0]))
            print("To use this extension, please install cython",
                  file=sys.stderr)
            cy_defs[i] = None
            break

def clean_pyc_files(dry_run=False):
    """remove all .pyc / .pyo files"""
    cwd = os.getcwd()
    if cwd.endswith("Viscid"):
        for root, _, files in os.walk(cwd, topdown=False):
            for name in files:
                if name.endswith('.pyc') or name.endswith('.pyo'):
                    if os.path.isfile(os.path.join(root, name)):
                        print('removing: %s' % os.path.join(root, name))
                        if not dry_run:
                            os.remove(os.path.join(root, name))
    else:
        print("Not in Viscid directory, not cleaning pyc/pyo files")

def clean_other_so_files(valid_so_list, dry_run=False):
    """remove all .pyc / .pyo files"""
    cwd = os.getcwd()
    if cwd.endswith("Viscid"):
        for root, _, files in os.walk(cwd, topdown=False):
            for name in files:
                name = os.path.join(root, name)
                if name.endswith(('.so', '.pyd')) and name not in valid_so_list:
                    if os.path.isfile(name):
                        print('removing other: %s' % name)
                        if not dry_run:
                            os.remove(name)
    else:
        print("Not in Viscid directory, not cleaning pyc/pyo files")


# get clean to remove inplace files
class Clean(clean):
    def run(self):
        try:
            super(Clean, self).run()
        except TypeError:
            clean.run(self)

        clean_pyc_files(self.dry_run)

        # clean inplace so files that are not in ext_modules
        # this will clean old object files if cython modules move
        so_file_list = []
        for ext in self.distribution.ext_modules:
            fn = os.path.join(*ext.name.split('.')) + ".so"
            so_file_list.append(os.path.abspath(fn))
        clean_other_so_files(so_file_list, self.dry_run)

        _libs_dir = os.path.join('viscid', '.libs')
        for other_dir in ['dist', 'Viscid.egg-info', _libs_dir]:
            other_dir = os.path.join(os.path.dirname(__file__), other_dir)
            if os.path.isdir(other_dir):
                print("removing '{0}/'".format(other_dir))
                if not self.dry_run:
                    shutil.rmtree(other_dir)

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


build_ext_ran = False
build_ext_failed = False
class BuildExt(build_ext):
    def run(self, *args, **kwargs):
        global build_ext_ran
        build_ext_ran = True
        try:
            build_ext.run(self, *args, **kwargs)
            # copy the weird extra dll dir on windows (fortran)
            flibdll_dir = os.path.join(self.build_lib, 'viscid', '.libs')
            if os.path.isdir(flibdll_dir):
                self.copy_tree(flibdll_dir, os.path.join(os.path.dirname(__file__),
                                                         'viscid', '.libs'))
        except Exception as e:
            global build_ext_failed
            build_ext_failed = True
            print(e, file=sys.stderr)
cmdclass["build_ext"] = BuildExt

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
try:
    fc = _fcompiler.new_fcompiler(dry_run=True)
except ValueError:
    fc = None

if fc is None:
    # warn the user at the very end so they're more likely to see the warning
    pass
else:
    # the folloing gfortran hacks are to work around a treacherous bug when
    # using anaconda + f2py (numpy) + a local gfortran on MacOS / Linux...
    # in short, anaconda ships its own libgfortran to make numpy / scipy
    # work, but this will conflict with some gfortran compilers...
    # AND numpy assumes that `gfortran -print-libgcc-file-name` will give the
    # path to libgfortran (it practice it doesn't on Debian / MacPorts)... SO
    # we let numpy discover the path to libgfortran using
    # `gfortran -print-file-name=libgfortran.so` (or dylib) because that will
    # give us the correct link path, and fortran code that we compile with f2py
    # will be correctly linked to the lib supplied by OUR compiler
    if sys.platform[:5] == 'linux' or sys.platform == 'darwin':
        from numpy.distutils.fcompiler import gnu as _gnu
        from numpy.distutils import fcompiler as _fcompiler

        def _get_libgfortran_dir(compiler_args):
            if sys.platform[:5] == 'linux':
                libgfortran_name = 'libgfortran.so'
            elif sys.platform == 'darwin':
                libgfortran_name = 'libgfortran.dylib'
            else:
                libgfortran_name = None

            libgfortran_dir = None
            if libgfortran_name:
                find_lib_arg = ['-print-file-name={0}'.format(libgfortran_name)]
                status, output = exec_command(compiler_args + find_lib_arg,
                                              use_tee=0)
                if not status:
                    libgfortran_dir = os.path.dirname(output)
            return libgfortran_dir

        class GnuFCompilerHack(_gnu.GnuFCompiler):
            def get_library_dirs(self):
                try:
                    opt = super(GnuFCompilerHack, self).get_library_dirs()
                except TypeError:
                    # old distutils use old-style classes
                    opt = _gnu.GnuFCompiler.get_library_dirs(self)
                if sys.platform[:5] == 'linux' or sys.platform == 'darwin':
                    libgfortran_dir = _get_libgfortran_dir(self.compiler_f77)
                    if libgfortran_dir:
                        opt.append(libgfortran_dir)
                return opt

        class Gnu95FCompilerHack(_gnu.Gnu95FCompiler):
            def get_library_dirs(self):
                try:
                    opt = super(Gnu95FCompilerHack, self).get_library_dirs()
                except TypeError:
                    # old distutils use old-style classes
                    opt = _gnu.Gnu95FCompiler.get_library_dirs(self)
                if sys.platform[:5] == 'linux' or sys.platform == 'darwin':
                    libgfortran_dir = _get_libgfortran_dir(self.compiler_f77)
                    if libgfortran_dir:
                        opt.append(libgfortran_dir)
                    return opt

        _fcompiler.load_all_fcompiler_classes()
        _fcompiler.fcompiler_class['gnu'] = ('gnu', GnuFCompilerHack,
                                             'GNU Fortran 77 compile Hack')
        _fcompiler.fcompiler_class['gnu95'] = ('gnu', Gnu95FCompilerHack,
                                               'GNU Fortran 95 compile Hack')

    # ok, now that that hack is over with, add our fortran extensions
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
                           stderr=PIPE).communicate()[0]
        if PY3K:
            cc_version = cc_version.decode()
        if "MacPorts" in cc_version:
            cc = "llvm-gcc"
            cc = Popen(["which", cc], stdout=PIPE).communicate()[0]
            if PY3K:
                cc = cc.decode()
            cc = cc.strip()
            os.environ["CC"] = cc
            print("switching compiler to", cc)
    except (CalledProcessError, OSError):
        print("I think there's a problem with your compiler ( CC =", cc,
              "), but I'll continue anyway...")


def get_viscid_version(init_py):
    with io.open(init_py, 'r', encoding="utf-8") as f:
        version = None
        quoted_str = r"((?<![\\])(?:'''|\"\"\"|\"|'))((?:.(?!(?<![\\])\1))*.?)\1"
        ver_re = r"__version__\s*=\s*" + quoted_str
        for line in f:
            m = re.search(ver_re, line)
            if m:
                return m.groups()[1]

try:
    data_files = []
    data_files += [('viscid/plot/images', glob("viscid/plot/images/*.jpg"))]
    data_files += [('viscid/plot/styles', glob('viscid/plot/styles/*.mplstyle'))]
    for dirpath, _, fnames in os.walk('viscid/sample'):
        fnames = [os.path.join(dirpath, fname)
                  for fname in fnames if not fname.startswith('.')]
        data_files += [(os.path.join(dirpath), fnames)]

    version = get_viscid_version("viscid/__init__.py")
    url = "https://github.com/viscid-hub/Viscid"
    download_url = "{0}/archive/{1}.zip".format(url, version)

    with open("README.md", "r") as fh:
        long_description = fh.read()

    setup(name='viscid',
          version=version,
          description='Visualize data on structured meshes in python',
          long_description=long_description,
          long_description_content_type="text/markdown",
          author='Kristofor Maynard',
          author_email='k.maynard@unh.edu',
          license='MIT',
          url=url,
          download_url=download_url,
          keywords=['visualization', 'physics'],
          install_requires=['numpy>=1.9'],
          packages=pkgs,
          cmdclass=cmdclass,
          include_dirs=[np.get_include()],
          ext_modules=ext_mods,
          scripts=scripts,
          data_files=data_files,
          zip_safe=False,
          classifiers=(
              "Programming Language :: Python :: 2.6",
              "Programming Language :: Python :: 2.7",
              "Programming Language :: Python :: 3",
              "Programming Language :: Python :: 3.3",
              "Programming Language :: Python :: 3.4",
              "Programming Language :: Python :: 3.5",
              "Programming Language :: Python :: 3.6",
              "Programming Language :: Python :: 3.7",
              "License :: OSI Approved :: MIT License",
              "Operating System :: OS Independent",
              "Topic :: Scientific/Engineering",
              "Topic :: Scientific/Engineering :: Physics",
              "Topic :: Scientific/Engineering :: Visualization",
          ),
         )

    # if installed, store list of installed files in a json file - this
    # manifest is used to implement an uninstall
    if os.path.isfile(RECORD_FNAME):
        try:
            with open(INSTALL_MANIFEST, 'r') as fin:
                inst_manifest = json.load(fin)
        except (IOError, FileNotFoundError):
            inst_manifest = dict()

        with open(RECORD_FNAME) as fin:
            file_list = [line.strip() for line in fin]

        init_pys = [s for s in file_list if '__init__.py' in s or s.endswith('.egg')]
        pkg_instdir = os.path.dirname(min(init_pys, key=len))

        inst_manifest[sys.executable] = dict(pkg_instdir=pkg_instdir,
                                             file_list=file_list)

        with open(INSTALL_MANIFEST, 'w') as fout:
            json.dump(inst_manifest, fout, indent=2, sort_keys=True)

        if desired_record_fname:
            shutil.copy(RECORD_FNAME, desired_record_fname)
        os.remove(RECORD_FNAME)

except SystemExit as e:
    # if os.uname()[0] == 'Darwin':
    #     print('\n'
    #           'NOTE: OS X has an issue you may be running into.\n'
    #           '      If the compile is complaining that it can\'t find\n'
    #           '      -lgcc_s.10.5, then run the following:\n'
    #           '      \n'
    #           '      $ sudo su root -c "mkdir -p /usr/local/lib && ln -s '
    #           '/usr/lib/libSystem.B.dylib /usr/local/lib/libgcc_s.10.5.dylib"'
    #           '      \n', file=sys.stderr)
    raise

# warn the user at the end if the fortran code was not built
if build_ext_ran and fc is None:
    print("\n"
          "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
          "WARNING: No fortran compiler found!\n"
          "\n"
          "         Modules that depend on Fortran code will not work (eg,\n"
          "         the jrrle reader), but this may or may not be a problem\n"
          "         for you since most of Viscid does not depend on Fortran.\n"
          "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
          "\n", file=sys.stderr)

if build_ext_ran and build_ext_failed:
    print("\n"
          "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
          "WARNING: Extension compilation failed. \n"
          "\n"
          "         You may use all the great Python-only features of Viscid,\n"
          "         but you will not have access to Viscid's interpolation and\n"
          "         streamline capabilities. To use these functions, you will\n"
          "         need a C compiler compatible with your OS / Python.\n"
          "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
          "\n", file=sys.stderr)

##
## EOF
##
