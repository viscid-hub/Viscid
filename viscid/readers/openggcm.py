#!/usr/bin/env python
""" Wrapper grid for some OpenGGCM convenience """

from __future__ import print_function, division
import os
import re
from itertools import islice
from operator import itemgetter

import numpy as np
try:
    import numexpr
    _has_numexpr = True
except ImportError:
    _has_numexpr = False

from viscid import logger
from viscid.compat import string_types
from viscid.readers.vfile_bucket import VFileBucket
from viscid.readers.ggcm_logfile import GGCMLogFile
from viscid.readers import vfile
from viscid import dataset
from viscid import grid
from viscid import field
from viscid.coordinate import wrap_crds
from viscid.calculator import plasma


def find_file_uptree(directory, basename, max_depth=8, _depth=0):
    """Find basename by going up the file tree

    Keep going up a directory until you find one that has the file
    "basename"

    Parameters:
        directory (str): directory to start the search
        basename (str): bare file name
        max_depth (int): max number of directories to seach

    Returns:
        Relative path to file, or None if not found
    """
    max_depth = 5

    if not os.path.isdir(directory):
        raise RuntimeError("this is non-sensicle")

    fname = os.path.join(directory, basename)
    # log_fname = "{0}/{1}.log".format(d, self.info["run"])
    if os.path.isfile(fname):
        return fname

    if _depth > max_depth or os.path.abspath(directory) == '/':
        return None

    return find_file_uptree(os.path.join(directory, ".."), basename,
                            _depth=(_depth + 1))

def group_ggcm_files_common(detector, fnames):
    infolst = []
    for name in fnames:
        m = re.match(detector, name)
        grps = m.groups()
        d = dict(runname=grps[0], ftype=grps[1],
                 fname=m.string)
        try:
            d["time"] = int(grps[2])
        except TypeError:
            # grps[2] is none for "RUN.3d.xdmf" files
            d["time"] = -1
        infolst.append(d)

    # careful with sorting, only consecutive files will be grouped
    infolst.sort(key=itemgetter("time"))
    infolst.sort(key=itemgetter("ftype"))
    infolst.sort(key=itemgetter("runname"))

    info_groups = []
    info_group = [infolst[0]]
    for info in infolst[1:]:
        last = info_group[-1]
        if info['runname'] == last['runname'] and \
           info['ftype'] == last['ftype']:
            info_group.append(info)
        else:
            info_groups.append(info_group)
            info_group = [info]
    info_groups.append(info_group)

    # turn info_groups into groups of just file names
    groups = []
    for info_group in info_groups:
        groups.append([info['fname'] for info in info_group])

    return groups


# these are just functions because refrences to instance methods can't be pickled,
# so grids couldn't be pickled over to other precesses
def mhd2gse_field_scalar_m1(fld, arr, copy_on_transform=False):  # pylint: disable=unused-argument
    # This is always copied since the -1.0 * arr will need new
    # memory anyway
    a = np.array(arr[:, ::-1, ::-1], copy=False)

    if copy_on_transform:
        if _has_numexpr:
            m1 = np.array([-1.0], dtype=arr.dtype)  # pylint: disable=unused-variable
            a = numexpr.evaluate("a * m1")
        else:
            a = a * -1
    else:
        a *= -1.0
    return a

def mhd2gse_field_scalar(fld, arr, copy_on_transform=False):  # pylint: disable=unused-argument
    # Note: field._data will be set to whatever is returned (after
    # being reshaped to the crd shape), so if you return a view,
    # field._data will be a view
    return np.array(arr[:, ::-1, ::-1], copy=copy_on_transform)

def mhd2gse_field_vector(fld, arr, copy_on_transform=False):
    layout = fld.layout
    if layout == field.LAYOUT_INTERLACED:
        a = np.array(arr[:, ::-1, ::-1, :], copy=False)
        factor = np.array([-1.0, -1.0, 1.0],
                          dtype=arr.dtype).reshape(1, 1, 1, -1)
    elif layout == field.LAYOUT_FLAT:
        a = np.array(arr[:, :, ::-1, ::-1], copy=False)
        factor = np.array([-1.0, -1.0, 1.0],
                          dtype=arr.dtype).reshape(-1, 1, 1, 1)
    else:
        raise RuntimeError("well what am i looking at then...")

    if copy_on_transform:
        if _has_numexpr:
            a = numexpr.evaluate("arr * factor")
        else:
            a = a * factor
    else:
        a *= factor
    return a

def mhd2gse_crds(crds, arr, copy_on_transform=False):  # pylint: disable=unused-argument
    return np.array(-1.0 * arr[::-1], copy=copy_on_transform)


class GGCMGrid(grid.Grid):
    r""" This defines some cool openggcm convinience stuff...
    The following attributes can be set by saying

        ``viscid.grid.readers.openggcm.GGCMGrid.flag = value``.

    This should be done before a call to readers.load_file so that all
    grids that are instantiated have the flags you want.

    Derived quantities accessable by dictionary lookup
        - T: Temperature, for now, just pressure / density
        - bx, by, bz: CC mag field components, if not already stored by
          component)
        - b: mag field as vector, layout affected by
          GGCMGrid.derived_vector_layout
        - v: velocity as vector, same idea as b
        - beta: plasma beta, just pp/b^2
        - psi: flux function (only works for 2d files/grids)

    Attributes:
        mhd_to_gse_on_read (bool, str): flips arrays on load to be in
            GSE crds. If 'auto', then try to use the runtime parameters
            to figure out if conversion is needed; auto requires
            reading the logfile. (default is False)
        copy_on_transform (bool): True means array will be contiguous
            after transform (if one is done), but makes data load
            50\%-60\% slower (default is True)
        force_vector_layout (str): inherited from grid.Grid, enforces
            layout for vector fields on load (default is
            :py:const:`viscid.field.LAYOUT_DEFAULT`)
    """
    _flip_vect_comp_names = "bx, by, b1x, b1y, " \
                            "vx, vy, rv1x, rv1y, " \
                            "jx, jy, ex, ey, ex_cc, ey_cc".split(', ')
    _flip_vect_names = "v, b, j, xj".split(', ')
    # _flip_vect_comp_names = []
    # _flip_vect_names = []

    mhd_to_gse_on_read = False  # True, False, auto, or auto_true
    copy_on_transform = False

    def _do_mhd_to_gse_on_read(self):
        """Return True if we """
        if isinstance(self.mhd_to_gse_on_read, string_types):
            # we already know what this data file needs
            if "_viscid_do_mhd_to_gse_on_read" in self.info:
                return self.info["_viscid_do_mhd_to_gse_on_read"]

            # what are we asking for?
            request = self.mhd_to_gse_on_read.lower()
            if request.startswith("auto"):
                # setup the default
                ret = False
                if request.endswith("true"):
                    ret = True

                # sanity check the logfile stuffs
                log_fname = self.info["_viscid_log_fname"]
                if log_fname == False:
                    raise RuntimeError("If you're using 'auto' for mhd->gse "
                                       "conversion, reading the logfile "
                                       "MUST be turned on.")
                elif log_fname == None:
                    logger.warn("Tried to determine coordinate system using "
                                "logfile parameters, but no logfile found. "
                                "Copy over the log file to use auto mhd->gse "
                                "conversion. (Using default {0})".format(ret))
                else:
                    # ok, we want auto, and we have a logfile so let's figure
                    # out the current layout
                    try:
                        # if we're using a mirdip IC, and low edge is at least
                        # twice smaller than the high edge, then assume
                        # it's a magnetosphere box with xl < 0.0 is the sunward
                        # edge in "MHD" coordinates
                        is_openggcm = (self.info['ggcm_mhd_type'] == "ggcm")
                        # this 2nd check is in case the ggcm_mhd view in the
                        # log file is mangled... this happens sometimes
                        is_openggcm |= (self.info['ggcm_mhd_ic_type'].startswith("mirdip"))
                        xl = float(self.info['mrc_crds_l'][0])
                        xh = float(self.info['mrc_crds_h'][0])
                        if is_openggcm and xl < 0.0 and xh > 0.0 and -2 * xl < xh:
                            ret = True
                    except KeyError as e:
                        raise RuntimeError("Could not determine coordiname system; "
                                           "either the logfile is mangled, or "
                                           "the libmrc options I'm using in infer "
                                           "crd system have changed ({0})"
                                           "".format(e.args[0]))
                self.info["_viscid_do_mhd_to_gse_on_read"] = ret
                return ret
            else:
                raise ValueError("Invalid value for mhd_to_gse_on_read: "
                                 "'{0}'; valid choices: (True, False, auto, "
                                 "auto_true)".format(self.mhd_to_gse_on_read))
            return True
        else:
            return self.mhd_to_gse_on_read

    def set_crds(self, crds_object):
        if self._do_mhd_to_gse_on_read():
            transform_dict = {}
            transform_dict['y'] = mhd2gse_crds
            transform_dict['x'] = mhd2gse_crds
            crds_object.transform_funcs = transform_dict
            crds_object.transform_kwargs = dict(copy_on_transform=self.copy_on_transform)
        super(GGCMGrid, self).set_crds(crds_object)

    def add_field(self, *fields):
        for f in fields:
            if self._do_mhd_to_gse_on_read():
                # what a pain... vector components also need to be flipped
                if f.name in self._flip_vect_comp_names:
                    f.post_reshape_transform_func = mhd2gse_field_scalar_m1
                elif f.name in self._flip_vect_names:
                    f.post_reshape_transform_func = mhd2gse_field_vector
                else:
                    f.post_reshape_transform_func = mhd2gse_field_scalar
                f.transform_func_kwargs = dict(copy_on_transform=self.copy_on_transform)
                f.info["crd_system"] = "gse"
            else:
                f.info["crd_system"] = "mhd"
        super(GGCMGrid, self).add_field(*fields)

    def _get_T(self):
        T = self["pp"] / self["rr"]
        T.name = "T"
        T.pretty_name = "T"
        return T

    def _get_bx(self):
        return self['b'].component_fields()[0]

    def _get_by(self):
        return self['b'].component_fields()[1]

    def _get_bz(self):
        return self['b'].component_fields()[2]

    def _get_vx(self):
        return self['v'].component_fields()[0]

    def _get_vy(self):
        return self['v'].component_fields()[1]

    def _get_vz(self):
        return self['v'].component_fields()[2]

    def _assemble_vector(self, base_name, comp_names="xyz", suffix="",
                         forget_source=True, **kwargs):
        opts = dict(forget_source=forget_source, **kwargs)
        # caching behavior depends on self.longterm_field_caches
        comps = [self[base_name + c + suffix] for c in comp_names]
        return field.scalar_fields_to_vector(base_name, comps, **opts)

    def _get_b(self):
        return self._assemble_vector("b", _force_layout=self.force_vector_layout,
                                     pretty_name="B")

    def _get_v(self):
        return self._assemble_vector("v", _force_layout=self.force_vector_layout,
                                     pretty_name="V")

    def _get_e(self):
        return self._assemble_vector("e", _force_layout=self.force_vector_layout,
                                     pretty_name="E")

    def _get_e_cc(self):
        return self._assemble_vector("e", suffix="_cc",
                                     _force_layout=self.force_vector_layout,
                                     pretty_name="E")

    def _get_j(self):
        return self._assemble_vector("j", _force_layout=self.force_vector_layout,
                                     pretty_name="J")

    @staticmethod
    def _calc_mag(vx, vy, vz):
        if _has_numexpr:
            vmag = numexpr.evaluate("sqrt(vx**2 + vy**2 + vz**2)")
            return vx.wrap(vmag, typ="Scalar")
        else:
            vmag = np.sqrt(vx**2 + vy**2 + vz**2)
            return vmag

    def _get_bmag(self):
        # caching behavior depends on self.longterm_field_caches
        bx, by, bz = self['bx'], self['by'], self['bz']
        bmag = self._calc_mag(bx, by, bz)
        bmag.name = "|B|"
        return bmag

    def _get_jmag(self):
        # caching behavior depends on self.longterm_field_caches
        jx, jy, jz = self['jx'], self['jy'], self['jz']
        jmag = self._calc_mag(jx, jy, jz)
        jmag.name = "|J|"
        return jmag

    def _get_speed(self):
        # caching behavior depends on self.longterm_field_caches
        vx, vy, vz = self['vx'], self['vy'], self['vz']
        speed = self._calc_mag(vx, vy, vz)
        speed.name = "Speed"
        return speed

    def _get_beta(self):
        # caching behavior depends on self.longterm_field_caches
        return plasma.calc_beta(self['pp'], self['b'])

    def _get_psi(self):
        B = self['b']
        # try to guess if a dim of a 3D field is invariant
        if B.nr_sdims > 2:
            slcs = [slice(None)] * B.nr_sdims
            for i, nxi in enumerate(B.sshape):
                if nxi <= 2:
                    slcs[i] = 0
            B = B[slcs]
        return plasma.calc_psi(B)


class GGCMFile(object):
    """Mixin some GGCM convenience stuff

    Attributes:
        read_log_file (bool): search for a log file to load some of the
            libmrc runtime parameters. This does not read parameters
            from all libmrc classes, but can be customized with
            :py:const`viscid.readers.ggcm_logfile.GGCMLogFile.
            watched_classes`. Defaults to False for performance.
    """
    _grid_type = GGCMGrid
    _iono = False

    # this can be set to true if these parameters are needed
    # i thought that reading a log file over sshfs would be a big
    # bottle neck, but it seems opening files over sshfs is appropriately
    # buffered, so maybe it's no big deal since we're only reading the
    # "views" printed at the beginning anyway
    read_log_file = False

    _collection = None
    vfilebucket = None

    def read_logfile(self):
        if self.read_log_file:
            log_basename = "{0}.log".format(self.info['run'])
            # FYI, default max_depth should be 8
            log_fname = find_file_uptree(self.dirname, log_basename)
            if log_fname is None:
                log_fname = find_file_uptree(".", log_basename)
            if log_fname is None:
                log_fname = find_file_uptree(self.dirname, "log.txt")
            if log_fname is None:
                log_fname = find_file_uptree(self.dirname, "log.log")

            if log_fname is not None:
                self.info["_viscid_log_fname"] = log_fname
                if self.vfilebucket is None:
                    self.vfilebucket = VFileBucket()
                log_f = self.vfilebucket.load_file(log_fname,
                                                   file_type=GGCMLogFile,
                                                   index_handle=False)
                # print("!!", log_f)
                self.info.update(log_f.info)
            else:
                # print("**", log_f)
                self.info["_viscid_log_fname"] = None
                logger.warn("You wanted to read parameters from the "
                            "logfile, but I couldn't find one. Maybe "
                            "you need to copy it from somewhere?")
        else:
            self.info["_viscid_log_fname"] = False


class GGCMFileFortran(GGCMFile, vfile.VFile):  # pylint: disable=abstract-method
    _detector = None

    _crds = None
    _fld_templates = None
    grid2 = None

    def __init__(self, fname, vfilebucket=None, crds=None, fld_templates=None,
                 **kwargs):
        if vfilebucket is None:
            vfilebucket = VFileBucket()

        self._crds = crds
        self._fld_templates = fld_templates

        super(GGCMFileFortran, self).__init__(fname, vfilebucket, **kwargs)

    @classmethod
    def group_fnames(cls, fnames):
        """Group File names

        The default implementation just returns fnames, but some file
        types might do something fancy here

        Parameters:
            fnames (list): names that can be logically grouped, as in
                a bunch of file names that are different time steps
                of a given run

        Returns:
            A list of things that can be given to the constructor of
            this class
        """
        return group_ggcm_files_common(cls._detector, fnames)

    def load(self, fname):
        if isinstance(fname, list):
            self._collection = fname
        else:
            self._collection = [fname]

        fname0 = self._collection[0]
        fname1 = self.collective_name(fname)

        # info['run'] is needed for finding the grid2 file
        basename = os.path.basename(fname0)
        self.info['run'] = re.match(self._detector, basename).group(1)
        self.info['fieldtype'] = re.match(self._detector, basename).group(2)

        # HACKY- setting dirname is done in super().load, but we
        # need it to read the log file, which needs to happen before
        # parsing since it sets flags for data transformation and
        # all that stuff
        self.dirname = os.path.dirname(os.path.abspath(fname1))
        self.read_logfile()
        super(GGCMFileFortran, self).load(fname1)

    def _parse(self):
        # look for and parse grid2 file or whatever else needs be done
        if self._crds is None:
            self._crds = self.make_crds()

        if len(self._collection) == 1:
            # load a single file
            _grid = self._parse_file(self.fname)
            self.add(_grid)
            self.activate(0)
        else:
            # load each file, and add it to teh bucket
            data_temporal = dataset.DatasetTemporal("GGCMTemporalCollection")

            self._fld_templates = self._make_template(self._collection[0])

            for fname in self._collection:
                f = self.vfilebucket.load_file(fname, index_handle=False,
                                               file_type=type(self),
                                               crds=self._crds,
                                               fld_templates=self._fld_templates)
                data_temporal.add(f)
            data_temporal.activate(0)
            self.add(data_temporal)
            self.activate(0)

    def make_crds(self):
        if self.info['fieldtype'] == 'iof':
            nlat, nlon = 181, 61
            crdlst = [['lat', [0.0, 180.0, nlat]],
                      ['lon', [0.0, 360.0, nlon]]]
            return wrap_crds("nonuniform_spherical", crdlst,
                             fill_by_linspace=True)

        else:
            return self.read_grid2()

    def read_grid2(self):
        # TODO: iof files can be hacked in here
        grid2_basename = "{0}.grid2".format(self.info['run'])
        self.grid2 = find_file_uptree(self.dirname, grid2_basename)
        if self.grid2 is None:
            self.grid2 = find_file_uptree(".", grid2_basename)
        if self.grid2 is None:
            raise IOError("Could not find a grid2 file for "
                          "{0}".format(self.fname))

        # load the cell centered grid
        with open(self.grid2, 'r') as fin:
            nx = int(next(fin).split()[0])
            gx = list(islice(fin, 0, nx, 1))
            gx = np.array(gx, dtype=float)

            ny = int(next(fin).split()[0])
            gy = list(islice(fin, 0, ny, 1))
            gy = np.array(gy, dtype=float)

            nz = int(next(fin).split()[0])
            gz = list(islice(fin, 0, nz, 1))
            gz = np.array(gz, dtype=float)

        xnc = np.empty(len(gx) + 1)
        ync = np.empty(len(gy) + 1)
        znc = np.empty(len(gz) + 1)

        for cc, nc in [(gx, xnc), (gy, ync), (gz, znc)]:
            hd = 0.5 * (cc[1:] - cc[:-1])
            nc[:] = np.hstack([cc[0] - hd[0], cc[:-1] + hd, cc[-1] + hd[-1]])

        # for 2d files
        crdlst = []
        for dim, nc, cc in zip("zyx", [znc, ync, xnc], [gz, gy, gx]):
            if self.info['fieldtype'].startswith('p'):
                self.info['plane'] = self.info['fieldtype'][1]
                if self.info['plane'] == dim:
                    # in the file name, the value is tenths of an Re
                    planeloc = float(self.info['fieldtype'].split('_')[1]) / 10
                    # FIXME: it is not good to depend on an attribute of
                    # GGCMGrid like this... it could lead to unexpected
                    # behavior if the user starts playing with the value
                    # of an instance, but I'm not sure in practice how or why
                    # since reading the grid should happen fairly early in
                    # the construction process
                    if GGCMGrid.mhd_to_gse_on_read and dim in 'xy':
                        planeloc *= -1
                    self.info['planeloc'] = planeloc
                    ccind = np.argmin(np.abs(cc - planeloc))
                    nc = nc[ccind:ccind + 2]
            crdlst.append([dim, nc])

        return wrap_crds("nonuniform_cartesian", crdlst)

    # def _parse_many(fnames):
    #     pass

    # def _parse_single(fnames):
    #     pass

##
## EOF
##
