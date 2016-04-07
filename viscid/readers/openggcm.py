#!/usr/bin/env python
""" Wrapper grid for some OpenGGCM convenience """

# look at what 'copy_on_transform' actually does... the fact
# that it's here is awkward

from __future__ import print_function, division
import os
import re
from itertools import islice
from operator import itemgetter
from datetime import datetime

import numpy as np
try:
    import numexpr
    _has_numexpr = True
except ImportError:
    _has_numexpr = False

from viscid import logger
from viscid.compat import string_types
from viscid.readers.vfile_bucket import ContainerFile
from viscid.readers.ggcm_logfile import GGCMLogFile
# from viscid.dataset import Dataset, DatasetTemporal
# from viscid.vutil import time_as_datetime, time_as_timedelta
from viscid import vutil
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
        raise RuntimeError("this is non-sensicle: {0}".format(directory))

    fname = os.path.join(directory, basename)
    # log_fname = "{0}/{1}.log".format(d, self.find_info("run"))
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
        except (TypeError, ValueError):
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
def mhd2gse_field_scalar_m1(fld, crds, arr, comp_slc=None,
                            copy_on_transform=False):  # pylint: disable=unused-argument
    # This is always copied since the -1.0 * arr will need new
    # memory anyway
    # a = np.array(arr[:, ::-1, ::-1], copy=False)
    if copy_on_transform:
        if _has_numexpr:
            m1 = np.array([-1.0], dtype=arr.dtype)  # pylint: disable=unused-variable
            arr = numexpr.evaluate("arr * m1")
        else:
            arr = arr * -1
    else:
        arr *= -1.0
    return arr

def mhd2gse_field_scalar(fld, crds, arr, comp_slc=None,
                         copy_on_transform=False):
    # Note: field._data will be set to whatever is returned (after
    # being reshaped to the crd shape), so if you return a view,
    # field._data will be a view
    if copy_on_transform:
        return np.array(arr, copy=True)
    else:
        return arr

def mhd2gse_field_vector(fld, crds, arr, comp_slc=None,
                         copy_on_transform=False):
    layout = fld.layout
    shp = [1] * len(crds.shape)
    if layout == field.LAYOUT_INTERLACED:
        shp.append(-1)
    elif layout == field.LAYOUT_FLAT:
        shp.insert(0, -1)
    else:
        raise RuntimeError("well what am i looking at then...")
    factor = np.array([-1.0, -1.0, 1.0], dtype=arr.dtype).reshape(shp)
    if comp_slc is not None:
        tmpslc = [slice(None) if s == 1 else comp_slc for s in shp]
        factor = factor[tmpslc]
    # print("** factor", factor.shape, factor.flatten())
    if copy_on_transform:
        if _has_numexpr:
            arr = numexpr.evaluate("arr * factor")
        else:
            arr = arr * factor
    else:
        arr *= factor
    return arr

# def mhd2gse_crds(crds, arr, copy_on_transform=False):  # pylint: disable=unused-argument
#     # print("transforming crds")
#     raise RuntimeError("This functionality is now in crds")
#     return np.array(-1.0 * arr[::-1], copy=copy_on_transform)


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
                            "jx, jy, xjx, xjy, " \
                            "ex, ey, ex_cc, ey_cc".split(', ')
    _flip_vect_names = "v, b, j, xj".split(', ')
    # _flip_vect_comp_names = []
    # _flip_vect_names = []

    mhd_to_gse_on_read = False  # True, False, auto, or auto_true
    copy_on_transform = False

    def _do_mhd_to_gse_on_read(self):
        """Return True if we """
        if isinstance(self.mhd_to_gse_on_read, string_types):
            # we already know what this data file needs
            if self.has_info("_viscid_do_mhd_to_gse_on_read"):
                return self.find_info("_viscid_do_mhd_to_gse_on_read")

            # what are we asking for?
            request = self.mhd_to_gse_on_read.lower()
            if request.startswith("auto"):
                # setup the default
                ret = False
                if request.endswith("true"):
                    ret = True

                # sanity check the logfile stuffs
                log_fname = self.find_info("_viscid_log_fname")
                if log_fname is False:
                    raise RuntimeError("If you're using 'auto' for mhd->gse "
                                       "conversion, reading the logfile "
                                       "MUST be turned on.")
                elif log_fname is None:
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
                        is_openggcm = (self.find_info('ggcm_mhd_type') == "ggcm")
                        # this 2nd check is in case the ggcm_mhd view in the
                        # log file is mangled... this happens sometimes
                        ic_type = self.find_info('ggcm_mhd_ic_type')
                        is_openggcm |= (ic_type.startswith("mirdip"))
                        xl = float(self.find_info('mrc_crds_l')[0])
                        xh = float(self.find_info('mrc_crds_h')[0])
                        if is_openggcm and xl < 0.0 and xh > 0.0 and -2 * xl < xh:
                            ret = True
                    except KeyError as e:
                        logger.warn("Could not determine coordiname system; "
                                    "either the logfile is mangled, or "
                                    "the libmrc options I'm using in infer "
                                    "crd system have changed (%s)", e.args[0])
                self.set_info("_viscid_do_mhd_to_gse_on_read", ret)
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
            # transform_dict = {}
            # transform_dict['y'] = mhd2gse_crds
            # transform_dict['x'] = mhd2gse_crds
            # crds_object.transform_funcs = transform_dict
            # crds_object.transform_kwargs = dict(copy_on_transform=self.copy_on_transform)
            crds_object.reflect_axes = "xy"
        super(GGCMGrid, self).set_crds(crds_object)

    def add_field(self, *fields):
        for f in fields:
            if self._do_mhd_to_gse_on_read():
                # what a pain... vector components also need to be flipped
                if isinstance(f, field.VectorField):
                    f.post_reshape_transform_func = mhd2gse_field_vector
                elif f.name in self._flip_vect_comp_names:
                    f.post_reshape_transform_func = mhd2gse_field_scalar_m1
                elif f.name in self._flip_vect_names:
                    raise NotImplementedError("this shouldn't happen")
                    # f.post_reshape_transform_func = mhd2gse_field_vector
                else:
                    f.post_reshape_transform_func = mhd2gse_field_scalar
                f.transform_func_kwargs = dict(copy_on_transform=self.copy_on_transform)
                f.meta["crd_system"] = "gse"
            else:
                f.meta["crd_system"] = "mhd"
        super(GGCMGrid, self).add_field(*fields)

    def _get_T(self):
        T = self["pp"] / self["rr"]
        T.name = "T"
        T.pretty_name = "T"
        return T

    def _get_bx(self):
        return self['b']['x']

    def _get_by(self):
        return self['b']['y']

    def _get_bz(self):
        return self['b']['z']

    def _get_vx(self):
        return self['v']['x']

    def _get_vy(self):
        return self['v']['y']

    def _get_vz(self):
        return self['v']['z']

    def _get_jx(self):
        return self['xjx']

    def _get_jy(self):
        return self['xjy']

    def _get_jz(self):
        return self['xjz']

    def _assemble_vector(self, base_name, comp_names="xyz", suffix="",
                         forget_source=False, **kwargs):
        opts = dict(forget_source=forget_source, **kwargs)
        # caching behavior depends on self.longterm_field_caches
        comps = [self[base_name + c + suffix] for c in comp_names]
        return field.scalar_fields_to_vector(comps, name=base_name, **opts)

    def _get_b(self):
        return self._assemble_vector("b", _force_layout=self.force_vector_layout,
                                     pretty_name="B")

    def _get_b1(self):
        return self._assemble_vector("b1", _force_layout=self.force_vector_layout,
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

    def _get_e_ec(self):
        return self._assemble_vector("e", suffix="_ec",
                                     _force_layout=self.force_vector_layout,
                                     pretty_name="E")

    def _get_j(self):
        return self._assemble_vector("j", _force_layout=self.force_vector_layout,
                                     pretty_name="J")

    def _get_xj(self):
        return self._assemble_vector("j", _force_layout=self.force_vector_layout,
                                     pretty_name="J")

    @staticmethod
    def _calc_mag(vx, vy, vz):
        if _has_numexpr:
            vmag = numexpr.evaluate("sqrt(vx**2 + vy**2 + vz**2)")
            return vx.wrap(vmag, fldtype="Scalar")
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
        speed.name = "speed"
        speed.pretty_name = "|V|"
        return speed

    def _get_beta(self):
        # caching behavior depends on self.longterm_field_caches
        return plasma.calc_beta(self['pp'], self['b'], scale=40.0)

    def _get_psi(self):
        B = self['b']

        rev = True if B.meta["crd_system"] == "gse" else False
        psi = plasma.calc_psi(B, rev=rev)
        return psi


class GGCMFile(object):  # pylint: disable=abstract-method
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

    _already_warned_about_logfile = False

    # this can be set to true if these parameters are needed
    # i thought that reading a log file over sshfs would be a big
    # bottle neck, but it seems opening files over sshfs is appropriately
    # buffered, so maybe it's no big deal since we're only reading the
    # "views" printed at the beginning anyway
    read_log_file = False

    _collection = None

    def read_logfile(self):
        if self.read_log_file:
            log_basename = "{0}.log".format(self.find_info('run'))
            # FYI, default max_depth should be 8
            log_fname = find_file_uptree(self.dirname, log_basename)
            if log_fname is None:
                log_fname = find_file_uptree(".", log_basename)
            if log_fname is None:
                log_fname = find_file_uptree(self.dirname, "log.txt")
            if log_fname is None:
                log_fname = find_file_uptree(self.dirname, "log.log")
            if log_fname is None:
                log_fname = find_file_uptree(self.dirname, "log")

            if log_fname is not None:
                self.set_info("_viscid_log_fname", log_fname)

                with GGCMLogFile(log_fname) as log_f:
                    # self._info.update(log_f.info)
                    for key, val in log_f.info.items():
                        self.update_info(key, val)
            else:
                # print("**", log_f)
                self.set_info("_viscid_log_fname", None)
                if not GGCMFile._already_warned_about_logfile:
                    logger.warn("You wanted to read parameters from the "
                                "logfile, but I couldn't find one. Maybe "
                                "you need to copy it from somewhere?")
                    GGCMFile._already_warned_about_logfile = True
        else:
            self.set_info("_viscid_log_fname", False)

    def _get_dipoletime_as_datetime(self):
        try:
            # FIXME: this should be STARTTIME, but that's not part
            # of a libmrc object
            diptime = ":".join(self.find_info('ggcm_dipole_dipoltime'))
            dipoletime = datetime.strptime(diptime, "%Y:%m:%d:%H:%M:%S.%f")
        except KeyError:
            raise KeyError("Can't use UT times without reading "
                           "a logfile that has a value for "
                           "ggcm_dipole_dipoltime")
        except ValueError:
            raise ValueError("Dipoletime from the logfile is mangled "
                             "({0}). You may need to fix the log file"
                             "by hand.".format(diptime))
        return dipoletime

    def _sub_translate_time(self, time):
        if isinstance(time, string_types):
            try:
                time = datetime.strptime(time, "UT%Y:%m:%d:%H:%M:%S.%f")
            except ValueError:
                pass

        if isinstance(time, datetime):
            if time.year < 1967:
                # times < 1967 are not valid GGCM UT times, so pass them to
                # the super class
                return NotImplemented
            dipoletime = self._get_dipoletime_as_datetime()
            # Why was this relative to time 0?
            # delta = dipoletime - datetime.strptime("00", "%S")
            delta = time - dipoletime
            return delta.total_seconds()

        return NotImplemented

    def _sub_format_time(self, time, style=".02f"):
        """
        Args:
            t (float): time
            info (dict): info dict
            style (str): for this method, can be anything
                :func:`viscid.vutil.format_time` can understand, or
                'UT' to print a UT time

        Returns:
            string or NotImplemented if style is not understood
        """
        if style.lower().startswith("ut"):
            style = style[2:].strip()
            dipoletime = self._get_dipoletime_as_datetime()
            ut_time = dipoletime + self.time_as_timedelta(time)
            return vutil.format_time(ut_time, style)
        else:
            return NotImplemented

    def _sub_time_as_datetime(self, time, epoch):
        try:
            dipoletime = self._get_dipoletime_as_datetime()
            ut_time = dipoletime + self.time_as_timedelta(time)
            return ut_time
        except (KeyError, ValueError):
            pass
        return NotImplemented


class GGCMFileFortran(GGCMFile, ContainerFile):  # pylint: disable=abstract-method
    """An abstract class from which jrrle and fortbin files are derived

    Note:
        All subclasses should implement a _shape_discovery_hack
    """
    _detector = None

    _crds = None
    _fld_templates = None
    grid2 = None

    def __init__(self, fname, crds=None, fld_templates=None, **kwargs):
        self._crds = crds
        self._fld_templates = fld_templates
        super(GGCMFileFortran, self).__init__(fname, **kwargs)

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
        self.set_info('run', re.match(self._detector, basename).group(1))
        self.set_info('fieldtype', re.match(self._detector, basename).group(2))

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
            _grid = self._parse_file(self.fname, self)
            self.add(_grid)
            self.activate(0)
        else:
            # load each file, and add it to the bucket
            data_temporal = self._make_dataset(self, dset_type="temporal",
                                               name="GGCMTemporalCollection")

            self._fld_templates = self._make_template(self._collection[0])

            for fname in self._collection:
                f = self._load_child_file(fname, index_handle=False,
                                          file_type=type(self),
                                          grid_type=self._grid_type,
                                          crds=self._crds,
                                          fld_templates=self._fld_templates)
                data_temporal.add(f)
            data_temporal.activate(0)
            self.add(data_temporal)
            self.activate(0)

    def make_crds(self):
        if self.get_info('fieldtype') == 'iof':
            # 181, 61
            nlon, nlat = self._shape_discovery_hack(self._collection[0])
            crdlst = [['lon', [0.0, 360.0, nlon]],
                      ['lat', [0.0, 180.0, nlat]]]
            return wrap_crds("uniform_spherical", crdlst)

        else:
            return self.read_grid2()

    def _shape_discovery_hack(self, filename):
        """Used if we can't get data's shape from grid2 file

        In short, for subclasses, they should read the metadata for
        the first field and return the data shape. This is not elegant,
        and it should be implemented by all subclasses.

        Returns:
            tuple of ints, shape of a field in the file
        """
        raise NotImplementedError()

    def read_grid2(self):
        # TODO: iof files can be hacked in here
        grid2_basename = "{0}.grid2".format(self.find_info('run'))
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
            gx = np.array(gx, dtype='f4')

            ny = int(next(fin).split()[0])
            gy = list(islice(fin, 0, ny, 1))
            gy = np.array(gy, dtype='f4')

            nz = int(next(fin).split()[0])
            gz = list(islice(fin, 0, nz, 1))
            gz = np.array(gz, dtype='f4')

        xnc = np.empty(len(gx) + 1, dtype=gx.dtype)
        ync = np.empty(len(gy) + 1, dtype=gy.dtype)
        znc = np.empty(len(gz) + 1, dtype=gz.dtype)

        for cc, nc in [(gx, xnc), (gy, ync), (gz, znc)]:
            hd = 0.5 * (cc[1:] - cc[:-1])
            nc[:] = np.hstack([cc[0] - hd[0], cc[:-1] + hd, cc[-1] + hd[-1]])

        # for 2d files
        crdlst = []
        for dim, nc, cc in zip("xyz", [xnc, ync, znc], [gx, gy, gz]):
            fieldtype = self.get_info('fieldtype')
            if fieldtype.startswith('p'):
                self.set_info('plane', fieldtype[1])
                if fieldtype[1] == dim:
                    planeloc = float(fieldtype.split('_')[1])
                    planeloc /= 10  # value is in tenths of Re
                    # FIXME: it is not good to depend on an attribute of
                    # GGCMGrid like this... it could lead to unexpected
                    # behavior if the user starts playing with the value
                    # of an instance, but I'm not sure in practice how or why
                    # since reading the grid should happen fairly early in
                    # the construction process
                    if GGCMGrid.mhd_to_gse_on_read and dim in 'xy':
                        planeloc *= -1
                    self.set_info('planeloc', planeloc)
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
