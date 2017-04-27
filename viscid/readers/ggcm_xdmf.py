import os
import re

from viscid import grid
from viscid.readers import xdmf
from viscid.readers import openggcm

class GGCMFileXDMF(openggcm.GGCMFile, xdmf.FileXDMF):  # pylint: disable=abstract-method
    """File type for GGCM style convenience stuff

    Attributes:
        read_log_file (bool): search for a log file to load some of the
            libmrc runtime parameters. This does not read parameters
            from all libmrc classes, but can be customized with
            :py:const`viscid.readers.ggcm_logfile.GGCMLogFile.
            watched_classes`. Defaults to False for performance.
    """
    _detector = r"^\s*(.*)\.(p[xyz]_[0-9]+|3d|3df)" \
                r"(?:\.([0-9]{6}))?\.(xmf|xdmf)\s*$"

    # you can override this in your viscidrc file with:
    #   "readers.ggcm_xdmf.GGCMFileXDMF.assume_mhd_crds": true
    # BUT, usually XDMF files have enough meta data so this isn't necessary,
    # and then one day, you'll be trying to load a non-magnetosphere file,
    # and be confused when the axes are flipped
    assume_mhd_crds = False


    def __init__(self, *args, **kwargs):
        if 'assume_mhd_crds' in kwargs:
            self.assume_mhd_crds = kwargs['assume_mhd_crds']
        else:
            kwargs['assume_mhd_crds'] = self.assume_mhd_crds

        super(GGCMFileXDMF, self).__init__(*args, **kwargs)

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
        return openggcm.group_ggcm_files_common(cls._detector, fnames)

    @classmethod
    def collective_name_from_group(cls, fnames):
        fname0 = fnames[0]
        basename = os.path.basename(fname0)
        run = re.match(cls._detector, basename).group(1)
        fldtype = re.match(cls._detector, basename).group(2)
        new_basename = "{0}.{1}.STAR.xdmf".format(run, fldtype)
        return os.path.join(os.path.dirname(fname0), new_basename)

    def load(self, fname):
        if not isinstance(fname, list):
            fname = [fname]

        if len(fname) > 1:
            self._collection = fname
        else:
            self._collection = None

        # HACKY- setting dirname is done in super().load, but we
        # need it to read the log file, which needs to happen before
        # parsing since it sets flags for data transformation and
        # all that stuff
        _fname = os.path.expanduser(os.path.expandvars(fname[0]))
        basename = os.path.basename(_fname)
        self.set_info('run', re.match(self._detector, basename).group(1))
        self.dirname = os.path.dirname(os.path.abspath(_fname))
        self.read_logfile()

        super(GGCMFileXDMF, self).load(fname[0])

    def _parse(self):
        if self._collection is not None:
            # assume we have a collection of temporal files, because why not
            data_temporal = self._make_dataset(self, dset_type="temporal",
                                               name="GGCMXDMFTemporalCollection")

            for fname in self._collection:
                self._parse_file(fname, data_temporal)
                # grids = self._parse_file(fname, data_temporal)
                # for _grid in grids:
                #     data_temporal.add(_grid)
            data_temporal.activate(0)
            self.add(data_temporal)
            self.activate(0)
        else:
            super(GGCMFileXDMF, self)._parse()

        # now this is a wicked hack :(
        if self.find_info('basetime', default=None) is None:
            try:
                timestr = self._child_files[0].find_item('openggcm*/time_str')
                basetime, _ = self.parse_timestring(timestr.decode())
            except (KeyError, ValueError):
                # if no basetime found, then use the special no dip tilt time
                # this is a selfish hack in that old files that I saved
                # will usually use this as basetime
                basetime = openggcm.GGCM_NO_DIPTILT

            self.set_info("basetime", basetime)


class GGCMIonoFileXDMF(GGCMFileXDMF):  # pylint: disable=abstract-method
    """Ionosphere Files"""
    _detector = r"^\s*(.*)\.(iof)(?:\.([0-9]{6}))?\.(xmf|xdmf)\s*$"
    _iono = True
    _grid_type = grid.Grid


class GGCMAncFileXDMF(GGCMFileXDMF):  # pylint: disable=abstract-method
    """Ancillary files; usually, these files have already been
    converted to GSE"""
    _detector = r"^\s*(.*)\.(mp_info|topo)(?:\.([0-9]{6}))?\.(xmf|xdmf)\s*$"
    _grid_type = grid.Grid
