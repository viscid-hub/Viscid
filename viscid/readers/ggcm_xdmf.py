import os
import re

from viscid import dataset
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

    def __init__(self, *args, **kwargs):
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
        super(GGCMFileXDMF, self).load(fname[0])

        basename = os.path.basename(self.fname)
        self.info['run'] = re.match(self._detector, basename).group(1)

        # look for a log file to auto-load some parameters about the run
        self.read_logfile()

    def _parse(self):
        if self._collection is not None:
            # assume we have a collection of temporal files, because why not
            data_temporal = dataset.DatasetTemporal("GGCMXDMFTemporalCollection")
            for fname in self._collection:
                grids = self._parse_file(fname)
                for _grid in grids:
                    data_temporal.add(_grid)
            data_temporal.activate(0)
            self.add(data_temporal)
            self.activate(0)
        else:
            super(GGCMFileXDMF, self)._parse()


class GGCMIonoFileXDMF(GGCMFileXDMF):  # pylint: disable=abstract-method
    """Jimmy's run length encoding files"""
    _detector = r"^\s*(.*)\.(iof)(?:\.([0-9]{6}))?\.(xmf|xdmf)\s*$"
    _iono = True
    _grid_type = grid.Grid


class GGCMAncFileXDMF(GGCMFileXDMF):  # pylint: disable=abstract-method
    """Jimmy's run length encoding files"""
    _detector = r"^\s*(.*)\.(mp_info)(?:\.([0-9]{6}))?\.(xmf|xdmf)\s*$"
    _grid_type = grid.Grid
