import os
import re

from viscid.readers import xdmf
from viscid.readers import athena


class AthenaFileXDMF(athena.AthenaFile, xdmf.FileXDMF):  # pylint: disable=abstract-method
    """File type for Athena style convenience stuff

    Makes AthenaGrid the default grid and handles grouping multiple
    files.
    """
    _detector = r"^\s*(.*)\.([0-9]+)\.(ath.xdmf)\s*$"

    _collection = None

    @classmethod
    def group_fnames(cls, fnames):
        return athena.group_athena_files_common(cls._detector, fnames)

    @classmethod
    def collective_name_from_group(cls, fnames):
        return athena.athena_collective_name_from_group(cls._detector,
                                                        fnames)

    def load(self, fname):
        if not isinstance(fname, list):
            fname = [fname]

        if len(fname) > 1:
            self._collection = fname
        else:
            self._collection = None
        super(AthenaFileXDMF, self).load(fname[0])

        basename = os.path.basename(self.fname)
        self.set_info('run', re.match(self._detector, basename).group(1))

    def _parse(self):
        if self._collection is not None:
            # assume we have a collection of temporal files, because why not
            data_temporal = self._make_dataset(self, dset_type="temporal",
                                               name="AthenaXDMFTemporalCollection")

            for fname in self._collection:
                grids = self._parse_file(fname, data_temporal)
                for _grid in grids:
                    data_temporal.add(_grid)
            data_temporal.activate(0)
            self.add(data_temporal)
            self.activate(0)
        else:
            super(AthenaFileXDMF, self)._parse()

##
## EOF
##
