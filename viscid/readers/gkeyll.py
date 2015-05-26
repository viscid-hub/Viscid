#!/usr/bin/env python
""" Wrapper grid for some OpenGGCM convenience """

# look at what 'copy_on_transform' actually does... the fact
# that it's here is awkward

from __future__ import print_function, division
import os
import re
from operator import itemgetter

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from viscid.readers.vfile_bucket import ContainerFile
from viscid.readers.hdf5 import FileHDF5, H5pyDataWrapper
from viscid import grid
from viscid import field
from viscid.coordinate import wrap_crds


class GkeyllGrid(grid.Grid):
    """"""
    def _get_bmag(self):
        return NotImplemented


class GkeyllFile(FileHDF5, ContainerFile):  # pylint: disable=abstract-method
    """"""
    _detector = r"^\s*(.*)_(q)_([0-9]+).(h5)\s*$"
    _grid_type = GkeyllGrid

    SAVE_ONLY = False

    _crds = None
    _fld_templates = None

    def __init__(self, fname, crds=None, fld_templates=None, **kwargs):
        assert HAS_H5PY
        self._crds = crds
        self._fld_templates = fld_templates
        super(GkeyllFile, self).__init__(fname, **kwargs)

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
        infolst = []
        for name in fnames:
            m = re.match(cls._detector, name)
            grps = m.groups()
            d = dict(runname=grps[0], ftype=grps[1],
                     fname=m.string)
            try:
                d["frame"] = int(grps[2])
            except (TypeError, ValueError):
                # grps[2] is none for "RUN.3d.xdmf" files
                d["frame"] = -1
            infolst.append(d)

        # careful with sorting, only consecutive files will be grouped
        infolst.sort(key=itemgetter("frame"))
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

    @classmethod
    def collective_name_from_group(cls, fnames):
        fname0 = fnames[0]
        basename = os.path.basename(fname0)
        run = re.match(cls._detector, basename).group(1)
        fldtype = re.match(cls._detector, basename).group(2)
        new_basename = "{0}.{1}.STAR.h5".format(run, fldtype)
        return os.path.join(os.path.dirname(fname0), new_basename)

    def load(self, fname):
        if isinstance(fname, list):
            self._collection = fname
        else:
            self._collection = [fname]

        fname0 = self._collection[0]
        fname1 = self.collective_name(fname)

        basename = os.path.basename(fname0)
        self.set_info('run', re.match(self._detector, basename).group(1))
        self.set_info('fieldtype', re.match(self._detector, basename).group(2))

        super(GkeyllFile, self).load(fname1)

    def _parse(self):
        if len(self._collection) == 1:
            # load a single file
            self._crds = self.make_crds(self.fname)
            _grid = self._parse_file(self.fname, self)
            self.add(_grid)
            self.activate(0)
        else:
            # load each file, and add it to the bucket
            self._crds = self.make_crds(self._collection[0])
            data_temporal = self._make_dataset(self, dset_type="temporal",
                                               name="GkeyllTemporalCollection")
            self._fld_templates = self._make_template(self._collection[0])

            for fname in self._collection:
                f = self._load_child_file(fname, index_handle=False,
                                          file_type=type(self),
                                          crds=self._crds,
                                          fld_templates=self._fld_templates)
                data_temporal.add(f)
            data_temporal.activate(0)
            self.add(data_temporal)
            self.activate(0)

    def _parse_file(self, filename, parent_node):
        # we do minimal file parsing here for performance. we just
        # make data wrappers from the templates we got from the first
        # file in the group, and package them up into grids

        # frame = int(re.match(self._detector, filename).group(3))

        _grid = self._make_grid(parent_node, name="<GkeyllGrid>",
                                **self._grid_opts)

        # FIXME: To get the time at load, we have to open all hdf5 files
        # which defeats the purpose of making templates etc. in attempt to
        # be lazy. Maybe there's a way to use frame above?
        with h5py.File(filename, 'r') as f:
            step = f['timeData'].attrs['vsStep']
            time = f['timeData'].attrs['vsTime']
        self.set_info("step", step)
        self.time = time
        _grid.time = time
        _grid.set_crds(self._crds)

        if self._fld_templates is None:
            self._fld_templates = self._make_template(filename)

        for i, meta in enumerate(self._fld_templates):
            h5_data = H5pyDataWrapper(self.fname, "StructGridField",
                                      comp_dim=-1, comp_idx=i)
            fld = field.wrap_field(h5_data, self._crds, meta['fld_name'])
            _grid.add_field(fld)
        return _grid

    def _make_template(self, filename):
        """"""
        with h5py.File(filename, 'r') as f:
            shape = f["StructGridField"].shape
            sshape = shape[:-1]
            nr_fields = shape[-1]

            template = []
            # TODO: use nr_fields to figure out the names of the fields?
            for i in range(nr_fields):
                d = dict(fld_name='f' + str(i), shape=sshape, fldnum=i)
                template.append(d)

            # self.set_info("all sorts of meta data?", value)
        return template

    def _get_single_crd(self, h5file, crd):
        idx = "xyz".index(crd)
        gridType = h5file['StructGrid'].attrs['vsKind']
        if gridType in ['uniform']:
            if idx >= len(h5file['StructGrid'].attrs['vsNumCells']):
                raise IndexError()
            lower = h5file['StructGrid'].attrs['vsLowerBounds'][idx]
            upper = h5file['StructGrid'].attrs['vsUpperBounds'][idx]
            num = h5file['StructGrid'].attrs['vsNumCells'][idx]
            return [lower, upper, num]
        elif gridType in ['structured']:
            raise NotImplementedError()

    def make_crds(self, fname):
        with h5py.File(fname, 'r') as f:
            clist = []

            # FIXME: xyz
            for crd in "xyz":
                try:
                    clist.append((crd, self._get_single_crd(f, crd)))
                except IndexError:
                    pass

            if f['StructGrid'].attrs['vsKind'] in ['uniform']:
                crds = wrap_crds("uniform_cartesian", clist)
            else:
                raise NotImplementedError()

            return crds

##
## EOF
##
