#!/usr/bin/env python
""" Wrapper grid for some OpenGGCM convenience """

# look at what 'copy_on_transform' actually does... the fact
# that it's here is awkward

from __future__ import print_function, division
import os
import re
from operator import itemgetter

import numpy as np

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


base_hydro_names = ['rho', 'rhoux', 'rhouy', 'rhouz', 'e']
base_hydro_pretty_names = [r'$\rho$', r'$\rho u_x$', r'$\rho u_y$', r'$\rho u_z$',
                           r'$e$']

base_5m_names = ['rho_e', 'rhoux_e', 'rhouy_e', 'rhouz_e', 'e_e',
                 'rho_i', 'rhoux_i', 'rhouy_i', 'rhouz_i', 'e_i',
                 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']
base_5m_pretty_names = [
    r'$\rho_e$', r'$\rho u_{x,e}$', r'$\rho u_{y,e}$', r'$\rho u_{z,e}$', r'$e_e$',
    r'$\rho_i$', r'$\rho u_{x,i}$', r'$\rho u_{y,i}$', r'$\rho u_{z,i}$', r'$e_i$',
    r'$E_x$', r'$E_y$', r'$E_z$', r'$B_x$', r'$B_y$', r'$B_z$']

base_10m_names = ['rho_e', 'rhoux_e', 'rhouy_e', 'rhouz_e',
                  'pxx_e', 'pxy_e', 'pxz_e', 'pyy_e', 'pyz_e', 'pzz_e',
                  'rho_i', 'rhoux_i', 'rhouy_i', 'rhouz_i',
                  'pxx_i', 'pxy_i', 'pxz_i', 'pyy_i', 'pyz_i', 'pzz_i',
                  'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']
base_10m_pretty_names = [
    r'$\rho_e$', r'$\rho u_{x,e}$', r'$\rho u_{y,e}$', r'$\rho u_{z,e}$',
    r'$\mathcal{P}_{xx,e}$', r'$\mathcal{P}_{xy,e}$', r'$\mathcal{P}_{xz,e}$',
    r'$\mathcal{P}_{yy,e}$', r'$\mathcal{P}_{yz,e}$', r'$\mathcal{P}_{zz,e}$',
    r'$\rho_i$', r'$\rho u_{x,i}$', r'$\rho u_{y,i}$', r'$\rho u_{z,i}$',
    r'$\mathcal{P}_{xx,i}$', r'$\mathcal{P}_{xy,i}$', r'$\mathcal{P}_{xz,i}$',
    r'$\mathcal{P}_{yy,i}$', r'$\mathcal{P}_{yz,i}$', r'$\mathcal{P}_{zz,i}$',
    r'$E_x$', r'$E_y$', r'$E_z$', r'$B_x$', r'$B_y$', r'$B_z$']

_type_info = {len(base_hydro_names): {'field_type': 'hydro',
                                      'names': base_hydro_names,
                                      'pretty_names': base_hydro_pretty_names},
              len(base_5m_names): {'field_type': 'five-moment',
                                   'names': base_5m_names,
                                   'pretty_names': base_5m_pretty_names},
              len(base_10m_names): {'field_type': 'ten-moment',
                                    'names': base_10m_names,
                                    'pretty_names': base_10m_pretty_names}}

def create_type_info(nr_fields):
    type_info = dict()
    type_info['field_type'] = 'unknown'
    type_info['names'] = []
    type_info['pretty_names'] = []
    for ncomp in range(nr_fields):
        type_info['names'].append('Q%d'%ncomp)
        type_info['pretty_names'].append(r'$\rm{Q%d}$'%ncomp)
    return type_info

class GkeyllGrid(grid.Grid):
    """"""
    def _assemble_vector(self, base_name, comp_names="xyz", suffix="",
                         forget_source=False, **kwargs):
        opts = dict(forget_source=forget_source, **kwargs)
        # caching behavior depends on self.longterm_field_caches
        comps = [self[base_name + c + suffix] for c in comp_names]
        return field.scalar_fields_to_vector(comps, name=base_name, **opts)

    def _get_ux_e(self):
        ux_e = self['rhoux_e'] / self['rho_e']
        ux_e.name = 'ux_e'
        ux_e.pretty_name = r'$u_{x,e}$'
        return ux_e

    def _get_uy_e(self):
        uy_e = self['rhouy_e'] / self['rho_e']
        uy_e.name = 'uy_e'
        uy_e.pretty_name = r'$u_{y,e}$'
        return uy_e

    def _get_uz_e(self):
        uz_e = self['rhouz_e'] / self['rho_e']
        uz_e.name = 'uz_e'
        uz_e.pretty_name = r'$u_{z,e}$'
        return uz_e

    def _get_Pxx_e(self):
        Pxx_e = self['pxx_e'] - self['rhoux_e']*self['rhoux_e'] / self['rho_e']
        Pxx_e.name = 'Pxx_e'
        Pxx_e.pretty_name = r'$P_{xx,e}$'
        return Pxx_e

    def _get_Pyy_e(self):
        Pyy_e = self['pyy_e'] - self['rhoux_e']*self['rhoux_e'] / self['rho_e']
        Pyy_e.name = 'Pyy_e'
        Pyy_e.pretty_name = r'$P_{yy,e}$'
        return Pyy_e

    def _get_Pzz_e(self):
        Pzz_e = self['pzz_e'] - self['rhoux_e']*self['rhoux_e'] / self['rho_e']
        Pzz_e.name = 'Pzz_e'
        Pzz_e.pretty_name = r'$P_{zz,e}$'
        return Pzz_e

    def _get_Pxy_e(self):
        Pxy_e = self['pxy_e'] - self['rhoux_e']*self['rhouy_e'] / self['rho_e']
        Pxy_e.name = 'Pxy_e'
        Pxy_e.pretty_name = r'$P_{xy,e}$'
        return Pxy_e

    def _get_Pxz_e(self):
        Pxz_e = self['pxz_e'] - self['rhoux_e']*self['rhouz_e'] / self['rho_e']
        Pxz_e.name = 'Pxz_e'
        Pxz_e.pretty_name = r'$P_{xz,e}$'
        return Pxz_e

    def _get_Pyz_e(self):
        Pyz_e = self['pyz_e'] - self['rhouy_e']*self['rhouz_e'] / self['rho_e']
        Pyz_e.name = 'Pyz_e'
        Pyz_e.pretty_name = r'$P_{yz,e}$'
        return Pyz_e

    def _get_ux_i(self):
        ux_i = self['rhoux_i'] / self['rho_i']
        ux_i.name = 'ux_i'
        ux_i.pretty_name = r'$u_{x,i}$'
        return ux_i

    def _get_uy_i(self):
        uy_i = self['rhouy_i'] / self['rho_i']
        uy_i.name = 'uy_i'
        uy_i.pretty_name = r'$u_{y,i}$'
        return uy_i

    def _get_uz_i(self):
        uz_i = self['rhouz_i'] / self['rho_i']
        uz_i.name = 'uz_i'
        uz_i.pretty_name = r'$u_{z,i}$'
        return uz_i

    def _get_Pxx_i(self):
        Pxx_i = self['pxx_i'] - self['rhoux_i']*self['rhoux_i'] / self['rho_i']
        Pxx_i.name = 'Pxx_i'
        Pxx_i.pretty_name = r'$P_{xx,i}$'
        return Pxx_i

    def _get_Pyy_i(self):
        Pyy_i = self['pyy_i'] - self['rhoux_i']*self['rhoux_i'] / self['rho_i']
        Pyy_i.name = 'Pyy_i'
        Pyy_i.pretty_name = r'$P_{yy,i}$'
        return Pyy_i

    def _get_Pzz_i(self):
        Pzz_i = self['pzz_i'] - self['rhoux_i']*self['rhoux_i'] / self['rho_i']
        Pzz_i.name = 'Pzz_i'
        Pzz_i.pretty_name = r'$P_{zz,i}$'
        return Pzz_i

    def _get_Pxy_i(self):
        Pxy_i = self['pxy_i'] - self['rhoux_i']*self['rhouy_i'] / self['rho_i']
        Pxy_i.name = 'Pxy_i'
        Pxy_i.pretty_name = r'$P_{xy,i}$'
        return Pxy_i

    def _get_Pxz_i(self):
        Pxz_i = self['pxz_i'] - self['rhoux_i']*self['rhouz_i'] / self['rho_i']
        Pxz_i.name = 'Pxz_i'
        Pxz_i.pretty_name = r'$P_{xz,i}$'
        return Pxz_i

    def _get_Pyz_i(self):
        Pyz_i = self['pyz_i'] - self['rhouy_i']*self['rhouz_i'] / self['rho_i']
        Pyz_i.name = 'Pyz_i'
        Pyz_i.pretty_name = r'$P_{yz,i}$'
        return Pyz_i

    def _get_u_e(self):
        # get from [ux_e, uy_e, uz_e]
        return self._assemble_vector("u", suffix='_e',
                                        _force_layout=self.force_vector_layout,
                                         pretty_name='u_e')

    def _get_u_i(self):
        # get from [ux_i, uy_i, uz_i]
        return self._assemble_vector("u", suffix='_i',
                                        _force_layout=self.force_vector_layout,
                                         pretty_name='u_i')

    def _get_b(self):
        # get from [Bx, By, Bz]
        return self._assemble_vector("B", _force_layout=self.force_vector_layout,
                                         pretty_name="b")

class GkeyllFile(FileHDF5, ContainerFile):  # pylint: disable=abstract-method
    """"""
    _detector = r"^\s*(.*)_(q)_([0-9]+).(h5)\s*$"
    _grid_type = GkeyllGrid

    SAVE_ONLY = False

    _fwrapper = None
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

    def get_file_wrapper(self, filename):
        if self._fwrapper is None:
            # self._fwrapper = GGCMFortbinFileWrapper(filename)
            return h5py.File(filename, 'r')
        else:
            raise NotImplementedError()
        #     assert (self._fwrapper.filename == filename or
        #             glob2(self._fwrapper.filename) == glob2(filename))
        # return self._fwrapper

    def set_file_wrapper(self, wrapper):
        raise NotImplementedError("This must be done at file init")

    def load(self, fname):
        if isinstance(fname, list):
            self._collection = fname
        else:
            self._collection = [fname]

        fname0 = self._collection[0]
        fname1 = self.collective_name(fname)

        basename = os.path.basename(fname0)
        self.set_info('run', re.match(self._detector, basename).group(1))
        self.set_info('field_type', re.match(self._detector, basename).group(2))

        super(GkeyllFile, self).load(fname1)

    def _parse(self):
        if len(self._collection) == 1:
            # load a single file
            if self._crds is None:
                self._crds = self.make_crds(self.fname)
            _grid = self._parse_file(self.fname, self)
            self.add(_grid)
            self.activate(0)
        else:
            # load each file, and add it to the bucket
            if self._crds is None:
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
            # FIXME: the transpose goes xyz -> zyx
            h5_data = H5pyDataWrapper(self.fname, "StructGridField",
                                      comp_dim=-1, comp_idx=i)
            fld = field.wrap_field(h5_data, self._crds, meta['fld_name'],
                                   center="cell", pretty_name=meta['pretty_name'])
            _grid.add_field(fld)
        _grid.set_info('field_type', self.get_info('field_type'))
        return _grid

    def _make_template(self, filename):
        """"""
        with self.get_file_wrapper(filename) as f:
            shape = f["StructGridField"].shape
            sshape = shape[:-1]
            nr_fields = shape[-1]

            try:
                type_info = _type_info[nr_fields]
            except KeyError:
                try:
                    # if the two scalar corrections potentials are stored
                    type_info = dict(_type_info[nr_fields - 2])
                    type_info['names'].append('phi_E')
                    type_info['pretty_names'].append(r'$\Phi_E$')
                    type_info['names'].append('phi_B')
                    type_info['pretty_names'].append(r'$\Phi_B$')
                except:
                    type_info = create_type_info(nr_fields)

            template = []
            # TODO: use nr_fields to figure out the names of the fields?
            for i in range(nr_fields):
                d = dict(fldnum=i, shape=sshape,
                         fld_name=type_info['names'][i],
                         pretty_name=type_info['pretty_names'][i])
                template.append(d)
            self.set_info("field_type", type_info['field_type'])
        return template

    @staticmethod
    def _get_single_crd(h5file, idx, nr_crds):
        gridType = h5file['StructGrid'].attrs['vsKind'].decode()
        if gridType in ['uniform']:
            if idx >= len(h5file['StructGrid'].attrs['vsNumCells']):
                raise IndexError()
            lower = h5file['StructGrid'].attrs['vsLowerBounds'][idx]
            upper = h5file['StructGrid'].attrs['vsUpperBounds'][idx]
            num = h5file['StructGrid'].attrs['vsNumCells'][idx]
            return [lower, upper, num + 1]
        elif gridType in ['rectilinear']:
            crd_arr = h5file['StructGrid/axis%d'%idx][:]
            return crd_arr
        elif gridType in ['structured']:
            if idx == 0:
                crd_arr = h5file['StructGrid'][:, 0, 0, 0]
            elif idx == 1:
                crd_arr = h5file['StructGrid'][0, :, 0, 1]
            elif idx == 2:
                crd_arr = h5file['StructGrid'][0, 0, :, 2]
            return crd_arr
        else:
            raise RuntimeError("Gkeyll StructGrid.vsKind not understood: {0}"
                               "".format(repr(gridType)))

    def make_crds(self, fname):
        with h5py.File(fname, 'r') as f:
            clist = []

            # FIXME: xyz
            crds = "xyz"
            nr_crds = len(f['StructGridField'].shape) - 1
            for i in range(nr_crds):
                try:
                    clist.append((crds[i], self._get_single_crd(f, i, nr_crds)))
                except IndexError:
                    pass

            if f['StructGrid'].attrs['vsKind'].decode() in ['uniform']:
                # FIXME: this goes xyz -> zyx
                crds = wrap_crds("uniform_cartesian", clist)
            else:
                crds = wrap_crds("nonuniform_cartesian", clist)

            return crds

##
## EOF
##
