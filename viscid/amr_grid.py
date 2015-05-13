"""AMR tools"""

from __future__ import print_function, division
# from timeit import default_timer

import numpy as np

from viscid.grid import Grid
from viscid.amr_field import AMRField
# from viscid import cyamr

def dataset_to_amr_grid(dset, template_skeleton=None):
    """Try to divine AMR-ness from a Dataset

    Args:
        dset (:py:class:`viscid.dataset.Dataset`): Should be a
            collection of grids that could represet a series of
            patches in an AMR hierarchy.
        patch_list (:py:class:`AMRTemplate`): a guess at how patches
            are arranged for speedup. Allows us not to have to re-build
            the neighbor relationships if we've seen this patch
            layout before

    Returns:
        AMRGrid if dset looks like an AMR hierarchy, or dset itself
        otherwise
    """
    if len(dset) <= 1:
        return dset, False
    if not all(isinstance(grd, Grid) for grd in dset):
        return dset, False

    # print(">> time:", dset.time)

    # t0 = default_timer()
    grid = AMRGrid(dset, skeleton=template_skeleton)
    # t1 = default_timertime()
    # print("Making AMR grid took {0:g} secs".format(t1 - t0))
    return grid, True


class AMRSkeleton(object):
    """Organizes the neighbor relationships of AMR grids"""
    patches = None
    neighbors_index = None  # list of dicts, parallel with patches
    all_neighbors_index = None  # list of dicts, parallel with patches

    xl = None  # shape == (npatches x 3)
    xm = None  # shape == (npatches x 3)
    xh = None  # shape == (npatches x 3)
    L = None  # shape == (npatches x 3)

    global_xl = None
    global_xh = None

    def __init__(self, dset):
        """Summary

        Args:
            dset (type): spatial dataset or list of grids
        """
        if len(dset) == 0:
            raise ValueError()
        if not all(isinstance(grd, Grid) for grd in dset):
            raise ValueError()

        self.patches = [AMRPatch(g.crds) for g in dset]
        npatches = len(self.patches)
        if npatches == 0:
            raise ValueError("AMR Skeleton with 0 patches? no can do.")
        p0_xl = self.patches[0].xl
        self.xl = np.empty([npatches, len(p0_xl)], dtype=p0_xl.dtype)
        self.xm = np.empty_like(self.xl)
        self.xh = np.empty_like(self.xl)
        self.L = np.empty_like(self.xl)

        for i, patch in enumerate(self.patches):
            self.xl[i, :] = patch.xl
            self.xm[i, :] = patch.xm
            self.xh[i, :] = patch.xh
            self.L[i, :] = patch.L

        self.global_xl = np.min(self.xl, axis=0)
        self.global_xh = np.max(self.xh, axis=0)

        # self.discover_neighbors()

    def compatable_with(self, dset):
        # TODO: i should make /store a hash for this
        # t0 = default_timer()
        if len(dset) != len(self.patches):
            return False

        for my_patch, grid in zip(self.patches, dset):
            if np.any(my_patch.n != grid.crds.shape_cc):
                return False
            if not np.allclose(my_patch.xl, grid.crds.xl_nc):
                return False
            # if n and xl match... do i need to check xh?
            # if not np.allclose(my_patch.xh, grid.crds.xh_nc):
            #     return False
        # t1 = default_timer()
        # print(">> compatable   {0:g} secs".format(t1 - t0))
        return True

    def discover_neighbors(self):
        d = dict(xm=[], xp=[], ym=[], yp=[], zm=[], zp=[])
        self.neighbors_index = [d.copy() for _ in self.patches]
        self.all_neighbors_index = [[] for _ in self.patches]

        # call find_relationships (N * (N - 1)) / 2 times
        for i, patch in enumerate(self.patches):
            for j, other in enumerate(self.patches[:i]):
                rels = patch.find_relationships(other)
                # print("RELS:", i, j, rels)
                for rel in rels:
                    self.neighbors_index[i][rel[0]].append(j)
                    self.neighbors_index[j][rel[1]].append(i)

                    if j not in self.all_neighbors_index[i]:
                        self.all_neighbors_index[i].append(j)
                    if i not in self.all_neighbors_index[j]:
                        self.all_neighbors_index[j].append(i)
        self.tell_patches_about_neighbors()

    def tell_patches_about_neighbors(self):
        for i, neighbors_dict in enumerate(self.neighbors_index):
            # print(">>", i, neighbors_dict)
            _d = {}
            for relationship, neighbor_inds in neighbors_dict.items():
                _d[relationship] = []
                for ind in neighbor_inds:
                    _d[relationship].append(self.patches[ind])
            _l = [self.patches[v] for v in self.all_neighbors_index[i]]
            self.patches[i].set_neighbors(_d, _l)

    def patch_by_loc(self, loc):
        """returns index of patch that contans loc"""
        loc = np.asarray(loc)
        contains = np.all(np.abs(loc - self.xm) <= (0.5 * self.L), axis=1)
        # argmax is slightly faster, but gives no indication if loc is outside
        # the global domain
        # return np.argmax(contains)
        nonzero_inds = np.flatnonzero(contains)
        if len(nonzero_inds) == 0:
            return -1
        else:
            return nonzero_inds[0]

    def patch_by_loc2(self, loc):
        """slower implementation"""
        loc = np.asarray(loc)
        for i, patch in enumerate(self.patches):
            if np.all(loc >= patch.xl) and np.all(loc <= patch.xh):
                return i
        return -1

    # def selected_inds(self, slc):
    #     """return list of indices of patches that are selected
    #     by a given slice"""
    #     # axes =
    #     contains = np.all(np.abs(loc - self.xm) <= (0.5 * self.L), axis=1)
    #     pass

# SLICING NOTES: have field level parsing of slices that raise a ValueError
# if the slices are ints (slice by index)

class AMRGrid(Grid):
    """The AMRGrid contains a list of patches"""
    skeleton = None

    _src_grids = None
    # fields = None

    def __init__(self, dset, skeleton=None):
        """Summary

        Args:
            dset: Spatial Dataset or list of grids
            skeleton (type, optional): if given, check if dset is
                compatable with existing skeleton. if it's not, or
                skeleton is None, then make a new skeleton and
                discover neighbors etc.
        """
        if len(dset) == 0:
            raise ValueError()
        if not all(isinstance(grd, Grid) for grd in dset):
            raise ValueError()

        if skeleton is not None and skeleton.compatable_with(dset):
            self.skeleton = skeleton
        else:
            self.skeleton = AMRSkeleton(dset)

        ## oh boy, this is uncomfortable
        self.topology_info = dset[0].topology_info
        self.geometry_info = dset[0].geometry_info
        self._src_crds = None
        self._crds = None
        self.fields = None

        self.force_vector_layout = dset[0].force_vector_layout
        self.longterm_field_caches = dset[0].longterm_field_caches
        ##

        ## this feels very awkward too
        time = dset[0].time
        info = dset[0]._info  # pylint: disable=protected-access
        parents = dset[0].parents

        super(AMRGrid, self).__init__(time=time, info=info, parents=parents)

        self._src_grids = [g for g in dset]
        # for g in dset:
        #     for fld in g.fields.values():
        #         self.fields

    @property
    def xl_nc(self):
        return self.skeleton.global_xl

    @property
    def xh_nc(self):
        return self.skeleton.global_xh

    @property
    def xl_cc(self):
        raise NotImplementedError("You probably want xl_nc for an amr grid")

    @property
    def xh_cc(self):
        raise NotImplementedError("You probably want xh_nc for an amr grid")

    # def _make_amr_field(self):
    #     fld = AMRField()

    def get_field(self, fldname, time=None, force_longterm_caches=False,
                  slc=None):  # pylint: disable=unused-argument
        fld_list = []
        assert force_longterm_caches == False  # FIXME
        selected_patches = list(range(len(self._src_grids)))
        patches = [self._src_grids[i] for i in selected_patches]
        fld_list = [p.get_field(fldname) for p in patches]
        amr_fld = AMRField(fld_list, self.skeleton)
        if slc:
            amr_fld = amr_fld.slice_and_keep(slc)
        return amr_fld


class AMRPatch(object):
    xl = None
    xh = None
    xm = None
    L = None
    n = None  # number of cells (== shape_cc)

    neighbors = None
    all_neighbors = None

    def __init__(self, crds):
        """
        Args:
            crds: StructuredCoordinate object
            patch_list: list of other patches to seach through to
                find neighbor relationships
        """
        self.xl = crds.xl_nc
        self.xh = crds.xh_nc
        self.xm = 0.5 * (self.xl + self.xh)
        self.L = self.xh - self.xl
        self.n = crds.shape_cc

        self.reset_neighbors()

    def reset_neighbors(self):
        self.neighbors = dict(xm=[], xp=[], ym=[], yp=[], zm=[], zp=[])
        self.all_neighbors = []

    def set_neighbors(self, relationship_dict, all_neighbors_list):
        self.neighbors = relationship_dict
        self.all_neighbors = all_neighbors_list

    def find_relationships(self, patch):
        # if r > 0, it's a p relationship for self, eles it's an m
        rel_list = []
        r = patch.xm - self.xm
        d = 0.5 * (self.L + patch.L)
        # print("A>", self.xl, self.xh, self.xm)
        # print("B>", patch.xl, patch.xh, patch.xm)
        # print("?", np.abs(r) - d)
        close = np.isclose(np.abs(r) - d, 0.0)
        for ri, flagi, ax in zip(r, close, 'zyx'):
            if flagi:
                if ri > 0:
                    rel_list.append((ax + 'p', ax + 'm'))
                else:
                    rel_list.append((ax + 'p', ax + 'm'))
        return rel_list


##
## EOF
##
