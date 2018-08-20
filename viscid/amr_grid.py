"""AMR tools"""

from __future__ import print_function, division
# from timeit import default_timer

import numpy as np

import viscid
from viscid.grid import Grid
from viscid.amr_field import AMRField
from viscid.cython import CythonNotBuilt
from viscid.cython import cyamr


__all__ = ["dataset_to_amr_grid"]


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

    xl = None  # shape == (npatches x 3)
    xm = None  # shape == (npatches x 3)
    xh = None  # shape == (npatches x 3)
    L = None  # shape == (npatches x 3)

    nr_neighbors = None  # shape == (npatches)
    neighbors = None  # shape == (npatches x 48)
    neighbor_mask = None  # shape == (npatches x 48)

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
        self.n = np.empty_like(self.xl, dtype='i')

        for i, patch in enumerate(self.patches):
            self.xl[i, :] = patch.xl
            self.xm[i, :] = patch.xm
            self.xh[i, :] = patch.xh
            self.L[i, :] = patch.L
            self.n[i, :] = patch.n

        self.global_xl = np.min(self.xl, axis=0)
        self.global_xh = np.max(self.xh, axis=0)

        try:
            # from timeit import default_timer as time
            # t0 = time()
            neighbor_info = cyamr.discover_neighbors(self)
            nr_neighbors, neighbors, neighbor_mask = neighbor_info
            self.nr_neighbors = nr_neighbors
            self.neighbors = neighbors
            self.neighbor_mask = neighbor_mask
            # print("nr_neighbors:", nr_neighbors)
            # cyamr.discover_neighbors(self)
            # t1 = time()
            # print("discover took: {0:g} secs".format(t1 - t0))

            # Sort neighbors such that face neighbors appear before edge
            # neighbors appear before corner neighbors
            # hamming weight is # of 1s in binary representation of an int
            # NOTE: 0 is intentionally set to the largest number since
            #       the mask is 0 for empty values
            # t0 = time()
            # hamming_weight = np.array([9, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            #                            3, 3, 4, 1], dtype='i')
            # # heapsort sort is not stable, do i need mergesort here?
            # new_order = np.argsort(hamming_weight[neighbor_mask >> 6], axis=1,
            #                        kind='heapsort')
            # neighbors = neighbors[new_order]
            # neighbor_mask = neighbor_mask[new_order]
            # t1 = time()
            # print("sort took: {0:g} secs".format(t1 - t0))
        except CythonNotBuilt:
            # at the moment, only cython code uses the neighbor index,
            # so we can ignore this for now, but maybe not in the future
            pass

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
                  slc=Ellipsis):  # pylint: disable=unused-argument
        fld_list = []
        assert force_longterm_caches is False  # FIXME
        selected_patches = list(range(len(self._src_grids)))
        patches = [self._src_grids[i] for i in selected_patches]
        fld_list = [p.get_field(fldname) for p in patches]
        amr_fld = AMRField(fld_list, self.skeleton)
        if slc != Ellipsis:
            amr_fld = amr_fld.slice_and_keep(slc)
        return amr_fld

    def print_tree(self, depth=-1, prefix=""):
        tree_prefix = viscid.vutil.tree_prefix

        print("{0}{1}".format(prefix, self))
        if self._src_grids and depth != 0:
            print("{0}Grid 0 of {1}".format(prefix, len(self._src_grids)))
            self._src_grids[0].print_tree(depth=depth - 1,
                                          prefix=prefix)


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

##
## EOF
##
