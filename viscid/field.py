#!/usr/bin/env python
# Fields belong in grids, or by themselves as a result of a calculation

from __future__ import print_function
from warnings import warn

import numpy as np

from . import coordinate
from . import vutil

LAYOUT_DEFAULT = None  # do not translate
LAYOUT_INTERLACED = "Interlaced"
LAYOUT_FLAT = "Flat"
LAYOUT_OTHER = "Other"

def wrap_field(typ, name, crds, data, **kwargs):
    """ **kwargs passed to field constructor """
    #
    #len(clist), clist[0][0], len(clist[0][1]), type)
    for cls in vutil.subclass_spider(Field):
        if cls.TYPE == typ:
            return cls(name, crds, data, **kwargs)
    raise NotImplementedError("can not decipher field")

def scalar_fields_to_vector(name, data, fldlist, **kwargs):
    if not name:
        name = fldlist[0].name
    center = fldlist[0].center
    crds = fldlist[0].crds
    time = fldlist[0].time
    # shape = fldlist[0].data.shape

    vfield = VectorField(name, crds, data, center=center, time=time, **kwargs)
    return vfield


class Field(object):
    TYPE = "None"
    CENTERING = ['Node', 'Cell', 'Grid', 'Face', 'Edge']

    name = None  # String
    center = None  # String in CENTERING
    crds = None  # Coordinate object
    time = None  # float

    source_data = None  # numpy-like object (h5py too)
    _cache = None  # this will always be a numpy array

    def __init__(self, name, crds, data, center="Node", time=0.0,
                 forget_source=False):
        self.name = name
        self.center = center
        self.time = time
        self.crds = crds
        self.data = data

        if forget_source:
            self._source_data = self.data

    def unload(self):
        self._purge_cache()

    @property
    def dim(self):
        try:
            return len(self.source_data.shape)
        except AttributeError:
            # FIXME
            return len(self.source_data[0].shape) + 1

    @property
    def shape(self):
        # TODO: fix this
        if self.center == "Node":
            # print("crd: node shape")
            return list(self.crds.shape)
        elif self.center == "Cell":
            # print("crd: cell shape")
            return list(self.crds.shape_cc)
        else:
            warn("Edge/Face vectors not implemented, assuming node shape")
            return self.crds.shape

    @property
    def dtype(self):
        if isinstance(self._source_data, np.ndarray):
            return self._source_data.dtype.name
        elif isinstance(self._source_data, (list, tuple)):
            return self._source_data[0].dtype.name
        else:
            raise TypeError("Can not decipher source data dtype")

    @property
    def data(self):
        """ if you want to fill the cache, this will do it, note that
        to empty the cache later you can always use unload """
        if self._cache is None:
            self._fill_cache()
        return self._cache

    @data.setter
    def data(self, dat):
        if self._cache:
            self._purge_cache()
        self.source_data = dat

    def _purge_cache(self):
        """ does not guarentee that the memory will be freed """
        self._cache = None

    def _fill_cache(self):
        self._cache = self._translate_data(self.source_data)

    def _translate_data(self, dat):
        # some magic may need to happen here to accept more than np/h5 data
        # override if a type does something fancy (eg, interlacing)
        # and dat.flags["OWNDATA"]  # might i want this?
        return self._dat_to_ndarray(dat)

    @staticmethod
    def _dat_to_ndarray(dat):
        """ if already numpy array, return it, else create new np
        array... maybe a new np array should be created regardless?
        probably not, no sense gumming up too much memory
        """
        if isinstance(dat, np.ndarray):
            return dat
        else:
            if isinstance(dat, (list, tuple)):
                # some hoops for performance... this stores flat by default
                dtype = dat[0].dtype.name
                shape = [len(dat)] + list(dat[0].shape)
                datarr = np.empty(shape, dtype=dtype)
                for i, d in enumerate(dat):
                    datarr[i, ...] = d
                return datarr
            else:
                return np.array(dat, dtype=dat.dtype.name)

    #TODO: some method that gracefully gets the correct crd arrays for
    # face and edge centered fields

    def slice(self, selection):
        """ select a slice of the data using selection dictionary
        returns a new field
        """
        cc = (self.center == "Cell")
        slices, crdlst = self.crds.make_slice(selection, use_cc=cc)

        # crdlst = [(d, slcrds[i]) for i, d in enumerate(self.crds.dirs)]
        #print(crdlst)
        crds = coordinate.wrap_crds(self.crds.TYPE, crdlst)
        # print(slices, self.data.shape)
        # print(self.data.shape, slices)
        return wrap_field(self.TYPE, self.name + "_slice", crds,
                          self.data[slices], center=self.center, time=0.0)

    def __getitem__(self, item):
        return self.data[item]


class ScalarField(Field):
    TYPE = "Scalar"


class VectorField(Field):
    TYPE = "Vector"

    force_layout = LAYOUT_DEFAULT
    _layout = None

    def __init__(self, name, crds, data, force_layout=LAYOUT_DEFAULT,
                 **kwargs):
        self.force_layout = force_layout
        super(VectorField, self).__init__(name, crds, data, **kwargs)

    @property
    def dim(self):
        return super(VectorField, self).dim - 1

    @property
    def ncomp(self):
        # print("LAYOUT: ", self.layout)
        if self.layout == LAYOUT_FLAT:
            return self.data.shape[0]
        elif self.layout == LAYOUT_INTERLACED:
            return self.data.shape[-1]
        elif self.layout == LAYOUT_OTHER:
            print(self.name, self.layout)
            warn("I don't know what your layout is, assuming vectors are "
                "the last index (interleaved)...")
            return self.data.shape[-1]
        else:
            raise RuntimeError("looks like data was never setup")

    @property
    def layout(self):
        """ make sure that the data is translated before you inquire
        about the layout
        """
        if self._cache is None:
            self._fill_cache()
        return self._layout

    def _translate_data(self, dat):
        # if dat is list of fields, make it into a list of source_data so that
        # elements can be passed bare to np.array(...)
        if isinstance(dat, (list, tuple)):
            # dat = [d.source_data if isinstance(d, Field) else d for d in dat]
            for i in range(len(dat)):
                if isinstance(dat[i], Field):
                    # use _source_data so that things don't get auto cached
                    # since we are caching own copy of the data anyway
                    dat[i] = dat[i]._source_data  # pylint: disable=W0212

        dat_layout = self.detect_layout(dat)

        #print("Translating: ", self.name, "; detected layout is ", dat_layout)

        # we will preserve layout or we already have the correct layout,
        # do no translation... just like Field._translate_data
        if self.force_layout == LAYOUT_DEFAULT or \
           self.force_layout == dat_layout:
            self._layout = dat_layout
            return self._dat_to_ndarray(dat)

        # if layout is found to be other, i cant do anything with that
        elif dat_layout == LAYOUT_OTHER:
            warn("Cannot auto-detect layout; not translating; "
                 "performance may suffer")
            self._layout = LAYOUT_OTHER
            return self._dat_to_ndarray(dat)

        # ok, we demand FLAT arrays, make it so
        elif self.force_layout == LAYOUT_FLAT:
            if dat_layout != LAYOUT_INTERLACED:
                raise RuntimeError("should not be here")

            ncomp = dat.shape[-1]  # dat is interlaced
            dat_dest = np.empty([ncomp] + self.shape, dtype=dat.dtype.name)
            for i in range(ncomp):
                # NOTE: I wonder if this is the fastest way to reorder
                dat_dest[i, ...] = dat[..., i]
                # NOTE: no special case for lists, they are not
                # interpreted this way
            self._layout = LAYOUT_FLAT
            return dat_dest

        # ok, we demand INTERLACED arrays, make it so
        elif self.force_layout == LAYOUT_INTERLACED:
            if dat_layout != LAYOUT_FLAT:
                raise RuntimeError("should not be here")

            if isinstance(dat, (list, tuple)):
                ncomp = len(dat)
                dtype = dat[0].dtype.name
            else:
                ncomp = dat.shape[0]
                dtype = dat.dtype.name

            dat_dest = np.empty(self.shape + [ncomp], dtype=dtype)
            for i in range(ncomp):
                dat_dest[..., i] = dat[i]

            self._layout = LAYOUT_INTERLACED
            return dat_dest

        # catch the remaining cases
        elif self.force_layout == LAYOUT_OTHER:
            raise RuntimeError("How should I know how to force other layout?")
        else:
            raise ValueError("Bad argument for layout forcing")

    def component_views(self):
        """ return numpy views to components individually, memory layout
        of the original field is maintained """
        ncomp = self.ncomp
        if self._layout == LAYOUT_FLAT:
            return [self.data[i, ...] for i in range(ncomp)]
        elif self._layout == LAYOUT_INTERLACED:
            return [self.data[..., i] for i in range(ncomp)]
        else:
            return [self.data[..., i] for i in range(ncomp)]

    def component_fields(self):
        n = self.name
        crds = self.crds
        c = self.center
        t = self.time
        views = self.component_views()
        lst = [None] * len(views)
        for i, v in enumerate(views):
            lst[i] = ScalarField("{0} {1}".format(n, i), crds, v, center=c,
                                 time=t)
        return lst

    def detect_layout(self, dat):
        """ returns LAYOUT_XXX """
        if isinstance(dat, (list, tuple)):
            return LAYOUT_FLAT

        shape = self.shape

        if list(dat.shape[1:]) == shape:
            return LAYOUT_FLAT
        elif list(dat.shape[:-1]) == shape:
            return LAYOUT_INTERLACED
        else:
            return LAYOUT_OTHER

class MatrixField(Field):
    TYPE = "Matrix"


class TensorField(Field):
    TYPE = "Tensor"


##
## EOF
##
