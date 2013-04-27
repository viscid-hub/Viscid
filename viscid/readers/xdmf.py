#!/usr/bin/env python
# FIXME: the timetype=list does not work for the amr sample, goto line 177

from __future__ import print_function
import os
import logging

try:
    from lxml import etree
    HAS_LXML = True
except ImportError:
    HAS_LXML = False
    logging.warn("lxml library not found, no xdmf support.")


import numpy as np

from . import vfile
from .vfile_bucket import VFileBucket
from .. import grid
from .. import dataset
from .. import coordinate
from .. import field

# class XDMFDataItem(data_item.DataItem):
#     def set_precision():
#         nptype = np.dtype({'Float': 'float', 'Int': 'int', 'UInt': 'unit',
#                'Char': 'int', 'UChar': 'int'}[numbertype] + str(8*precision))


class FileXDMF(vfile.VFile):
    """ on init, parse an xdmf file into datasets, grids, and fields """
    _detector = r".*\.(xmf|xdmf)\s*$"
    _xdmf_defaults = {
        "Attribute": {
            "Name": None,
            "AttributeType": "Scalar",  # Vector,Tensor,Tensor6,Matrix,GlobalID
            "Center": "Node"  # Cell,Grid,Face,Edge
            },
        "DataItem": {
            "Name": None,
            "ItemType": "Uniform",  # Collection, tree,
                                    # HyperSlab, coordinates, Funciton
            "Dimensions": None,  # KJI order
            "NumberType": "Float",  # Int, UInt, Char, UChar
            "Precision": 4,  # 1, 4, 8
            "Format": "XML",  # HDF, Binary
            "Endian": "Native",  # Big, Little
            "Compression": "Raw",  # Zlib, Bzip2
            "Seek": 0
            },
        "Domain": {
            "Name": None
            },
        "Geometry": {
            "GeometryType": "XYZ"  # XY, X_Y_Z, VxVyVz, Origin_DxDyDz
            },
        "Grid": {
            "Name": None,
            "GridType": "Uniform",  # Collection,Tree,Subset
            "CollectionType": "Spatial",  # Temporal
            "Section": "DataItem"  # All
            },
        "Information": {
            "Name": None,
            "Value": None
            },
        "Xdmf": {
            "Version": None
            },
        "Topology": {
            "Name": None,
            "TopologyType": None,  # Polyvertex | Polyline | Polygon |
                                   # Triangle | Quadrilateral | Tetrahedron |
                                   # Pyramid| Wedge | Hexahedron | Edge_3 |
                                   # Triagle_6 | Quadrilateral_8 |
                                   # Tetrahedron_10 | Pyramid_13 | Wedge_15 |
                                   # Hexahedron_20 | Mixed |
                                   # 2DSMesh | 2DRectMesh | 2DCoRectMesh |
                                   # 3DSMesh | 3DRectMesh | 3DCoRectMesh
            "NodesPerElement": None,
            "NumberOfElement": None,
            "Dimensions": None,
            "Order": None,
            "BaseOffset": 0
             },
        "Time": {
            "Type": "Single",
            "Value": None
            }
        }

    tree = None

    def __init__(self, fname, vfilebucket=None, **kwargs):
        """ vfilebucket is a bucket for loading any hdf5 files. it can be
        None if you want the bucket to be local to this Dataset, but if you're
        loading a bunch of files, you should really be using a global bucket,
        as something like in readers.load(file) """

        if vfilebucket is None:
            # gen an empty bucket on the fly if the calling script
            # doesnt want a global bucket for all files
            self.vfilebucket = VFileBucket()

        super(FileXDMF, self).__init__(fname, vfilebucket, **kwargs)

    def _parse(self):
        self.tree = etree.parse(self.fname)
        #print(self.tree)
        self.tree.xinclude()  # TODO: implement with built in xml stuff
                              # and do xpointer by hand?
        #print(self.tree)
        root = self.tree.getroot()

        # search for all root grids, and parse them
        domain_grids = root.findall("./Domain/Grid")
        for dg in domain_grids:
            grd = self._parse_grid(dg)
            self.add(grd)

        if len(self.children) > 0:
            self.activate(0)

    def _fill_attrs(self, el):
        defs = self._xdmf_defaults[el.tag]
        ret = {}
        for opt, defval in defs.items():
            if defval is None:
                ret[opt] = el.get(opt)
            else:
                ret[opt] = type(defval)(el.get(opt, defval))
        return ret

    def _parse_grid(self, el, parent_grid=None, time=None):
        attrs = self._fill_attrs(el)
        grd = None
        coords = None

        # parse topology, or cascade parent grid's topology
        topology = el.find("./Topology")
        topoattrs = None
        if topology is not None:
            topoattrs = self._fill_attrs(topology)
        elif parent_grid and parent_grid.topology_info:
            topoattrs = parent_grid.topology_info

        # parse geometry, or cascade parent grid's geometry
        geometry = el.find("./Geometry")
        geoattrs = None
        if geometry is not None:
            crds, geoattrs = self._parse_geometry(geometry, topoattrs)
        elif parent_grid and parent_grid.geometry_info:
            geoattrs = parent_grid.geometry_info
            crds = parent_grid.crds  # this can be None and that's ok

        # parse time
        if time is None:
            t = el.find("./Time")
            if t is not None:
                pt, tattrs = self._parse_time(t)
                if tattrs["Type"] == "Single":
                    time = pt
        # cascade a parent grid's time
        if time is None and parent_grid and parent_grid.time is not None:
            time = parent_grid.time

        gt = attrs["GridType"]
        if gt == "Collection":
            times = None
            ct = attrs["CollectionType"]
            if ct == "Temporal":
                grd = dataset.DatasetTemporal(attrs["Name"])
                ttag = el.find("./Time")
                if ttag is not None:
                    times, tattrs = self._parse_time(ttag)
            elif ct == "Spatial":
                grd = dataset.Dataset(attrs["Name"])
            else:
                logging.warn("Unknown collection type {0}, ignoring "
                             "grid".format(ct))

            for i, subgrid in enumerate(el.findall("./Grid")):

                t = times[i] if (times is not None and i < len(times)) else time
                # print(subgrid, grd, t)
                self._parse_grid(subgrid, parent_grid=grd, time=t)
            grd.activate(0)

        elif gt == "Uniform":
            if not (topoattrs and geoattrs):
                logging.warn("Xdmf Uniform grids must have "
                             "topology / geometry.")
            else:
                grd = grid.Grid(attrs["Name"])
                for attribute in el.findall("./Attribute"):
                    fld = self._parse_attribute(attribute, crds, topoattrs,
                                                time)
                    if time:
                        fld.time = time
                    grd.add_field(fld)

        elif gt == "Tree":
            logging.warn("Xdmf Tree Grids not implemented, ignoring "
                         "this grid")
        elif gt == "Subset":
            logging.warn("Xdmf Subset Grids not implemented, ignoring "
                         "this grid")
        else:
            logging.warn("Unknown grid type {0}, ignoring this grid".format(gt))

        # fill attributes / data items
        # if grid and gt == "Uniform":
        #     for a in el.findall("./Attribute"):
        #         fld = self._parse_attribute(a)
        #         grid.add_field(fld)

        if grd:
            if time is not None:
                grd.time = time
            if topoattrs is not None:
                grd.topology_info = topoattrs
            if geoattrs is not None:
                grd.geometry_info = geoattrs
            if coords is not None:
                grd.set_crds(coords)
            if parent_grid is not None:
                parent_grid.add(grd)

        return grd  # can be None

    def _parse_geometry(self, geo, topoattrs):
        """ geo is the element tree item, returns Coordinate object and
            xml attributes """
        geoattrs = self._fill_attrs(geo)
        # coords = None
        crdlist = None
        crdtype = None

        topotype = topoattrs["TopologyType"]

        if topotype in ['3DRectMesh', '3DCoRectMesh',
                        '2DRectMesh', '2DCoRectMesh']:
            crdtype = "Rectilinear"
        elif topotype in ['2DSMesh', '3DSMesh']:
            crdtype = "Spherical"
        else:
            raise NotImplementedError("Unstructured grids not yet supported")

        # parse geometry into coords
        geotype = geoattrs["GeometryType"]
        if geotype == "XYZ":
            data, attrs = self._parse_dataitem(geo.find("./DataItem"),
                                               keep_flat=True)
            x = data[0::3]
            y = data[1::3]
            z = data[2::3]
            crdlist = (('z', z), ('y', y), ('x', x))

        elif geotype == "XY":
            data, attrs = self._parse_dataitem(geo.find("./DataItem"),
                                               keep_flat=True)
            x = data[0::2]
            y = data[1::2]
            z = np.zeros(len(x))
            crdlist = (('z', z), ('y', y), ('x', x))

        elif geotype == "X_Y_Z":
            crdlist = [None] * 3
            for i, crd in enumerate(['Z', 'Y', 'X']):
                di = geo.find("./dataitem[@name='{0}']".format(crd))
                if di is None:
                    raise RuntimeError("expected a V{0} element".format(crd))
                data, attrs = self._parse_dataitem(di, keep_flat=True)
                crdlist[i] = (crd.lower(), data)

        elif geotype == "VXVYVZ":
            crdlist = [None] * 3
            for i, crd in enumerate(['Z', 'Y', 'X']):
                di = geo.find("./DataItem[@Name='V{0}']".format(crd))
                if di is None:
                    raise RuntimeError("expected a V{0} element".format(crd))
                data, attrs = self._parse_dataitem(di, keep_flat=True)
                crdlist[i] = (crd.lower(), data)

        elif geotype == "ORIGIN_DXDYDZ":
            # this is for rectilinear grids with uniform spacing
            dataitems = geo.findall("./DataItem")
            data_o, attrs_o = self._parse_dataitem(dataitems[0])
            data_dx, attrs_dx = self._parse_dataitem(dataitems[1])
            nstr = None
            if topoattrs["Dimensions"]:
                nstr = topoattrs["Dimensions"]
            elif topoattrs["NumberOfElements"]:
                nstr = topoattrs["NumberOfElements"]
            else:
                raise ValueError("ORIGIN_DXDYDZ has no number of elements...")
            n = (int(num) for num in nstr)
            crdlist = [None] * 3
            for i, crd in enumerate(['z', 'y', 'x']):
                crdlist[i] = (crd, data_dx[i] * np.arange(n[i]) + data_o[i])

        else:
            logging.warn("Invalid GeometryType: {0}".format(geotype))

        crds = coordinate.wrap_crds(crdtype, crdlist)
        return crds, geoattrs

    def _parse_attribute(self, item, crds, topoattrs, time=0.0):
        attrs = self._fill_attrs(item)
        data, dataattrs = self._parse_dataitem(item.find("./DataItem"))
        name = attrs["Name"]
        center = attrs["Center"]
        typ = attrs["AttributeType"]
        fld = field.wrap_field(typ, name, crds, data, center=center, time=time)
        return fld

    def _parse_dataitem(self, item, keep_flat=False):
        """ returns the data as a numpy array, or HDF data item """
        attrs = self._fill_attrs(item)

        dimensions = attrs["Dimensions"]
        if dimensions:
            dimensions = [int(d) for d in dimensions.split(' ')]

        numbertype = attrs["NumberType"]
        precision = attrs["Precision"]
        nptype = np.dtype({'Float': 'float', 'Int': 'int', 'UInt': 'unit',
               'Char': 'int', 'UChar': 'int'}[numbertype] + str(8 * precision))

        fmt = attrs["Format"]

        if fmt == "XML":
            arr = np.fromstring(item.text, sep=' ', dtype=nptype)
            if dimensions and not keep_flat:
                arr = arr.reshape(dimensions)
            return arr, attrs

        if fmt == "HDF":
            fname, loc = item.text.strip().split(':')
            if not fname == os.path.abspath(fname):
                fname = os.path.join(self.dirname, fname)
            h5file = self.vfilebucket.load(fname, index_handle=False).file
            arr = h5file[loc]
            return arr, attrs

        if fmt == "Binary":
            raise NotImplementedError("binary xdmf data not implemented")

        logging.warn("Invalid DataItem Format.")
        return (None, None)

    def _parse_time(self, timetag):
        """ returns the time(s) as float, or numpy array, time attributes"""
        attrs = self._fill_attrs(timetag)
        timetype = attrs["Type"]

        if timetype == 'Single':
            return float(timetag.get('Value')), attrs
        elif timetype == 'List':
            return self._parse_dataitem(timetag.find('.//DataItem'))[0], attrs
        elif timetype == 'Range':
            raise NotImplementedError("Time Range not yet implemented")
            #dat, dattrs = self._parse_dataitem(timetag.find('.//DataItem'))
            # TODO: this is not the most general, but I think it'll work
            # as a first stab, plus I will probably not need Range ever
            #tgridtag = timetag.find("ancestor::Grid[@GridType='Collection']"
            #                             "[@CollectionType='Temporal'][1]"))
            #n = len(tgridtag.find(.//Grid[@GridType='Collection']
            #        [CollectionType=['Spatial']]))
            #return np.linspace(dat[0], dat[1], n)
            #return np.arange(dat[0], dat[1])
        elif timetype == 'HyperSlab':
            dat, dattrs = self._parse_dataitem(timetag.find('.//DataItem'))
            arr = np.array([dat[0] + i * dat[1] for i in range(int(dat[2]))])
            return arr, attrs
        else:
            logging.warn("invalid TimeType.\n")

# class FileGgcm2dXdmf(FileXDMF):
#     _detector = r'.*\.p[xyz]_[0-9]+(\..*)?\.(xmf|xdmf)\s*$'

#     def __init__(self, fname=None):
#         super(FileGgcm2dXdmf, self).__init__(fname=fname)


# class FileGgcm3dXdmf(FileXDMF):
#     _detector = r'.*\.3df(\..*)?\.(xmf|xdmf)\s*$'

#     def __init__(self, fname=None):
#         super(FileGgcm3dXdmf, self).__init__(fname=fname)

if __name__ == '__main__':
    import sys
    # import os
    import viscid

    _viscid_root = os.path.dirname(viscid.__file__)
    f = FileXDMF(_viscid_root + '/../../sample/local_0001.py_0.xdmf')
    sys.stderr.write("{0}\n".format(f))

##
## EOF
##
