#!/usr/bin/env python
# FIXME: the timetype=list does not work for the amr sample, goto line 177

from __future__ import print_function
import os
import logging
from xml.etree import ElementTree

import numpy as np

from . import _xdmf_include
from . import vfile
from .vfile_bucket import VFileBucket
from .hdf5 import FileLazyHDF5
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
            "Center": "node"  # cell,Grid,face,edge
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

    # tree = None

    def __init__(self, fname, vfilebucket=None, **kwargs):
        """ vfilebucket is a bucket for loading any hdf5 files. it can be
        None if you want the bucket to be local to this Dataset, but if you're
        loading a bunch of files, you should really be using a global bucket,
        as something like in readers.load_file(file) """

        if vfilebucket is None:
            # gen an empty bucket on the fly if the calling script
            # doesnt want a global bucket for all files
            self.vfilebucket = VFileBucket()

        super(FileXDMF, self).__init__(fname, vfilebucket, **kwargs)

    def _parse(self):
        # lxml has better xpath support, so it's preferred, but it stops
        # if an xinclude doesn't exist, so for now use our custom extension
        # of the default python xml lib
        # if HAS_LXML:
        #     # sweet, you have it... use the better xml library
        #     tree = etree.parse(self.fname) #pylint: disable=E0602
        #     tree.xinclude()  # TODO: gracefully ignore include problems
        #     root = tree.getroot()
        tree = ElementTree.parse(self.fname)
        root = tree.getroot()
        _xdmf_include.include(root, base_url=self.fname)

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
        crds = None

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
            if len(grd.children) > 0:
                grd.activate(0)

        elif gt == "Uniform":
            if not (topoattrs and geoattrs):
                logging.warn("Xdmf Uniform grids must have "
                             "topology / geometry.")
            else:
                grd = self._grid_type(attrs["Name"], **self._grid_opts)
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
            if crds is not None:
                grd.set_crds(crds)
            if parent_grid is not None:
                parent_grid.add(grd)

        return grd  # can be None

    def _parse_geometry(self, geo, topoattrs):
        """ geo is the element tree item, returns Coordinate object and
            xml attributes """
        geoattrs = self._fill_attrs(geo)
        # crds = None
        crdlist = None
        crdtype = None

        topotype = topoattrs["TopologyType"]

        # parse geometry into crds
        geotype = geoattrs["GeometryType"]
        if geotype.upper() == "XYZ":
            data, attrs = self._parse_dataitem(geo.find("./DataItem"),
                                               keep_flat=True)
            # x = data[0::3]
            # y = data[1::3]
            # z = data[2::3]
            # crdlist = (('z', z), ('y', y), ('x', x))
            # quietly do nothing... we don't support unstructured grids
            # or 3d spherical yet, and 2d spherical can be figured out
            # if we assume the grid spans the whole sphere
            crdlist = None

        elif geotype.upper() == "XY":
            data, attrs = self._parse_dataitem(geo.find("./DataItem"),
                                               keep_flat=True)
            # x = data[0::2]
            # y = data[1::2]
            # z = np.zeros(len(x))
            # crdlist = (('z', z), ('y', y), ('x', x))
            # quietly do nothing... we don't support unstructured grids
            # or 3d spherical yet, and 2d spherical can be figured out
            # if we assume the grid spans the whole sphere
            crdlist = None

        elif geotype.upper() == "X_Y_Z":
            crdlookup = {'z': 0, 'y': 1, 'x': 2}
            crdlist = [['z', None], ['y', None], ['x', None]]
            # can't use ./DataItem[@Name='X'] so python2.6 works
            dataitems = geo.findall("./DataItem")
            for di in dataitems:
                crd_name = di.attrib["Name"].lower()
                data, attrs = self._parse_dataitem(di, keep_flat=True)
                crdlist[crdlookup.pop(crd_name)][1] = data
            if len(crdlookup) > 0:
                raise RuntimeError("XDMF format error: Coords not specified "
                                   "for {0} dimesions"
                                   "".format(list(crdlookup.keys())))

        elif geotype.upper() == "VXVYVZ":
            crdlookup = {'z': 0, 'y': 1, 'x': 2}
            crdlist = [['z', None], ['y', None], ['x', None]]
            # can't use ./DataItem[@Name='VX'] so python2.6 works
            dataitems = geo.findall("./DataItem")
            for di in dataitems:
                crd_name = di.attrib["Name"].lstrip('V').lower()
                data, attrs = self._parse_dataitem(di, keep_flat=True)
                crdlist[crdlookup.pop(crd_name)][1] = data
            if len(crdlookup) > 0:
                raise RuntimeError("XDMF format error: Coords not specified "
                                   "for {0} dimesions"
                                   "".format(list(crdlookup.keys())))

        elif geotype.upper() == "ORIGIN_DXDYDZ":
            # this is for grids with uniform spacing
            dataitems = geo.findall("./DataItem")
            data_o, attrs_o = self._parse_dataitem(dataitems[0])
            data_dx, attrs_dx = self._parse_dataitem(dataitems[1])
            dtyp = data_o.dtype
            nstr = None
            if topoattrs["Dimensions"]:
                nstr = topoattrs["Dimensions"]
            elif topoattrs["NumberOfElements"]:
                nstr = topoattrs["NumberOfElements"]
            else:
                raise ValueError("ORIGIN_DXDYDZ has no number of elements...")
            n = [int(num) for num in nstr.split()]
            crdlist = [None] * 3
            for i, crd in enumerate(['z', 'y', 'x']):
                crd_arr = data_dx[i] * np.arange(n[i], dtype=dtyp) + data_o[i]
                crdlist[i] = (crd, crd_arr)

        else:
            logging.warn("Invalid GeometryType: {0}".format(geotype))

        if topotype in ['3DCoRectMesh', '2DCoRectMesh']:
            crdtype = "uniform_cartesian"
        if topotype in ['3DRectMesh', '2DRectMesh']:
            crdtype = "nonuniform_cartesian"
        elif topotype in ['2DSMesh']:
            crdtype = "nonuniform_spherical"
            ######## this doesn't quite work, but it's too heavy to be useful
            ######## anyway... if we assume a 2d spherical grid spans the
            ######## whole sphere, and radius doesnt matter, all we need are
            ######## the nr_phis / nr_thetas, so let's just do that
            # # this asserts that attrs["Dimensions"] will have the zyx
            # # dimensions
            # # turn x, y, z -> phi, theta, r
            # dims = [int(s) for
            #         s in reversed(topoattrs["Dimensions"].split(' '))]
            # dims = [1] * (3 - len(dims)) + dims
            # nr, ntheta, nphi = [d for d in dims]
            # # dtype = crdlist[0][1].dtype
            # # phi, theta, r = [np.empty((n,), dtype=dtype) for n in dims]
            # z, y, x = (crdlist[i][1].reshape(dims) for i in range(3))
            # nphitheta = nphi * ntheta
            # r = np.sqrt(x[::nphitheta, 0, 0]**2 + y[::nphitheta, 0, 0]**2 +
            #             z[::nphitheta, 0, 0]**2)
            # ir = nr // 2  # things get squirrly near the extrema
            # theta = (180.0 / np.pi) * \
            #         (np.arccos(z[ir, :, ::nphi] / r[ir]).reshape(-1))
            # itheta = ntheta // 2
            # phi = (180.0 / np.pi) * \
            #       np.arctan2(y[ir, itheta, :], x[ir, itheta, :])
            # print(dims, nr, ntheta, nphi)
            # print("r:", r.shape, r)
            # print("theta:", theta.shape, theta)
            # print("phi:", phi.shape, phi)
            # raise RuntimeError()
            ######## general names in spherical crds
            # ntheta, nphi = [int(s) for s in topoattrs["Dimensions"].split(' ')]
            # crdlist = [['theta', [0.0, 180.0, ntheta]],
            #            ['phi', [0.0, 360.0, nphi]]]
            ######## names on a map
            nlat, nlon = [int(s) for s in topoattrs["Dimensions"].split(' ')]
            crdlist = [['lat', [0.0, 180.0, nlat]],
                       ['lon', [0.0, 360.0, nlon]]]

        elif topologytype in ['3DSMesh']:
            raise NotImplementedError("3D spherical grids not yet supported")
        else:
            raise NotImplementedError("Unstructured grids not yet supported")

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
            h5file = self.vfilebucket.load_file(fname, index_handle=False,
                                                file_type=FileLazyHDF5)
            arr = h5file.get_data(loc)  #pylint: disable=E1103
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
