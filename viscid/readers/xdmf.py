#!/usr/bin/env python
# FIXME: the timetype=list does not work for the amr sample, goto line 177

from __future__ import print_function
import os
import sys
# from xml.etree import ElementTree

import numpy as np

from viscid.compat import element_tree
from viscid import logger
from viscid.readers.vfile_bucket import ContainerFile
from viscid.readers.hdf5 import FileLazyHDF5
from viscid import amr_grid
from viscid import coordinate

# class XDMFDataItem(data_item.DataItem):
#     def set_precision():
#         nptype = np.dtype({'Float': 'float', 'Int': 'int', 'UInt': 'unit',
#                'Char': 'int', 'UChar': 'int'}[numbertype] + str(8*precision))

class FileXDMF(ContainerFile):  # pylint: disable=abstract-method
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

    h5_root_dir = None
    _last_amr_skeleton = None  # experimental, should be moved
    # tree = None

    def __init__(self, fname, h5_root_dir=None, **kwargs):
        """XDMF File"""
        if h5_root_dir is not None:
            h5_root_dir = os.path.expandvars(h5_root_dir)
            self.h5_root_dir = os.path.expanduser(h5_root_dir)

        super(FileXDMF, self).__init__(fname, **kwargs)

    def _parse(self):
        grids = self._parse_file(self.fname, self)
        for grid in grids:
            self.add(grid)

        if len(self.children) > 0:
            self.activate(0)

    def _parse_file(self, fname, parent_node):
        # lxml has better xpath support, so it's preferred, but it stops
        # if an xinclude doesn't exist, so for now use our custom extension
        # of the default python xml lib
        # if HAS_LXML:
        #     # sweet, you have it... use the better xml library
        #     tree = etree.parse(self.fname)  # pylint: disable=E0602
        #     tree.xinclude()  # TODO: gracefully ignore include problems
        #     root = tree.getroot()
        grids = []
        tree = element_tree.parse(fname)
        element_tree.xinclude(tree, base_url=fname)
        root = tree.getroot()

        # search for all root grids, and parse them
        domain_grids = root.findall("./Domain/Grid")
        for dg in domain_grids:
            grd = self._parse_grid(dg, parent_node)
            grids.append(grd)
        return grids

    def _fill_attrs(self, el):
        defs = self._xdmf_defaults[el.tag]
        ret = {}
        for opt, defval in defs.items():
            if defval is None:
                ret[opt] = el.get(opt)
            else:
                ret[opt] = type(defval)(el.get(opt, defval))
        return ret

    def _parse_grid(self, el, parent_node=None, time=None):
        attrs = self._fill_attrs(el)
        grd = None
        crds = None

        # parse topology, or cascade parent grid's topology
        topology = el.find("./Topology")
        topoattrs = None
        if topology is not None:
            topoattrs = self._fill_attrs(topology)
        elif parent_node and parent_node.topology_info:
            topoattrs = parent_node.topology_info

        # parse geometry, or cascade parent grid's geometry
        geometry = el.find("./Geometry")
        geoattrs = None
        if geometry is not None:
            crds, geoattrs = self._parse_geometry(geometry, topoattrs)
        elif parent_node and parent_node.geometry_info:
            geoattrs = parent_node.geometry_info
            crds = parent_node.crds  # this can be None and that's ok

        # parse time
        if time is None:
            t = el.find("./Time")
            if t is not None:
                pt, tattrs = self._parse_time(t)
                if tattrs["Type"] == "Single":
                    time = pt
        # cascade a parent grid's time
        if time is None and parent_node and parent_node.time is not None:
            time = parent_node.time

        gt = attrs["GridType"]
        if gt == "Collection":
            times = None
            ct = attrs["CollectionType"]
            if ct == "Temporal":
                grd = self._make_dataset(parent_node, dset_type="temporal",
                                         name=attrs["Name"])
                self._inject_info(el, grd)
                ttag = el.find("./Time")
                if ttag is not None:
                    times, tattrs = self._parse_time(ttag)
            elif ct == "Spatial":
                grd = self._make_dataset(parent_node, name=attrs["Name"])
                self._inject_info(el, grd)
            else:
                logger.warning("Unknown collection type %s, ignoring grid", ct)

            for i, subgrid in enumerate(el.findall("./Grid")):
                t = times[i] if (times is not None and i < len(times)) else time
                # print(subgrid, grd, t)
                self._parse_grid(subgrid, parent_node=grd, time=time)
            if len(grd.children) > 0:
                grd.activate(0)

        elif gt == "Uniform":
            if not (topoattrs and geoattrs):
                logger.warning("Xdmf Uniform grids must have "
                               "topology / geometry.")
            else:
                grd = self._make_grid(parent_node, name=attrs["Name"],
                                      **self._grid_opts)
                self._inject_info(el, grd)
                for attribute in el.findall("./Attribute"):
                    fld = self._parse_attribute(grd, attribute, crds,
                                                topoattrs, time)
                    if time:
                        fld.time = time
                    grd.add_field(fld)

        elif gt == "Tree":
            logger.warning("Xdmf Tree Grids not implemented, ignoring "
                           "this grid")
        elif gt == "Subset":
            logger.warning("Xdmf Subset Grids not implemented, ignoring "
                           "this grid")
        else:
            logger.warning("Unknown grid type %s, ignoring this grid", gt)

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

            # EXPERIMENTAL AMR support, _last_amr_grid shouldn't be an attribute
            # of self, since that will only remember the most recently generated
            # amr grid, but that's ok for now
            # if gt == "Uniform":
            #     print(">!", crds._TYPE, crds.xl_nc, grd.time)
            #     print(">!?", type(parent_node), parent_node.children._ordered,
            #           len(parent_node.children))
            if gt == "Collection" and ct == "Spatial":
                grd, is_amr = amr_grid.dataset_to_amr_grid(grd,
                                                           self._last_amr_skeleton)
                if is_amr:
                    self._last_amr_skeleton = grd.skeleton

            if parent_node is not None:
                parent_node.add(grd)

        return grd  # can be None

    def _parse_geometry(self, geo, topoattrs):
        """ geo is the element tree item, returns Coordinate object and
            xml attributes """
        geoattrs = self._fill_attrs(geo)
        # crds = None
        crdlist = None
        crdtype = None
        crdkwargs = {}

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
            crdlookup = {'x': 0, 'y': 1, 'z': 2}
            crdlist = [['x', None], ['y', None], ['z', None]]
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
            crdkwargs["full_arrays"] = True

        elif geotype.upper() == "VXVYVZ":
            crdlookup = {'x': 0, 'y': 1, 'z': 2}
            crdlist = [['x', None], ['y', None], ['z', None]]
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
            crdkwargs["full_arrays"] = True

        elif geotype.upper() == "ORIGIN_DXDYDZ":
            # this is for grids with uniform spacing
            dataitems = geo.findall("./DataItem")
            data_o, _ = self._parse_dataitem(dataitems[0])
            data_dx, _ = self._parse_dataitem(dataitems[1])
            dtyp = data_o.dtype
            nstr = None
            if topoattrs["Dimensions"]:
                nstr = topoattrs["Dimensions"]
            elif topoattrs["NumberOfElements"]:
                nstr = topoattrs["NumberOfElements"]
            else:
                raise ValueError("ORIGIN_DXDYDZ has no number of elements...")
            n = [int(num) for num in nstr.split()]
            # FIXME: OpenGGCM output uses ZYX ordering even though the xdmf
            # website says it should be XYZ, BUT, the file opens correctly
            # in Paraview with zyx, so... I guess i need to do this [::-1]
            # nonsense here
            data_o, data_dx, n = data_o[::-1], data_dx[::-1], n[::-1]
            crdlist = [None] * 3
            for i, crd in enumerate(['x', 'y', 'z']):
                n_nc, n_cc = n[i], n[i] - 1
                crd_arr = [data_o[i], data_o[i] + (n_cc * data_dx[i]), n_nc]
                crdlist[i] = (crd, crd_arr)
            crdkwargs["dtype"] = dtyp
            crdkwargs["full_arrays"] = False
        else:
            logger.warning("Invalid GeometryType: %s", geotype)

        if topotype in ['3DCoRectMesh', '2DCoRectMesh']:
            crdtype = "uniform_cartesian"
        elif topotype in ['3DRectMesh', '2DRectMesh']:
            if crdkwargs.get("full_arrays", True):
                crdtype = "nonuniform_cartesian"
            else:  # HACK, hopefully not used ever
                crdtype = "uniform_cartesian"
        elif topotype in ['2DSMesh']:
            crdtype = "uniform_spherical"  # HACK!
            ######## this doesn't quite work, but it's too heavy to be useful
            ######## anyway... if we assume a 2d spherical grid spans the
            ######## whole sphere, and radius doesnt matter, all we need are
            ######## the nr_phis / nr_thetas, so let's just do that
            # # this asserts that attrs["Dimensions"] will have the xyz
            # # dimensions
            # # turn x, y, z -> phi, theta, r
            # dims = [int(s) for
            #         s in reversed(topoattrs["Dimensions"].split(' '))]
            # dims = [1] * (3 - len(dims)) + dims
            # nr, ntheta, nphi = [d for d in dims]
            # # dtype = crdlist[0][1].dtype
            # # phi, theta, r = [np.empty((n,), dtype=dtype) for n in dims]
            # x, y, z = (crdlist[i][1].reshape(dims) for i in range(3))
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
            ntheta, nphi = [int(s) for s in topoattrs["Dimensions"].split(' ')]
            crdlist = [['phi', [0.0, 360.0, nphi]],
                       ['theta', [0.0, 180.0, ntheta]]]
            crdkwargs["full_arrays"] = False
            crdkwargs["units"] = 'deg'

        elif topotype in ['3DSMesh']:
            raise NotImplementedError("3D spherical grids not yet supported")
        else:
            raise NotImplementedError("Unstructured grids not yet supported")

        crds = coordinate.wrap_crds(crdtype, crdlist, **crdkwargs)
        return crds, geoattrs

    def _parse_attribute(self, parent_node, item, crds, topoattrs, time=0.0):
        """
        Args:
            parent_node (Dataset, Grid, or None): Hint what the parent will
                be. Necessary if _make_field makes decisions based on the
                info dict
        """
        attrs = self._fill_attrs(item)
        data, dataattrs = self._parse_dataitem(item.find("./DataItem"))
        name = attrs["Name"]
        center = attrs["Center"]
        fldtype = attrs["AttributeType"]
        fld = self._make_field(parent_node, fldtype, name, crds, data,
                               center=center, time=time, zyx_native=True)
        self._inject_info(item, fld)

        return fld

    def _parse_dataitem(self, item, keep_flat=False):
        """ returns the data as a numpy array, or HDF data item """
        attrs = self._fill_attrs(item)

        dimensions = attrs["Dimensions"]
        if dimensions:
            dimensions = [int(d) for d in dimensions.split(' ')]

        numbertype = attrs["NumberType"]
        precision = attrs["Precision"]
        nptype = np.dtype({'Float': 'f', 'Int': 'i', 'UInt': 'u',
                           'Char': 'i', 'UChar': 'u'}[numbertype] + str(precision))

        fmt = attrs["Format"]

        if fmt == "XML":
            arr = np.fromstring(item.text, sep=' ', dtype=nptype)
            if dimensions and not keep_flat:
                arr = arr.reshape(dimensions)
            return arr, attrs

        if fmt == "HDF":
            fname, loc = item.text.strip().split(':')

            # FIXME: startswith '/' is unix path name specific
            if self.h5_root_dir is not None:
                fname = os.path.join(self.h5_root_dir, fname)
            elif not fname.startswith('/'):
                fname = os.path.join(self.dirname, fname)
            h5file = self._load_child_file(fname, index_handle=False,
                                           file_type=FileLazyHDF5)
            arr = h5file.get_data(loc)
            return arr, attrs

        if fmt == "Binary":
            raise NotImplementedError("binary xdmf data not implemented")

        logger.warning("Invalid DataItem Format.")
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
            # dat, dattrs = self._parse_dataitem(timetag.find('.//DataItem'))
            # TODO: this is not the most general, but I think it'll work
            # as a first stab, plus I will probably not need Range ever
            # tgridtag = timetag.find("ancestor::Grid[@GridType='Collection']"
            #                         "[@CollectionType='Temporal'][1]"))
            # n = len(tgridtag.find(.//Grid[@GridType='Collection']
            #         [CollectionType=['Spatial']]))
            # return np.linspace(dat[0], dat[1], n)
            # return np.arange(dat[0], dat[1])
        elif timetype == 'HyperSlab':
            dat, dattrs = self._parse_dataitem(timetag.find('.//DataItem'))
            arr = np.array([dat[0] + i * dat[1] for i in range(int(dat[2]))])
            return arr, attrs
        else:
            logger.warning("invalid TimeType.\n")

    def _parse_information(self, information):
        """ parse generic information tag """
        attrs = self._fill_attrs(information)
        name = attrs["Name"]
        val = attrs["Value"]
        if val is None:
            _di = information.find(".//DataItem")
            if _di:
                val, _ = self._parse_dataitem(_di)
            else:
                val = information.text
        return name, val

    def _inject_info(self, el, target):
        for _, information in enumerate(el.findall("./Information")):
            _name, _val = self._parse_information(information)
            target.set_info(_name, _val)


def _main():
    import viscid

    f = FileXDMF(os.path.join(viscid.sample_dir, 'local_0001.py_0.xdmf'))
    sys.stderr.write("{0}\n".format(f))

    return 0

if __name__ == '__main__':
    sys.exit(_main())

##
## EOF
##
