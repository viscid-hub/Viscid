from __future__ import print_function
from os import path

import numpy as np

from .. import coordinate
from .. import field
from ..calculator import calc

def load_topo(itop_name, ihem_name):
    """ 0 = closed; 1 = south open; 2 = north open; 3 = sw """
    #x, y, z, itop = np.loadtxt(fname, unpack=True, usecols=(0,1,2,6))
    with open(itop_name, 'r') as f:
        nx, ny, nz = [int(s) for s in f.readline().split(',')]
        x = np.fromstring(f.readline(), sep=' ', dtype='f') * -1
        y = np.fromstring(f.readline(), sep=' ', dtype='f') * -1
        z = np.fromstring(f.readline(), sep=' ', dtype='f')
        itop = np.fromstring(f.readline(), sep=' ', dtype='f')
    #itop = np.where(itop == 3, np.nan, itop) # this should already be?
    topo_arr = 1.5 * itop
    print(np.min(itop), np.max(itop))
    print(np.min(topo_arr), np.max(topo_arr))
    # print("!!!!", np.min(topo_arr), np.max(topo_arr))

    if path.exists(ihem_name):
        with open(ihem_name, 'r') as f:
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            ihem = np.fromstring(f.readline(), sep=' ', dtype='f')
            #print("HHHH", np.min(ihem), np.max(ihem))
            #ihem = np.where(itop == 1, ihem, 0) # this should already be?
            topo_arr += 0.5 * ihem
    else:
        warn("ihem file {0} not found, hem will be ignored".format(ihem_name))
    #print("!!!!", np.min(topo_arr), np.max(topo_arr))
    print(np.min(topo_arr), np.max(topo_arr))

    crds = coordinate.wrap_crds([('z', z), ('y', y), ('x', x)], type="Rectilinear")
    fld = field.wrap_field("Topology", "Node", crds, topo_arr.reshape((nz, ny, nx)),
                           0.0, "Scalar")

    # rr = tvtk.RectilinearGrid()
    # rr.point_data.scalars = topo_arr
    # rr.dimensions = (nx, ny, nz)
    # rr.point_data.scalars.name = 'Topology'
    # rr.x_coordinates = x
    # rr.y_coordinates = y
    # rr.z_coordinates = z
    # rr_src = VTKDataSource(data=rr)
    return fld

def shear_angle(topo_fld, b_fld, safety):
    mpx_lst = []  # magnetopause x position
    msx_lst = []  # magnetosheath x position
    y_lst = []
    z_lst = []
    shear_lst = []
    ixlst = []
    iylst = []
    izlst = []

    rad_to_deg = 180.0 / np.pi

    topo_x, topo_y, topo_z = topo_fld.crds[('x', 'y', 'z')]
    nx, ny, nz = topo_fld.crds.get_shape(['x', 'y', 'z'])
    shear_array = np.empty((nz, ny), dtype='f') * np.nan
    #shear_array = np.zeros((nz, ny), dtype='f')

    # 2d array for 2d plotting

    #print(topo_x[nz/2], topo_y[ny/2], topo_x[nx-1])
    #print(topo_fld[nz/2,ny/2,nx-1])

    for iz in range(nz):
        for iy in range(ny):
            arm = 0
            impause = None
            imsheath = None

            if topo_fld[iz,iy,0] == 0.0:
                #print(topo_fld[iz,iy,nx-1])
                arm=True
            else:
                pass
                # continue # uncomment to ignore above/below the cusps


            # do x backward, closed -> open
            # we already know ix-1 is a closed line
            # in russia, backward is foreward
            # (coord_x & coord_y monotonically decrease due to JimmyCrd -> GSE)
            for ix in range(1, nx):
                itop = topo_fld[iz,iy,ix]
                if not arm:
                    if itop == 0.0:
                        arm = True
                    continue

                #print(ix, iy, iz, topo_fld[iz, iy, ix])
                if impause is None:
                    # finding magnetopause
                    # print(itop)
                    if itop != 0.0 and itop != 4.5: # 4.5 is unknown topo
                        impause = ix + 1
                        # print("AAA")
                elif imsheath is None:
                    # finding magnetosheath
                    if itop == 3.0:
                        imsheath = ix + 1
                        # print("BBB")
                        break

            # if we did not see sw field lines along this line
            if imsheath is None:
                continue

            pt_mpause = (topo_x[impause] + safety, topo_y[iy], topo_z[iz])
            pt_msheath = (topo_x[imsheath] - safety, topo_y[iy], topo_z[iz])

            b_mpause = calc.nearest_val(b_fld, pt_mpause)
            b_msheath = calc.nearest_val(b_fld, pt_msheath)
            maga = np.linalg.norm(b_mpause)
            magb = np.linalg.norm(b_msheath)
            cosangle = np.dot(b_mpause, b_msheath) / (maga * magb)
            sinangle = np.cross(b_mpause, b_msheath) / (maga * magb)
            angle = np.arccos(cosangle) * rad_to_deg
            #angle = np.arctan()

            # if (iy==64 or iy==96) and iz==64:
            #     print("pt mpause: ", pt_mpause, "   pt msheath: ", pt_msheath)
            #     print("b mpause: ", b_mpause, "   b msheath: ", b_msheath)
            #     print(maga, magb)
            #     print(cosangle, "angle = ", angle)
            mpx_lst.append(pt_mpause[0] - safety)  # recorrect out the safety distance
            msx_lst.append(pt_msheath[0] + safety)  # recorrect out the safety distance
            y_lst.append(pt_mpause[1])
            z_lst.append(pt_mpause[2])
            shear_lst.append(angle)
            shear_array[iz, iy] = angle
            ixlst.append(impause)
            iylst.append(iy)
            izlst.append(iz)

    shear_crds = coordinate.wrap_crds([('z', topo_fld.crds['z']), ('y', topo_fld.crds['y'])],
                                      "Rectilinear")
    shear_fld = field.wrap_field("Shear Angle", "Node", shear_crds,
                                shear_array, time=topo_fld.time, type="Scalar")
    #print(min(shear_lst), max(shear_lst))
    return mpx_lst, msx_lst, y_lst, z_lst, shear_lst, shear_fld, ixlst, iylst, izlst

def find_xpoint(topo_fld):
    dat = topo_fld.data
    x, y, z = topo_fld.crds[('x', 'y', 'z')]
    nz, ny, nx = dat.shape

    list_x = []
    list_y = []
    list_z = []

    for iz in range(1, nz-1):
        for iy in range(1, ny-1):
            for ix in range(1, nx-1):
                itopo = dat[iz, iy, ix]
                if itopo == 1.0 or itopo == 2.0:
                    nfound = np.zeros((5,))
                    nfound[int(itopo)] += 1

                    ## search all neighbors 1 node away
                    # scan neighbors for matches (faces)
                    for L in [-1, 1]:
                        nfound[int(dat[ix+L, iy, iz])] += 1
                        nfound[int(dat[ix, iy+L, iz])] += 1
                        nfound[int(dat[ix, iy, iz+L])] += 1

                    # scan neighbors for matches (corners)
                    for L in [-1, 1]:
                        #print(ix, iy, iz, L, dat[ix+L, iy+1, iz+1])

                        nfound[int(dat[ix+L, iy+1, iz+1])] += 1
                        nfound[int(dat[ix+L, iy+1, iz-1])] += 1
                        nfound[int(dat[ix+L, iy-1, iz+1])] += 1
                        nfound[int(dat[ix+L, iy-1, iz-1])] += 1

                    # scan neighbors for matches (edges)
                    for L in [-1, 1]:
                        nfound[int(dat[ix+L, iy+1, iz])] += 1
                        nfound[int(dat[ix+L, iy-1, iz])] += 1
                        nfound[int(dat[ix+L, iy, iz+1])] += 1
                        nfound[int(dat[ix+L, iy, iz-1])] += 1

                    # remaining edges
                    nfound[int(dat[ix, iy+1, iz])] += 1
                    nfound[int(dat[ix, iy-1, iz])] += 1
                    nfound[int(dat[ix, iy, iz+1])] += 1
                    nfound[int(dat[ix, iy, iz-1])] += 1

                    #if (nfound[0]*nfound[1]*nfound[3]) != 0.0 or (nfound[0]*nfound[2]*nfound[3]) != 0:
                    if (nfound[0]*nfound[1]*nfound[2]*nfound[3]) != 0.0:
                        list_x.append(x[ix])
                        list_y.append(y[iy])
                        list_z.append(z[iz])

    return list_x, list_y, list_z


def find_xpoint_fast(topo_fld, mpause_ix, mpause_iy, mpause_iz):
    dat = topo_fld.data
    x, y, z = topo_fld.crds[('x', 'y', 'z')]
    nz, ny, nx = dat.shape

    list_x = []
    list_y = []
    list_z = []

    for ix, iy, iz in zip(mpause_ix, mpause_iy, mpause_iz):
        if ix == 0 or ix == nx - 2:
            continue
        if iy == 0 or iy == ny - 2:
            continue
        if iz == 0 or iz == nz - 2:
            continue

        nfound = np.zeros((5,))

        itopo = dat[iz, iy, ix]

        # scan neighbors for topology
        for niz in range(-1, 2):
            for niy in range(-1, 2):
                for nix in range(-1, 2):
                    nfound[int(dat[iz+niz, iy+niy, ix+nix])] += 1

        #if (nfound[0]*nfound[1]*nfound[3]) > 0.0 or (nfound[0]*nfound[2]*nfound[3]) > 0:
        if (nfound[0]*nfound[1]*nfound[2]*nfound[3]) > 0.0:
            print('isect: ', x[ix], y[iy], z[iz], nfound)
            list_x.append(x[ix])
            list_y.append(y[iy])
            list_z.append(z[iz])

    return list_x, list_y, list_z
