#!/usr/bin/env python
"""Minimum Variance Analysis and boundary normal crd tools"""

from __future__ import print_function, division
import os

import numpy as np
import viscid


__all__ = ["paraboloid", "paraboloid_normal", "fit_paraboloid",
           "get_mp_info", "find_mp_edges"]

_dtf = 'f8'
_paraboloid_dt = np.dtype([('x0', _dtf), ('y0', _dtf), ('z0', _dtf),
                           ('ax', _dtf), ('ay', _dtf), ('az', _dtf)])


def paraboloid(y, z, x0, y0, z0, ax, ay, az):
    """Generic paraboloid function"""
    return ax * (((y - y0) / ay)**2 + ((z - z0) / az)**2) + x0

def paraboloid_normal(y, z, x0, y0, z0, ax, ay, az, normalize=True):  # pylint: disable=unused-argument
    """Normal vector of a generic paraboloid"""
    dyF = 2.0 * (y - y0) / ay**2
    dzF = 2.0 * (z - z0) / az**2
    dxF = (-1.0 / ax) * np.ones_like(dyF)

    normal = np.array([dxF, dyF, dzF])
    if normalize:
        normal = normal / np.linalg.norm(normal, axis=0)
    return normal

def fit_paraboloid(fld, p0=(9.0, 0.0, 0.0, 1.0, -1.0, -1.0), tolerance=0.0):
    """Fit paraboloid it GSE coordinates x ~ y**2 + z**2

    Args:
        fld (:py:class:`viscid.field.ScalarField`): field of x values
        p0 (sequence): initial guess for parabaloid
            (x0, y0, z0, ax, ay, az), where (x0, y0, z0) is the nose
            location (should be subsolar for 0 dipole tilt), and the
            ay, az, and az coefficients determine the curvature

    Returns:
        numpy.recarray: record array of parameters with length 2; the
            1st value is the fit value, and the 2nd is one sigma of
            the fit
    """
    from scipy.optimize import curve_fit

    def paraboloid_yz(yz, x0, y0, z0, ax, ay, az):
        return paraboloid(yz[0], yz[1], x0, y0, z0, ax, ay, az)

    Y, Z = fld.meshgrid_flat(prune=True)
    popt, pcov = curve_fit(paraboloid_yz, np.vstack((Y, Z)),
                           fld.data.reshape(-1), p0=p0)
    perr = np.sqrt(np.diag(pcov))
    parab = np.recarray([2], dtype=_paraboloid_dt)
    # wow.. needing a nested loop to fill a recarray is stooopid,
    # am i missing something about recarrays?
    for i, vals in zip(range(len(parab)), [popt, perr]):
        for name, val in zip(parab.dtype.names, vals):
            parab[i][name] = val
    if tolerance:
        for n in parab.dtype.names:
            if n != "ax" and  np.abs(parab[1][n] / parab[0][n]) > tolerance:
                viscid.logger.warning("paraboloid parameter {0} didn't converge to "
                                      "within {1:g}%\n{0} = {2:g} +/- {3:g}"
                                      "".format(n, 100 * tolerance, parab[0][n],
                                                parab[1][n]))
    return parab

def get_mp_info(pp, b, j, e, cache=True, cache_dir=None,
                slc="x=5.5j:11.0j, y=-4.0j:4.0j, z=-3.6j:3.6j",
                fit="mp_xloc", fit_p0=(9.0, 0.0, 0.0, 1.0, -1.0, -1.0)):
    """Get info about m-pause as flattened fields

    Notes:
        The first thing this function does is mask locations where
        the GSE-y current density < 1e-4. This masks out the bow
        shock and current free regions. This works for southward IMF,
        but it is not very general.

    Parameters:
        pp (ScalarcField): pressure
        b (VectorField): magnetic field
        j (VectorField): current density
        e (VectorField, None): electric field (same centering as b). If
            None, then the info that requires E will be filled with NaN
        cache (bool, str): Save to and load from cache, if "force",
            then don't load from cache if it exists, but do save a
            cache at the end
        cache_dir (str): Directory for cache, if None, same directory
            as that file to which the grid belongs
        slc (str): slice that gives a box that contains the m-pause
        fit (str): to which resulting field should the paraboloid be fit,
            defaults to mp_xloc, but pp_max_xloc might be useful in some
            circumstances
        fit_p0 (tuple): Initial guess vector for paraboloid fit

    Returns:
        dict: Unless otherwise noted, the entiries are 2D (y-z) fields

          - **mp_xloc** location of minimum abs(Bz), this works
            better than max of J^2 for FTEs
          - **mp_sheath_edge** location where Jy > 0.1 * Jy when
            coming in from the sheath side
          - **mp_sphere_edge** location where Jy > 0.1 * Jy when
            coming in from the sphere side
          - **mp_width** difference between m-sheath edge and
            msphere edge
          - **mp_shear** magnetic shear taken 6 grid points into
            the m-sheath / m-sphere
          - **pp_max** max pp
          - **pp_max_xloc** location of max pp
          - **epar_max** max e parallel
          - **epar_max_xloc** location of max e parallel
          - **paraboloid** numpy.recarray of paraboloid fit. The
            parameters are given in the 0th element, and
            the 1st element contains the 1-sigma values for the fit

    Raises:
        RuntimeError: if using MHD crds instead of GSE crds
    """
    if not cache_dir:
        cache_dir = pp.find_info("_viscid_dirname", "./")
    run_name = pp.find_info("run", None)
    if cache and run_name:
        t = pp.time
        mp_fname = "{0}/{1}.mpause.{2:06.0f}".format(cache_dir, run_name, t)
    else:
        mp_fname = ""

    try:
        force = cache.strip().lower() == "force"
    except AttributeError:
        force = False

    try:
        if force or not mp_fname or not os.path.isfile(mp_fname + ".xdmf"):
            raise IOError()

        mp_info = {}
        with viscid.load_file(mp_fname + ".xdmf") as dat:
            fld_names = ["mp_xloc", "mp_sheath_edge", "mp_sphere_edge",
                         "mp_width", "mp_shear", "pp_max", "pp_max_xloc",
                         "epar_max", "epar_max_xloc"]
            for fld_name in fld_names:
                mp_info[fld_name] = dat[fld_name]["x=0"]

    except (IOError, KeyError):
        mp_info = {}

        crd_system = viscid.as_crd_system(b, None)
        if crd_system != 'gse':
            raise RuntimeError("get_mp_info can't work in MHD crds, "
                               "switch to GSE please")

        if j.nr_patches == 1:
            pp_block = pp[slc]
            b_block = b[slc]
            j_block = j[slc]
            if e is None:
                e_block = np.nan * viscid.empty_like(j_block)
            else:
                e_block = e[slc]
        else:
            # interpolate an amr grid so we can proceed
            obnd = pp.get_slice_extent(slc)
            dx = np.min(pp.skeleton.L / pp.skeleton.n, axis=0)
            nx = np.ceil((obnd[1] - obnd[0]) / dx)
            vol = viscid.seed.Volume(obnd[0], obnd[1], nx, cache=True)
            pp_block = vol.wrap_field(viscid.interp_trilin(pp, vol),
                                      name="P").as_cell_centered()
            b_block = vol.wrap_field(viscid.interp_trilin(b, vol),
                                     name="B").as_cell_centered()
            j_block = vol.wrap_field(viscid.interp_trilin(j, vol),
                                     name="J").as_cell_centered()
            if e is None:
                e_block = np.nan * viscid.empty_like(j_block)
            else:
                e_block = vol.wrap_field(viscid.interp_trilin(e, vol),
                                         name="E").as_cell_centered()

        # jsq = viscid.dot(j_block, j_block)
        bsq = viscid.dot(b_block, b_block)

        # extract ndarrays and mask out bow shock / current free regions
        maskval = 1e-4
        jy_mask = j_block['y'].data < maskval
        masked_bsq = 1.0 * bsq
        masked_bsq.data = np.ma.masked_where(jy_mask, bsq)

        xcc = j_block.get_crd_cc('x')
        nx = len(xcc)

        mp_xloc = np.argmin(masked_bsq, axis=0)  # indices
        mp_xloc = mp_xloc.wrap(xcc[mp_xloc.data])  # location

        pp_max = np.max(pp_block, axis=0)
        pp_max_xloc = np.argmax(pp_block, axis=0)  # indices
        pp_max_xloc = pp_max_xloc.wrap(xcc[pp_max_xloc.data])  # location

        epar = viscid.project(e_block, b_block)
        epar_max = np.max(epar, axis=0)
        epar_max_xloc = np.argmax(epar, axis=0)  # indices
        epar_max_xloc = pp_max_xloc.wrap(xcc[epar_max_xloc.data])  # location

        _ret = find_mp_edges(j_block, 0.1, 0.1, maskval=maskval)
        sheath_edge, msphere_edge, mp_width, sheath_ind, sphere_ind = _ret

        # extract b and b**2 at sheath + 6 grid points and sphere - 6 grid pointns
        # clipping cases where things go outside the block. clipped ponints are
        # set to nan
        step = 6
        # extract b
        if b_block.layout == "flat":
            comp_axis = 0
            ic, _, iy, iz = np.ix_(*[np.arange(si) for si in b_block.shape])
            ix = np.clip(sheath_ind + step, 0, nx - 1)
            b_sheath = b_block.data[ic, ix, iy, iz]
            ix = np.clip(sheath_ind - step, 0, nx - 1)
            b_sphere = b_block.data[ic, ix, iy, iz]
        elif b_block.layout == "interlaced":
            comp_axis = 3
            _, iy, iz = np.ix_(*[np.arange(si) for si in b_block.shape[:-1]])
            ix = np.clip(sheath_ind + step, 0, nx - 1)
            b_sheath = b_block.data[ix, iy, iz]
            ix = np.clip(sheath_ind - step, 0, nx - 1)
            b_sphere = b_block.data[ix, iy, iz]
        # extract b**2
        bmag_sheath = np.sqrt(np.sum(b_sheath**2, axis=comp_axis))
        bmag_sphere = np.sqrt(np.sum(b_sphere**2, axis=comp_axis))
        costheta = (np.sum(b_sheath * b_sphere, axis=comp_axis) /
                    (bmag_sphere * bmag_sheath))
        costheta = np.where((sheath_ind + step < nx) & (sphere_ind - step >= 0),
                            costheta, np.nan)
        mp_shear = mp_width.wrap((180.0 / np.pi) * np.arccos(costheta))

        # don't bother with pretty name since it's not written to file
        # plane_crds = b_block.crds.slice_keep('x=0', cc=True)
        # fld_kwargs = dict(center="Cell", time=b.time)
        mp_width.name = "mp_width"
        mp_xloc.name = "mp_xloc"
        sheath_edge.name = "mp_sheath_edge"
        msphere_edge.name = "mp_sphere_edge"
        mp_shear.name = "mp_shear"
        pp_max.name = "pp_max"
        pp_max_xloc.name = "pp_max_xloc"
        epar_max.name = "epar_max"
        epar_max_xloc.name = "epar_max_xloc"

        mp_info = {}
        mp_info["mp_width"] = mp_width
        mp_info["mp_xloc"] = mp_xloc
        mp_info["mp_sheath_edge"] = sheath_edge
        mp_info["mp_sphere_edge"] = msphere_edge
        mp_info["mp_shear"] = mp_shear
        mp_info["pp_max"] = pp_max
        mp_info["pp_max_xloc"] = pp_max_xloc
        mp_info["epar_max"] = epar_max
        mp_info["epar_max_xloc"] = epar_max_xloc

        # cache new fields to disk
        if mp_fname:
            viscid.save_fields(mp_fname + ".h5", list(mp_info.values()))

    try:
        _paraboloid_params = fit_paraboloid(mp_info[fit], p0=fit_p0)
        mp_info["paraboloid"] = _paraboloid_params
    except ImportError as _exception:
        try:
            msg = _exception.message
        except AttributeError:
            msg = _exception.msg
        mp_info["paraboloid"] = viscid.DeferredImportError(msg)

    mp_info["mp_width"].pretty_name = "Magnetopause Width"
    mp_info["mp_xloc"].pretty_name = "Magnetopause $X_{gse}$ Location"
    mp_info["mp_sheath_edge"].pretty_name = "Magnetosheath Edge"
    mp_info["mp_sphere_edge"].pretty_name = "Magnetosphere Edge"
    mp_info["mp_shear"].pretty_name = "Magnetic Shear"
    mp_info["pp_max"].pretty_name = "Max Pressure"
    mp_info["pp_max_xloc"].pretty_name = "Max Pressure Location"
    mp_info["epar_max"].pretty_name = "Max E Parallel"
    mp_info["epar_max_xloc"].pretty_name = "Max E Parallel Location"

    return mp_info

def find_mp_edges(j_block, msphere_thresh=0.1, sheath_thresh=0.1,
                  maskval=1e-4):
    """Find x location of msphere and msheath edges using current (J)

    Note:
        GSE coordinates only please

    Args:
        j_block (VectorField): Current density containing the whole
            magnetopause
        msphere_thresh (float): thereshold of current on the
            magnetosphere side as a fraction of the maximum
            current density, i.e., 0.1 is 10% of the max
        sheath_thresh (float): thereshold of current on the
            magnetosheath side as a fraction of the maximum
            current density, i.e., 0.1 is 10% of the max
        maskval (float, None): if not None, then mask out J values
            less than maskval; useful for masking out bowshock, and
            current free regions

    Returns:
        tuple: sheath and sphere fields / values

          - **sheath_edge**: float or 2D ScalarField of x values
          - **msphere_edge**: float or 2D ScalarField of x values
          - **mp_width**: **sheath_edge** - **msphere_edge**
          - **sheath_ind**: index of sheath_edge x location
          - **sphere_ind**: index of msphere_edge x location

    """
    if maskval is not None:
        jy_mask = j_block['y'].data < maskval
    else:
        jy_mask = np.zeros_like(j_block['y'].data, dtype='bool')

    xcc = j_block.get_crd_cc('x')
    nx = len(xcc)

    masked_jy = 1.0 * j_block['y']
    masked_jy.data = np.ma.masked_where(jy_mask, j_block['y'])

    jy_absmax = np.amax(np.abs(masked_jy), axis=0, keepdims=True)

    msphere_mask = (masked_jy > msphere_thresh * jy_absmax)
    sheath_mask = (masked_jy > sheath_thresh * jy_absmax)
    jy_absmax = None

    sphere_ind = np.argmax(msphere_mask, axis=0)
    if isinstance(sphere_ind, viscid.field.Field):
        msphere_edge = np.where(sphere_ind > 0, xcc[sphere_ind.data], np.nan)
        msphere_edge = sphere_ind.wrap(msphere_edge)
    else:
        msphere_edge = np.where(sphere_ind > 0, xcc[sphere_ind], np.nan)

    # reverse it to go from the other direction
    sheath_ind = nx - 1 - np.argmax(sheath_mask['x=::-1'], axis=0)
    if isinstance(sheath_ind, viscid.field.Field):
        sheath_edge = np.where(sheath_ind < (nx - 1), xcc[sheath_ind.data], np.nan)
        sheath_edge = sheath_ind.wrap(sheath_edge)
    else:
        sheath_edge = np.where(sheath_ind < (nx - 1), xcc[sheath_ind], np.nan)

    # in MHD crds, it my be sufficient to swap msp and msh at this point
    mp_width = sheath_edge - msphere_edge
    return sheath_edge, msphere_edge, mp_width, sheath_ind, sphere_ind

def _main():
    f = viscid.load_file("$WORK/xi_fte_001/*.3d.[4050f].xdmf")
    mp = get_mp_info(f['pp'], f['b'], f['j'], f['e_cc'], fit='mp_xloc',
                     slc="x=6.5j:10.5j, y=-4j:4j, z=-4.8j:3j", cache=False)

    y, z = mp['pp_max_xloc'].meshgrid_flat(prune=True)
    x = mp['pp_max_xloc'].data.reshape(-1)

    Y, Z = mp['pp_max_xloc'].meshgrid(prune=True)
    x2 = paraboloid(Y, Z, *mp['paraboloid'][0])

    skip = 117
    n = paraboloid_normal(Y, Z, *mp['paraboloid'][0]).reshape(3, -1)[:, ::skip]

    minvar_y = Y.reshape(-1)[::skip]
    minvar_z = Z.reshape(-1)[::skip]
    minvar_n = np.zeros([3, len(minvar_y)])
    for i in range(minvar_n.shape[0]):
        p0 = [0.0, minvar_y[i], minvar_z[i]]
        p0[0] = mp['pp_max_xloc']['y={0[0]}f, z={0[1]}f'.format(p0)]
        minvar_n[:, i] = viscid.find_minvar_lmn_around(f['b'], p0, l=2.0, n=64)[2, :]

    # 2d plots, normals don't look normal in the matplotlib projection
    if False:  # pylint: disable=using-constant-test
        from viscid.plot import vpyplot as vlt
        from matplotlib import pyplot as plt

        normals = paraboloid_normal(Y, Z, *mp['paraboloid'][0])
        p0 = np.array([x2, Y, Z]).reshape(3, -1)
        p1 = p0 + normals.reshape(3, -1)

        vlt.scatter_3d(np.vstack([x, y, z])[:, ::skip], equal=True)
        for i in range(0, p0.shape[1], skip):
            plt.gca().plot([p0[0, i], p1[0, i]],
                               [p0[1, i], p1[1, i]],
                               [p0[2, i], p1[2, i]], color='c')
        # z2 = _ellipsiod(X, Y, *popt)
        plt.gca().plot_surface(Y, Z, x2, color='r')
        vlt.show()

    # mayavi 3d plots, normals look better here
    if True:  # pylint: disable=using-constant-test
        from viscid.plot import vlab
        vlab.points3d(x[::skip], y[::skip], z[::skip], scale_factor=0.25,
                      color=(0.0, 0.0, 1.0))

        mp_width = mp['mp_width']['x=0']
        mp_sheath_edge = mp['mp_sheath_edge']['x=0']
        mp_sphere_edge = mp_sheath_edge - mp_width

        vlab.mesh(x2, Y, Z, scalars=mp_width.data)
        vlab.mesh(mp_sheath_edge.data, Y, Z, opacity=0.75, color=(0.75, ) * 3)
        vlab.mesh(mp_sphere_edge.data, Y, Z, opacity=0.75, color=(0.75, ) * 3)

        n = paraboloid_normal(Y, Z, *mp['paraboloid'][0]).reshape(3, -1)[:, ::skip]
        vlab.quiver3d(x2.reshape(-1)[::skip],
                      Y.reshape(-1)[::skip],
                      Z.reshape(-1)[::skip],
                      n[0], n[1], n[2], color=(1, 0, 0))
        vlab.quiver3d(x2.reshape(-1)[::skip],
                      Y.reshape(-1)[::skip],
                      Z.reshape(-1)[::skip],
                      minvar_n[0], minvar_n[1], minvar_n[2], color=(0, 0, 1))
        vlab.show()

if __name__ == "__main__":
    import sys  # pylint: disable=wrong-import-position,wrong-import-order
    sys.exit(_main())


##
## EOF
##
