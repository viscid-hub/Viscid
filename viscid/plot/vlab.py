"""Convevience module for making 3d plots with Mayavi

Note:
    You can't set rc parameters for this module!
"""
from __future__ import print_function, division
import os
import sys

import numpy as np
import mayavi
from mayavi import mlab
from mayavi.modules.axes import Axes
from mayavi.sources.builtin_surface import BuiltinSurface
from mayavi.sources.vtk_data_source import VTKDataSource
from traits.trait_errors import TraitError
from tvtk.api import tvtk
import viscid
from viscid import NOT_SPECIFIED
from viscid import field


# anything placed in here is not multiprocessing-safe
_global_ns = dict()


def add_source(src, figure=None):
    """Add a vtk data source to a figure

    Args:
        src (VTKDataSource): Description
        figure (mayavi.core.scene.Scene): specific figure, or None for
            :py:func:`mayavi.mlab.gcf`

    Returns:
        None
    """
    if not figure:
        figure = mlab.gcf()

    if src not in figure.children:
        engine = figure.parent
        engine.add_source(src, scene=figure)
    return src

def add_lines(lines, scalars=None, figure=None, name="NoName"):
    """Add list of lines to a figure

    Args:
        lines (list): See :py:func:`lines2source`
        scalars (ndarray): See :py:func:`lines2source`
        figure (mayavi.core.scene.Scene): specific figure, or None for
            :py:func:`mayavi.mlab.gcf`
        name (str): name of vtk object

    Returns:
        :py:class:`mayavi.sources.vtk_data_source.VTKDataSource`
    """
    src = lines2source(lines, scalars=scalars, name=name)
    add_source(src, figure=figure)
    return src

def add_field(fld, figure=None, center="", name=""):
    """Add a Viscid Field to a mayavi figure

    Args:
        fld (Field): Some Viscid Field
        figure (mayavi.core.scene.Scene): specific figure, or None for
            :py:func:`mayavi.mlab.gcf`
        center (str): 'cell' or 'node', leave blank to use fld.center
        name (str): name of vtk object, leave black for fld.name

    Returns:
        :py:class:`mayavi.sources.vtk_data_source.VTKDataSource`
    """
    src = field2source(fld, center=center, name=name)
    add_source(src, figure=figure)
    return src

def points2source(vertices, scalars=None, name="NoName"):
    # if scalars:
    #     scalars = [scalars]
    verts, scalars, _, other = viscid.vutil.prepare_lines([vertices], scalars)

    src = mlab.pipeline.scalar_scatter(verts[0], verts[1], verts[2])
    if scalars is not None:
        if scalars.dtype == np.dtype('u1'):
            sc = tvtk.UnsignedCharArray()
            sc.from_array(scalars.T)
            scalars = sc
        src.mlab_source.dataset.point_data.scalars = scalars
        src.mlab_source.dataset.modified()
    src.name = name
    return src

def lines2source(lines, scalars=None, name="NoName"):
    """Turn a list of lines as ndarrays into vtk data source

    Args:
        lines (list): List of 3xN, 4xN, 6xN ndarrays of xyz, xyzs, or
            xyzrgb data for N points along the line. N need not be the
            same for all lines.
        scalars (ndarray, list): Scalars for each point, or each line.
            See :py:func:`viscid.vutil.prepare_lines` for more details
        name (str): name of vtk object

    Returns:
        :py:class:`mayavi.sources.vtk_data_source.VTKDataSource`

    See Also:
        * :py:func:`viscid.vutil.prepare_lines`
    """
    r = viscid.vutil.prepare_lines(lines, scalars, do_connections=True)
    lines, scalars, connections, other = r

    src = mlab.pipeline.line_source(lines[0], lines[1], lines[2])
    if scalars is not None:
        if scalars.dtype == np.dtype('u1'):
            sc = tvtk.UnsignedCharArray()
            sc.from_array(scalars.T)
            scalars = sc
        src.mlab_source.dataset.point_data.scalars = scalars
        src.mlab_source.dataset.modified()
    src.mlab_source.dataset.lines = connections
    src.name = name
    return src

def field2source(fld, center=None, name=None):
    """Convert a field to a vtk data source

    This dispatches to either :meth:`field_to_point_source` or
    :meth:`field_to_cell_source` depending on the centering of
    `fld`.

    Parameters:
        fld: field to convert
        center (str): Either "cell", "node", or "" to use the
            same centering as fld
        name (str): Add specific name. Leave as "" to use fld.name

    Returns:
        mayavi source

    Raises:
        NotImplementedError: If center (or fld.center) is not
            recognized
    """
    if not center:
        center = fld.center
    center = center.lower()

    if center == "node":
        src = field2point_source(fld, name=name)
    elif center == "cell":
        src = field2cell_source(fld, name=name)
    else:
        raise NotImplementedError("cell / node only for now")

    return src

def field2point_source(fld, name=None):
    """Convert a field to a vtk point data source"""
    grid, arr = _prep_field(fld)
    dat_target = grid.point_data

    if fld.iscentered("Cell"):
        grid.dimensions = tuple(fld.crds.shape_cc)
        grid.x_coordinates = fld.get_crd_cc(0)  # ('x')
        grid.y_coordinates = fld.get_crd_cc(1)  # ('y')
        grid.z_coordinates = fld.get_crd_cc(2)  # ('z')
    elif fld.iscentered("Node"):
        grid.dimensions = tuple(fld.crds.shape_nc)
        grid.x_coordinates = fld.get_crd_nc(0)  # ('x')
        grid.y_coordinates = fld.get_crd_nc(1)  # ('y')
        grid.z_coordinates = fld.get_crd_nc(2)  # ('z')
    else:
        raise ValueError("cell or node only please")

    src = _finalize_source(fld, arr, grid, dat_target)
    if name:
        src.name = name
    return src

def field2cell_source(fld, name=None):
    """Convert a field to a vtk cell data source"""
    grid, arr = _prep_field(fld)
    dat_target = grid.cell_data

    if fld.iscentered("Cell"):
        grid.dimensions = tuple(fld.crds.shape_nc)
        grid.x_coordinates = fld.get_crd_nc(0)  # ('x')
        grid.y_coordinates = fld.get_crd_nc(1)  # ('y')
        grid.z_coordinates = fld.get_crd_nc(2)  # ('z')
    elif fld.iscentered("Node"):
        raise NotImplementedError("can't do lossless cell data from nodes yet")
    else:
        raise ValueError("cell or node only please")

    src = _finalize_source(fld, arr, grid, dat_target)
    if name:
        src.name = name
    return src


def _prep_field(fld):
    grid = tvtk.RectilinearGrid()

    # note, the transpose operations are b/c fld.data is now xyz ordered,
    # but vtk expects zyx data

    if isinstance(fld, field.ScalarField):
        zyx_dat = fld.data.T
        arr = np.reshape(zyx_dat, (-1,))
        # vtk expects zyx data, but fld.data is now xyz
    elif isinstance(fld, field.VectorField):
        if fld.layout == field.LAYOUT_INTERLACED:
            zyx_dat = np.transpose(fld.data, (2, 1, 0, 3))
            arr = np.reshape(zyx_dat, (-1, 3))
        elif fld.layout == field.LAYOUT_FLAT:
            zyx_dat = np.transpose(fld.data, (0, 3, 2, 1))
            arr = np.reshape(np.rollaxis(zyx_dat, 0, len(fld.shape)), (-1, 3))
        else:
            raise ValueError()
    else:
        raise ValueError("Unexpected fld type: {0}".format(type(fld)))
    # swap endian if needed
    if str(arr.dtype).startswith(">"):
        arr = arr.byteswap().newbyteorder()
    return grid, arr

def _finalize_source(fld, arr, grid, dat_target):
    if isinstance(fld, field.ScalarField):
        dat_target.scalars = arr
        dat_target.scalars.name = fld.name
    elif isinstance(fld, field.VectorField):
        dat_target.vectors = arr
        dat_target.vectors.name = fld.name
    src = VTKDataSource(data=grid)
    src.name = fld.name
    return src

def _prep_vector_source(v_src, scalars):
    """Side-effect: v_src will be modified if scalars are given"""
    if isinstance(v_src, viscid.field.Field):
        v_src = field2source(v_src, center='node')

    if scalars is not None:
        if isinstance(scalars, viscid.field.Field):
            scalars = field2source(scalars, center='node')
        v_src._point_scalars_list.append(scalars.name)  # pylint: disable=protected-access
        v_src.data.point_data.scalars = scalars.data.point_data.scalars
        v_src.point_scalars_name = scalars.name
    return v_src, scalars

def scalar_cut_plane(src, center=None, **kwargs):
    """Wraps `mayavi.mlab.pipeline.scalar_cut_plane`

    Note:
        This function will automatically switch to the default
        Matplotlib colormap (or the one from your viscidrc file)

        If you call this multiple times with the same
        `viscid.field.Field`, you should consider using field2source
        yourself and passing the Mayavi source object

    Args:
        src (Mayavi Source or ScalarField): If src is a ScalarField,
            then the field is wrapped into a Mayavi Source and added
            to the figure
        center (str): centering for the Mayavi source, 'cell' will
            make the grid visible, while 'node' will interpolate
            between points
        **kwargs: Passed to `mayavi.mlab.pipeline.scalar_cut_plane`

    Keyword Arguments:
        cmap (str, None, False): see :py:func:`apply_cmap`
        alpha (number, sequence): see :py:func:`apply_cmap`
        clim (sequence): see :py:func:`apply_cmap`
        symmetric (bool): see :py:func:`apply_cmap`
        logscale (bool): see :py:func:`apply_cmap`

    Returns:
        `mayavi.modules.scalar_cut_plane.ScalarCutPlane`
    """
    if isinstance(src, viscid.field.Field):
        src = field2source(src, center=center)
    kwargs, cmap_kwargs = _extract_cmap_kwargs(kwargs)
    scp = mlab.pipeline.scalar_cut_plane(src, **kwargs)
    apply_cmap(scp, **cmap_kwargs)
    return scp

def vector_cut_plane(v_src, scalars=None, color_mode='vector', **kwargs):
    """Wraps `mayavi.mlab.pipeline.vector_cut_plane`

    Note:
        This function will automatically switch to the default
        Matplotlib colormap (or the one from your viscidrc file)

        If you call this multiple times with the same
        `viscid.field.Field`, you should consider using field2source
        yourself and passing the Mayavi source object

    Args:
        v_src (Mayavi Source, or VectorField): Vector to cut-plane. If
            a Mayavi Source, then it must be node centered.
        scalars (Mayavi Source, or ScalarField): Optional scalar data.
            If a Mayavi Source, then it must be node centered. This
            will enable scale_mode and color_mode by 'scalar'
        color_mode (str): Color by 'vector', 'scalar', or 'none'
        **kwargs: Passed to `mayavi.mlab.pipeline.vector_cut_plane`

    Keyword Arguments:
        cmap (str, None, False): see :py:func:`apply_cmap`
        alpha (number, sequence): see :py:func:`apply_cmap`
        clim (sequence): see :py:func:`apply_cmap`
        symmetric (bool): see :py:func:`apply_cmap`
        logscale (bool): see :py:func:`apply_cmap`

    Returns:
        `mayavi.modules.vector_cut_plane.VectorCutPlane`
    """
    v_src, scalars = _prep_vector_source(v_src, scalars)
    kwargs, cmap_kwargs = _extract_cmap_kwargs(kwargs)
    vcp = mlab.pipeline.vector_cut_plane(v_src, **kwargs)

    apply_cmap(vcp, mode='vector', **cmap_kwargs)
    apply_cmap(vcp, mode='scalar', **cmap_kwargs)

    vcp.glyph.color_mode = 'color_by_{0}'.format(color_mode.strip().lower())

    return vcp

def mesh_from_seeds(seeds, scalars=None, fill_holes=NOT_SPECIFIED, **kwargs):
    """Wraps `mayavi.mlab.mesh` for Viscid seed generators

    Note:
        This function will automatically switch to the default
        Matplotlib colormap (or the one from your viscidrc file)

    Args:
        seeds (Viscid.SeedGen): Some seed generator with a 2D mesh
            representation
        scalars (ndarray, ScalarField): data mapped onto the mesh,
            i.e., the result of viscid.interp_trilin(seeds, ...)
        **kwargs: Passed to `mayavi.mlab.mesh`

    Keyword Arguments:
        cmap (str, None, False): see :py:func:`apply_cmap`
        alpha (number, sequence): see :py:func:`apply_cmap`
        clim (sequence): see :py:func:`apply_cmap`
        symmetric (bool): see :py:func:`apply_cmap`
        logscale (bool): see :py:func:`apply_cmap`

    Returns:
        `mayavi.modules.surface.Surface`
    """
    if scalars is not None:
        vertices, scalars = seeds.wrap_mesh(scalars, fill_holes=fill_holes)
    else:
        vertices, = seeds.wrap_mesh(fill_holes=fill_holes)

    return mesh(vertices[0], vertices[1], vertices[2], scalars=scalars,
                **kwargs)

def mesh(x, y, z, scalars=None, **kwargs):
    """Wraps `mayavi.mlab.mesh`

    Note:
        This function will automatically switch to the default
        Matplotlib colormap (or the one from your viscidrc file)

    Args:
        x (TYPE): 2D array of vertices' x-values
        y (TYPE): 2D array of vertices' y-values
        z (TYPE): 2D array of vertices' z-values
        scalars (ndarray, ScalarField): optional scalar data
        **kwargs: Passed to `mayavi.mlab.mesh`

    Keyword Arguments:
        cmap (str, None, False): see :py:func:`apply_cmap`
        alpha (number, sequence): see :py:func:`apply_cmap`
        clim (sequence): see :py:func:`apply_cmap`
        symmetric (bool): see :py:func:`apply_cmap`
        logscale (bool): see :py:func:`apply_cmap`

    Returns:
        `mayavi.modules.surface.Surface`
    """
    if scalars is not None:
        if isinstance(scalars, viscid.field.Field):
            scalars = scalars.data
        scalars = scalars.reshape(x.shape)

    kwargs, cmap_kwargs = _extract_cmap_kwargs(kwargs)
    m = mlab.mesh(x, y, z, scalars=scalars, **kwargs)

    if scalars is not None:
        apply_cmap(m, **cmap_kwargs)
    return m

def quiver3d(*args, **kwargs):
    """Wraps `mayavi.mlab.quiver3d`

    Args:
        *args: passed to `mayavi.mlab.quiver3d`
        **kwargs: Other Arguments are popped, then kwargs is passed to
            `mayavi.mlab.quiver3d`

    Keyword Arguments:
        cmap (str, None, False): see :py:func:`apply_cmap`
        alpha (number, sequence): see :py:func:`apply_cmap`
        clim (sequence): see :py:func:`apply_cmap`
        symmetric (bool): see :py:func:`apply_cmap`
        logscale (bool): see :py:func:`apply_cmap`

    Returns:
        TYPE: Description
    """
    kwargs, cmap_kwargs = _extract_cmap_kwargs(kwargs)
    quivers = mlab.quiver3d(*args, **kwargs)
    apply_cmap(quivers, mode='scalar', **cmap_kwargs)
    apply_cmap(quivers, mode='vector', **cmap_kwargs)
    return quivers

def points3d(*args, **kwargs):
    """Wraps `mayavi.mlab.points3d`

    Args:
        *args: passed to `mayavi.mlab.points3d`
        **kwargs: Other Arguments are popped, then kwargs is passed to
            `mayavi.mlab.points3d`

    Keyword Arguments:
        modify_args (bool): if True (default), then check if args is a
            single 2d sequence of shape 3xN or Nx3. Then split them up
            appropriately. if False, then args are passed through
            to mlab.points3d unchanged, nomatter what.
        cmap (str, None, False): see :py:func:`apply_cmap`
        alpha (number, sequence): see :py:func:`apply_cmap`
        clim (sequence): see :py:func:`apply_cmap`
        symmetric (bool): see :py:func:`apply_cmap`
        logscale (bool): see :py:func:`apply_cmap`

    Returns:
        TYPE: Description
    """
    modify_args = kwargs.pop('modify_args', True)
    kwargs, cmap_kwargs = _extract_cmap_kwargs(kwargs)

    if modify_args and len(args) < 3:
        a0 = np.asarray(args[0])
        if len(a0.shape) > 1 and a0.shape[0] == 3:
            args = [a0[0, :].reshape(-1),
                    a0[1, :].reshape(-1),
                    a0[2, :].reshape(-1)] + list(args[1:])
        elif len(a0.shape) > 1 and a0.shape[1] == 3:
            args = [a0[:, 0].reshape(-1),
                    a0[:, 1].reshape(-1),
                    a0[:, 2].reshape(-1)] + list(args[1:])

    points = mlab.points3d(*args, **kwargs)
    apply_cmap(points, **cmap_kwargs)
    return points

def streamline(v_src, scalars=None, **kwargs):
    """Wraps `mayavi.mlab.pipeline.streamline`; mind the caveats

    Note:
        This function will automatically switch to the default
        Matplotlib colormap (or the one from your viscidrc file)

        Side-effect: If scalars are given, then v_src is modified to
        point to the scalar data!

        If v_src and scalars are Mayavi sources, they must be node
        centered.

        If you call this multiple times with the same v_src and
        scalars, you should consider using field2source yourself and
        passing the Mayavi source objects, unless you're using
        different scalars with the same vector field, since this
        function has side-effects on the vector sourc.

    Args:
        v_src (Mayavi Source, or VectorField): Vector to streamline. If
            a Mayavi Source, then it must be node centered.
        scalars (Mayavi Source, or ScalarField): Optional scalar data.
            If a Mayavi Source, then it must be node centered.
        **kwargs: Passed to `mayavi.mlab.mesh`

    Keyword Arguments:
        cmap (str, None, False): see :py:func:`apply_cmap`
        alpha (number, sequence): see :py:func:`apply_cmap`
        clim (sequence): see :py:func:`apply_cmap`
        symmetric (bool): see :py:func:`apply_cmap`
        logscale (bool): see :py:func:`apply_cmap`

    Returns:
        `mayavi.modules.streamline.Streamline`
    """
    v_src, scalars = _prep_vector_source(v_src, scalars)
    kwargs, cmap_kwargs = _extract_cmap_kwargs(kwargs)

    sl = mlab.pipeline.streamline(v_src, **kwargs)
    apply_cmap(sl, mode='vector', **cmap_kwargs)
    apply_cmap(sl, mode='scalar', **cmap_kwargs)
    return sl

def iso_surface(src, backface_culling=True, **kwargs):
    """Wraps `mayavi.mlab.pipeline.iso_surface`; mind the caveats

    Note that backfaces are culled by default.

    Note:
        This function will automatically switch to the default
        Matplotlib colormap (or the one from your viscidrc file)

        If src is a Mayavi source, it must be node centered.

        If you call this multiple times with the same
        `viscid.field.Field`, you should consider using field2source
        yourself and passing the Mayavi source object

    Args:
        src (Mayavi Source or ScalarField): If src is a ScalarField,
            then the field is wrapped into a Mayavi Source and added
            to the figure. If a Mayavi Source, then it must be node
            centered.
        backface_culling (bool): Cull backfaces by default. Useful for
            translucent surfaces.
        **kwargs: Passed to `mayavi.mlab.pipeline.scalar_cut_plane`

    Keyword Arguments:
        cmap (str, None, False): see :py:func:`apply_cmap`
        alpha (number, sequence): see :py:func:`apply_cmap`
        clim (sequence): see :py:func:`apply_cmap`
        symmetric (bool): see :py:func:`apply_cmap`
        logscale (bool): see :py:func:`apply_cmap`

    Returns:
        `mayavi.modules.iso_surface.IsoSurface`
    """
    if isinstance(src, viscid.field.Field):
        src = field2source(src, center='node')
    kwargs, cmap_kwargs = _extract_cmap_kwargs(kwargs)
    iso = mlab.pipeline.iso_surface(src, **kwargs)
    apply_cmap(iso, **cmap_kwargs)
    iso.actor.property.backface_culling = backface_culling
    return iso

def plot_line(line, scalars=None, **kwargs):
    """Wrap :py:func:`plot_lines` for a single line"""
    if scalars is not None:
        scalars = [scalars]
    return plot_lines([line], scalars=scalars, **kwargs)

def plot_lines(lines, scalars=None, style="tube", figure=None,
               name="Lines", tube_radius=0.05, tube_sides=6, **kwargs):
    """Make 3D mayavi plot of lines

    Scalars can be a bunch of single values, or a bunch of rgb data
    to set the color of each line / vertex explicitly. This is
    explained in :py:func:`lines2source`.

    Example:
        A common use case of setting the line color from a topology
        will want to use :py:func:`viscid.topology2color`::

            >>> import viscid
            >>> from viscid.plot import vlab
            >>>
            >>> B = viscid.make_dipole()
            >>> seeds = viscid.Line([-4, 0, 0], [4, 0, 0])
            >>> lines, topology = viscid.calc_streamlines(B, seeds,
            >>>                                           ibound=0.05)
            >>> scalars = viscid.topology2color(topology)
            >>> vlab.plot_lines(lines, scalars, tube_radius=0.02)
            >>> vlab.savefig("dipole.x3d")
            >>> viscid.meshlab_convert("dipole.x3d", "dae")
            >>> vlab.show()

    Parameters:
        lines (list): See :py:func:`lines2source`
        scalars (TYPE): See :py:func:`lines2source`
        style (str): 'tube' or 'none'
        figure (mayavi.core.scene.Scene): specific figure, or None for
            :py:func:`mayavi.mlab.gcf`
        name (str): Description
        tube_radius (float): Radius if style == 'tube'
        tube_sides (int): Angular resolution if style == 'tube'
        **kwargs: passed to :meth:`mayavi.mlab.pipeline.surface`. This
            is useful for setting a colormap among other things.

    Keyword Arguments:
        cmap (str, None, False): see :py:func:`apply_cmap`
        alpha (number, sequence): see :py:func:`apply_cmap`
        clim (sequence): see :py:func:`apply_cmap`
        symmetric (bool): see :py:func:`apply_cmap`
        logscale (bool): see :py:func:`apply_cmap`

    Returns:
        Mayavi surface module

    Raises:
        ValueError: if style is neither tube nor strip
    """
    style = style.lower()
    if not figure:
        figure = mlab.gcf()

    src = lines2source(lines, scalars=scalars, name=name)

    # always use the stripper since actually turns a collection of line
    # segments into a line... that way capping will cap lines, not line
    # segments, etc.
    lines = mlab.pipeline.stripper(src, figure=figure)
    if style == "tube":
        lines = mlab.pipeline.tube(lines, figure=figure, tube_radius=tube_radius,
                                   tube_sides=tube_sides)
    elif style == "none" or not style:
        pass
    else:
        raise ValueError("Unknown style for lines: {0}".format(style))

    kwargs, cmap_kwargs = _extract_cmap_kwargs(kwargs)
    surface = mlab.pipeline.surface(lines, **kwargs)
    apply_cmap(surface, **cmap_kwargs)
    return surface

def plot_ionosphere(fld, radius=1.063, figure=None, bounding_lat=0.0,
                    rotate=None, crd_system="gse", **kwargs):
    """Plot an ionospheric field

    Args:
        fld (Field): Some spherical (phi, theta) / (lot, lat) field
        radius (float): Defaults to 1Re + 400km == 1.063Re
        figure (mayavi.core.scene.Scene): specific figure, or None for
            :py:func:`mayavi.mlab.gcf`
        bounding_lat (float): Description
        rotate (None, sequence, str, datetime64): sequence of length 4
            that contains (angle, ux, uy, uz) for the angle and axis of
            a rotation, or a UT time as string or datetime64 to rotate
            earth to a specific date/time, or a cotr object in
            conjunction with crd_system
        crd_system (str, other): Used if rotate is datetime-like. Can
            be one of ('gse', 'mhd'), or anything that returns from
            :py:func:`viscid.as_crd_system`.
        **kwargs: passed to :py:func:`mayavi.mlab.mesh`

    Keyword Arguments:
        cmap (str, None, False): see :py:func:`apply_cmap`
        alpha (number, sequence): see :py:func:`apply_cmap`
        clim (sequence): see :py:func:`apply_cmap`
        symmetric (bool): see :py:func:`apply_cmap`
        logscale (bool): see :py:func:`apply_cmap`

    No Longer Raises:
        ValueError: Description
    """
    if figure is None:
        figure = mlab.gcf()

    fld = viscid.as_spherefield(fld, order=('phi', 'theta'), units='deg')
    phil, thetal = fld.xl
    phih, thetah = fld.xh
    nphi, ntheta = fld.shape

    sphere = viscid.Sphere([0, 0, 0], r=radius, ntheta=ntheta, nphi=nphi,
                           thetalim=(thetal, thetah), philim=(phil, phih),
                           theta_phi=False)
    verts, arr = sphere.wrap_mesh(fld.data, fill_holes=True)

    kwargs, cmap_kwargs = _extract_cmap_kwargs(kwargs)
    if 'name' not in kwargs:
        kwargs['name'] = fld.name
    m = mlab.mesh(verts[0], verts[1], verts[2], scalars=arr, figure=figure,
                  **kwargs)

    m.parent.parent.filter.splitting = 0

    if bounding_lat:
        rp = 1.5 * radius
        z = radius * np.cos((np.pi / 180.0) * bounding_lat)
        clip = mlab.pipeline.data_set_clipper(m.module_manager.parent)
        clip.widget.widget.place_widget(-rp, rp, -rp, rp, -z, z)
        clip.update_pipeline()
        clip.widget.widget.enabled = False
        insert_filter(clip, m.module_manager)
        # m.module_manager.parent.parent.filter.auto_orient_normals = True
    else:
        pass
        # m.module_manager.parent.filter.auto_orient_normals = True
    m.actor.mapper.interpolate_scalars_before_mapping = True

    apply_cmap(m, **cmap_kwargs)

    m.actor.actor.rotate_z(180)
    _apply_rotation(m, 'sm', rotate, crd_system=crd_system)

    return m

def plot_nulls(nulls, Acolor=(0.0, 0.263, 0.345), Bcolor=(0.686, 0.314, 0.0),
               Ocolor=(0.239, 0.659, 0.557), **kwargs):
    kwargs.setdefault('scale_mode', 'none')
    kwargs.setdefault('scale_factor', 0.3)

    if not isinstance(nulls, dict):
        empty = np.ones((3, 0))
        nulls = dict(O=[empty, nulls], A=[empty, empty], B=[empty, empty])

    Opts = nulls['O'][1]
    if Ocolor is not None and Opts.shape[1]:
        mlab.points3d(Opts[0], Opts[1], Opts[2], color=Ocolor, name="Onulls",
                      **kwargs)

    Apts = nulls['A'][1]
    if Ocolor is not None and Opts.shape[1]:
        mlab.points3d(Apts[0], Apts[1], Apts[2], color=Acolor, name="Anulls",
                      **kwargs)

    Bpts = nulls['B'][1]
    if Bcolor is not None and Bpts.shape[1]:
        mlab.points3d(Bpts[0], Bpts[1], Bpts[2], color=Bcolor, name="Bnulls",
                      **kwargs)

def fancy_axes(figure=None, target=None, nb_labels=5, xl=None, xh=None,
               tight=False, symmetric=False, padding=0.05, opacity=0.7,
               face_color=None, line_width=2.0, grid_color=None,
               labels=True, label_color=None, label_shadow=True,
               consolidate_labels=True):
    """Make axes with 3 shaded walls and a grid similar to matplotlib

    Args:
        figure (mayavi.core.scene.Scene): specific figure, or None for
            :py:func:`mayavi.mlab.gcf`
        target (Mayavi Element): If either xl or xh are not given, then
            get that limit from a bounding box around `target`
        nb_labels (int, sequence): number of labels in all, or each
            (x, y, z) directions
        xl (float, sequence): lower corner of axes
        xh (float, sequence): upper corner of axes
        tight (bool): If False, then let xl and xh expand to make nicer
            labels. This uses matplotlib to determine new extrema
        symmetric (bool): If True, then xl + xh = 0
        padding (float): add padding as a fraction of the total length
        opacity (float): opacity of faces
        face_color (sequence): color (r, g, b) of faces
        line_width (float): Width of grid lines
        grid_color (sequence): Color of grid lines
        labels (bool): Whether or not to put axis labels on
        label_color (sequence): color of axis labels
        label_shadow (bool): Add shadows to all labels
        consolidate_labels (bool): if all nb_labels are the same, then
            only make one axis for the labels

    Returns:
        VTKDataSource: source to which 2 surfaces and 3 axes belong
    """
    if figure is None:
        figure = mlab.gcf()

    # setup xl and xh
    if xl is None or xh is None:
        _outline = mlab.outline(target, figure=figure)

        if xl is None:
            xl = _outline.bounds[0::2]
        if xh is None:
            xh = _outline.bounds[1::2]
        _outline.remove()

    nb_labels = np.broadcast_to(nb_labels, (3,))
    xl = np.array(np.broadcast_to(xl, (3,)))
    xh = np.array(np.broadcast_to(xh, (3,)))
    L = xh - xl

    xl -= padding * L
    xh += padding * L

    # now adjust xl and xh to be prettier
    if symmetric:
        tight = False
    if not tight:
        from matplotlib.ticker import AutoLocator
        for i in range(len(xl)):  # pylint: disable=consider-using-enumerate
            l = AutoLocator()
            l.create_dummy_axis()
            l.set_view_interval(xl[i], xh[i])
            locs = l()
            xl[i] = locs[0]
            xh[i] = locs[-1]

    dx = (xh - xl) / (nb_labels - 1)
    grid = tvtk.ImageData(dimensions=nb_labels, origin=xl, spacing=dx)
    src = VTKDataSource(data=grid)
    src.name = "fancy_axes"

    if face_color is None:
        face_color = figure.scene.background
    if grid_color is None:
        grid_color = figure.scene.foreground
    if label_color is None:
        label_color = grid_color

    face = mlab.pipeline.surface(src, figure=figure, opacity=opacity,
                                 color=face_color)
    face.actor.property.frontface_culling = True

    if line_width:
        grid = mlab.pipeline.surface(src, figure=figure, opacity=1.0,
                                     color=grid_color, line_width=line_width,
                                     representation='wireframe')
        grid.actor.property.frontface_culling = True

    if labels:
        def _make_ax_for_labels(_i, all_axes=False):
            if all_axes:
                _ax = Axes(name='axes-labels')
            else:
                _ax = Axes(name='{0}-axis-labels'.format('xyz'[_i]))
                # VTK bug... y_axis and z_axis are flipped... how is VTK still
                # the de-facto 3d plotting library?
                if _i == 0:
                    _ax.axes.x_axis_visibility = True
                    _ax.axes.y_axis_visibility = False
                    _ax.axes.z_axis_visibility = False
                elif _i == 1:
                    _ax.axes.x_axis_visibility = False
                    _ax.axes.y_axis_visibility = False
                    _ax.axes.z_axis_visibility = True  # VTK bug
                elif _i == 2:
                    _ax.axes.x_axis_visibility = False
                    _ax.axes.y_axis_visibility = True  # VTK bug
                    _ax.axes.z_axis_visibility = False
                else:
                    raise ValueError()
            _ax.property.opacity = 0.0
            _ax.axes.number_of_labels = nb_labels[_i]
            # import IPython; IPython.embed()
            _ax.title_text_property.color = label_color
            _ax.title_text_property.shadow = label_shadow
            _ax.label_text_property.color = label_color
            _ax.label_text_property.shadow = label_shadow
            src.add_module(_ax)

        if consolidate_labels and np.all(nb_labels[:] == nb_labels[0]):
            _make_ax_for_labels(0, all_axes=True)
        else:
            _make_ax_for_labels(0, all_axes=False)
            _make_ax_for_labels(1, all_axes=False)
            _make_ax_for_labels(2, all_axes=False)

    return src

axes = mlab.axes
xlabel = mlab.xlabel
ylabel = mlab.ylabel
zlabel = mlab.zlabel
title = mlab.title
outline = mlab.outline
orientation_axes = mlab.orientation_axes
view = mlab.view

def _extract_cmap_kwargs(kwargs):
    cmap_kwargs = dict()
    cmap_kwargs["cmap"] = kwargs.pop("cmap", None)
    cmap_kwargs["alpha"] = kwargs.pop("alpha", None)
    cmap_kwargs["clim"] = kwargs.pop("clim", None)
    cmap_kwargs["symmetric"] = kwargs.pop("symmetric", False)
    cmap_kwargs["logscale"] = kwargs.pop("logscale", False)
    return kwargs, cmap_kwargs

def colorbar(*args, **kwargs):
    """Wraps mayavi.mlab.colorbar and adjusts cmap if you so choose"""
    cmap = kwargs.pop("cmap", False)
    kwargs, cmap_kwargs = _extract_cmap_kwargs(kwargs)
    cmap_kwargs.pop("cmap")
    ret = mlab.colorbar(*args, **kwargs)
    apply_cmap(ret, cmap=cmap, **cmap_kwargs)
    return ret

def scalarbar(*args, **kwargs):
    """Wraps mayavi.mlab.scalarbar and adjusts cmap if you so choose"""
    cmap = kwargs.pop("cmap", False)
    kwargs, cmap_kwargs = _extract_cmap_kwargs(kwargs)
    cmap_kwargs.pop("cmap")
    ret = mlab.scalarbar(*args, **kwargs)
    apply_cmap(ret, cmap=cmap, **cmap_kwargs)
    return ret

def vectorbar(*args, **kwargs):
    """Wraps mayavi.mlab.vectorbar and adjusts cmap if you so choose"""
    cmap = kwargs.pop("cmap", False)
    kwargs, cmap_kwargs = _extract_cmap_kwargs(kwargs)
    cmap_kwargs.pop("cmap")
    ret = mlab.vectorbar(*args, **kwargs)
    apply_cmap(ret, cmap=cmap, **cmap_kwargs)
    return ret

def get_cmap(cmap=None, lut=None, symmetric=False):
    """Get a Matplotlib colormap as an rgba ndarray

    Args:
        cmap (str): name of colormap, or an ndarray of rgb(a) colors
        lut (int): number of entries desired in the lookup table

    Returns:
        ndarray: Nx4 array of N rgba colors
    """
    import matplotlib
    if symmetric and not cmap:
        cmap = matplotlib.rcParams.get("viscid.symmetric_cmap", None)
    try:
        cm = matplotlib.cm.get_cmap(name=cmap, lut=lut)
        rgba = (255 * np.asarray(cm(np.linspace(0, 1, cm.N)))).astype('i')
    except TypeError:
        rgba = np.asarray(cmap)
        if np.all(rgba >= 0.0) and np.all(rgba <= 1.0):
            rgba = (255 * rgba).astype('i')
        else:
            rgba = rgba.astype('i')
            if np.any(rgba < 0) or np.any(rgba > 255):
                raise ValueError("cmap ndarray must have color values between "
                                 "0 and 255 or 0.0 and 1.0")

    if rgba.shape[1] not in (3, 4) and rgba.shape[0] in (3, 4):
        rgba = np.array(rgba.T)

    if rgba.shape[1] == 3:
        rgba = np.hstack([rgba, 255 * np.ones_like(rgba[:, :1])])

    return rgba

def apply_cmap(target, cmap=None, lut=None, alpha=None, mode='scalar',
               clim=None, symmetric=False, logscale=False):
    """Apply a Matplotlib colormap to a Mayavi object & adjust limits

    Args:
        target: Some Mayavi object on mode to apply the colormap
        cmap (sequence, None, False): name of a Matplotlib colormap, or
            a sequence of rgb(a) colors, or None to use the default,
            or False to leave the colormap alone.
        lut (int): number of entries desired in the lookup table
        alpha (number, sequence): scalar or array that sets the alpha
            (opacity) channel in the range [0..255]. This is expanded
            to both ends of the colormap using linear interpolation,
            i.e., [0, 255] will be a linear ramp from transparent to
            opaque over the whole colormap.
        mode (str): one of 'scalar', 'vector', or 'other'
        clim (sequence): contains (vmin, vmax) for color scale
        symmetric (bool): force the limits on the colorbar to be
            symmetric around 0, and if no `cmap` is given, then also
            use the default symmetric colormap
        logscale (bool): Use a logarithmic color scale

    Raises:
        AttributeError: Description
        ValueError: Description
    """
    mode = mode.strip().lower()

    # get the mayavi lut object
    try:
        if mode == "scalar":
            mvi_lut = target.module_manager.scalar_lut_manager.lut
        elif mode == "vector":
            mvi_lut = target.module_manager.vector_lut_manager.lut
        else:
            if mode != "other":
                raise ValueError("mode should be 'scalar', 'vector', or "
                                 "'other'; not '{0}'".format(mode))
            raise AttributeError()
    except AttributeError:
        mvi_lut = target.lut

    # set the limits on the colorbar
    if isinstance(clim, (list, tuple)):
        mvi_lut.range = [clim[0], clim[1]]
    elif clim == 0:
        symmetric = True
    elif clim:
        symmetric = clim

    if logscale and symmetric:
        viscid.logger.warning("logscale and symmetric are mutually exclusive;"
                              "ignoring symmetric.")

    if logscale:
        mvi_lut.scale = 'log10'
    elif symmetric:
        # float(True) -> 1
        val = float(symmetric) * np.max(np.abs(mvi_lut.range))
        mvi_lut.range = [-val, val]

    vmin, vmax = mvi_lut.range
    is_symmetric = bool(np.isclose(vmax, -1 * vmin, atol=0))

    # now set the colormap
    changed = False
    if cmap is False:
        rgba = None if alpha is None else mvi_lut.table.to_array()
    else:
        rgba = get_cmap(cmap=cmap, lut=lut, symmetric=is_symmetric)
        changed = True

    if alpha is not None:
        alpha = np.asarray(alpha).reshape(-1)
        rgba[:, -1] = np.interp(np.linspace(0, 1, len(rgba)),
                                np.linspace(0, 1, len(alpha)), alpha)
        changed = True

    if changed:
        mvi_lut.table = rgba

def insert_filter(filtr, module_manager):
    """Insert a filter above an existing module_manager

    Args:
        filter (TYPE): Description
        module_manager (TYPE): Description
    """
    filtr.parent.children.remove(module_manager)
    filtr.children.append(module_manager)

def _apply_rotation(obj, from_system, rotate=None, crd_system='gse'):
    if hasattr(rotate, "get_rotation_wxyz"):
        rotate = rotate.get_rotation_wxyz(from_system, crd_system)
    else:
        cotr = viscid.as_cotr(rotate)
        rotate = cotr.get_rotation_wxyz(from_system, crd_system)

    if len(rotate) != 4:
        raise ValueError("Rotate should be [angle, ux, uy, uz], got {0}"
                         "".format(rotate))
    obj.actor.actor.rotate_wxyz(*rotate)

def plot_blue_marble(r=1.0, figure=None, nphi=128, ntheta=64, map_style=None,
                     lines=False, res=2, rotate=None, crd_system='gse'):
    """Plot Earth using the Natural Earth dataset maps

    Args:
        r (float): radius of earth
        figure (mayavi.core.scene.Scene): specific figure, or None for
            :py:func:`mayavi.mlab.gcf`
        nphi (int): phi resolution of Earth's mesh
        ntheta (int): theta resolution of Earth's mesh
        map_style (str): Nothing for standard map, or 'faded'
        lines (bool): Whether or not to show equator, tropics,
            arctic circles, and a couple meridians.
        res (int): Resolution in thousands of pixels longitude (must
            be one of 1, 2, 4, 8)
        rotate (None, sequence, str, datetime64): sequence of length 4
            that contains (angle, ux, uy, uz) for the angle and axis of
            a rotation, or a UT time as string or datetime64 to rotate
            earth to a specific date/time, or a cotr object in
            conjunction with crd_system
        crd_system (str, other): Used if rotate is datetime-like. Can
            be one of ('gse', 'mhd'), or anything that returns from
            :py:func:`viscid.as_crd_system`.

    Returns:
        (VTKDataSource, mayavi.modules.surface.Surface)
    """
    # make a plane, then deform it into a sphere
    eps = 1e-4
    ps = tvtk.PlaneSource(origin=(r, r * np.pi - eps, r * 0.0),
                          point1=(r, r * np.pi - eps, r * 2 * np.pi),
                          point2=(r, eps, 0.0),
                          x_resolution=nphi,
                          y_resolution=ntheta)

    ps.update()
    transform = tvtk.SphericalTransform()
    tpoly = tvtk.TransformPolyDataFilter(transform=transform,
                                         input_connection=ps.output_port)
    tpoly.update()
    src = VTKDataSource(data=tpoly.output, name="blue_marble")
    surf = mlab.pipeline.surface(src)

    # now load a jpg, and use it to texture the sphere

    linestr = '_lines' if lines else ''
    assert map_style in (None, '', 'faded')
    assert res in (1, 2, 4, 8)
    map_style = '_{0}'.format(map_style) if map_style else ''
    img_name = "images/earth{0}{1}_{2}k.jpg".format(map_style, linestr, res)
    fname = os.path.realpath(os.path.dirname(__file__) + '/' + img_name)
    img = tvtk.JPEGReader(file_name=fname)
    texture = tvtk.Texture(input_connection=img.output_port, interpolate=1)
    surf.actor.enable_texture = True
    surf.actor.texture = texture
    surf.actor.property.color = (1.0, 1.0, 1.0)

    # rotate 180deg b/c i can't rotate the texture to make the prime meridian
    surf.actor.actor.rotate_z(180)
    _apply_rotation(surf, 'geo', rotate, crd_system=crd_system)

    add_source(src, figure=figure)

    return src, surf

plot_natural_earth = plot_blue_marble

def plot_earth_3d(figure=None, daycol=(1, 1, 1), nightcol=(0, 0, 0),
                  radius=1.0, res=24, crd_system="gse", night_only=False,
                  **kwargs):
    """Plot a black and white sphere (Earth) showing sunward direction

    Parameters:
        figure (mayavi.core.scene.Scene): specific figure, or None for
            :py:func:`mayavi.mlab.gcf`
        daycol (tuple, optional): color of dayside (RGB)
        nightcol (tuple, optional): color of nightside (RGB)
        res (optional): rosolution of teh sphere
        crd_system (str, other): One of ('mhd', 'gse'), or anything
            that returns from :py:func:`viscid.as_crd_system`.

    Returns:
        Tuple (day, night) as vtk sources
    """
    if figure is None:
        figure = mlab.gcf()

    crd_system = viscid.as_crd_system(crd_system)
    if crd_system == "mhd":
        theta_dusk, theta_dawn = 270, 90
    elif crd_system == "gse":
        theta_dusk, theta_dawn = 90, 270
    else:
        # use GSE convention?
        theta_dusk, theta_dawn = 90, 270

    night = BuiltinSurface(source='sphere', name='night')
    night.data_source.set(center=(0, 0, 0), radius=radius,
                          start_theta=theta_dusk, end_theta=theta_dawn,
                          theta_resolution=res, phi_resolution=res)
    mod = mlab.pipeline.surface(night, color=nightcol, figure=figure, **kwargs)
    mod.actor.property.backface_culling = True

    if not night_only:
        day = BuiltinSurface(source='sphere', name='day')
        day.data_source.set(center=(0, 0, 0), radius=radius,
                            start_theta=theta_dawn, end_theta=theta_dusk,
                            theta_resolution=res, phi_resolution=res)
        mod = mlab.pipeline.surface(day, color=daycol, figure=figure, **kwargs)
        mod.actor.property.backface_culling = True
    else:
        day = None

    return day, night

def to_mpl(figure=None, ax=None, size=None, antialiased=True, hide=True,
           fit=None, **kwargs):
    """Display a mayavi figure inline in an Jupyter Notebook.

    This function takes a screenshot of a figure and blits it to a matplotlib
    figure using matplotlib.pyplot.imshow()

    Args:
        figure: A mayavi figure, if not specified, uses mlab.gcf()
        ax: Matplotlib axis of the destination (plt.gca() if None)
        size (None, tuple): if given, resize the scene in pixels (x, y)
        antialiased (bool): Antialias mayavi plot
        hide (bool): if True, try to hide the render window
        fit (None, bool): Resize mpl window to fit image exactly. If
            None, then fit if figure does not currently exist.
        **kwargs: passed to mayavi.mlab.screenshot()
    """
    if figure is None:
        figure = mlab.gcf()

    if size is not None:
        resize(size, figure=figure)

    pixmap = mlab.screenshot(figure, antialiased=antialiased, **kwargs)

    # try to hide the window... Qt backend only
    if hide:
        hide_window(figure)

    if ax is None:
        from viscid.plot import vpyplot as _
        from matplotlib import pyplot as plt
        # if there are no figures, and fit is None, then fit
        if fit is None:
            fit = not bool(plt.get_fignums)
        ax = plt.gca()

    if fit:
        pltfig = ax.figure
        dpi = pltfig.get_dpi()
        pltfig.set_size_inches([s / dpi for s in figure.scene.get_size()],
                               forward=True)
        pltfig.subplots_adjust(top=1, bottom=0, left=0, right=1,
                               hspace=0, wspace=0)
    ax.imshow(pixmap)
    ax.axis('off')

def figure(*args, **kwargs):
    global_fig = kwargs.pop('global_fig', False)
    offscreen = kwargs.pop('offscreen', False)
    hide = kwargs.pop('hide', None)
    size = kwargs.pop('size', kwargs.pop('figsize', None))
    if size:
        kwargs['size'] = size

    fig = None

    if global_fig:
        fig = _global_ns.get('global_fig', None)
        if fig is not None and fig.scene is None:
            del _global_ns['global_fig']
            fig = None

    # make new figure
    if fig is None:
        fig = mlab.figure(*args, **kwargs)
        # if size was set, run resize to account for the height of window
        # decorations
        if size:
            resize(size, figure=fig)
        # hide window by default?
        if hide or (hide is None and offscreen):
            hide_window(fig)
        # send it offscreen?
        if offscreen:
            make_fig_offscreen(fig, hide=False)

    if global_fig:
        _global_ns['global_fig'] = fig

    return fig

def make_fig_offscreen(figure, hide=True):
    if hide:
        hide_window(figure)
    figure.scene.off_screen_rendering = True
    return figure

def show(stop=False):
    """Calls :meth:`mayavi.mlab.show(stop=stop)`"""
    mlab.show(stop=stop)

def clf(figure=None):
    """Clear source data, then clear figure

    Args:
        figure (mayavi.core.scene.Scene): specific figure, or None for
            :py:func:`mayavi.mlab.gcf`
    """
    if not figure:
        figure = mlab.gcf()
    clear_data(figure)
    mlab.clf(figure)

def remove_source(src):
    """Safely remove a specific vtk source

    Args:
        src (vtk_data_source): vtk data source to remove
    """
    src.stop()
    try:
        try:
            src.data.release_data()
        except TraitError:
            src.data.release_data_flag = 1
        src.cell_scalars_name = ''
        src.cell_tensors_name = ''
        src.cell_vectors_name = ''
        src.point_scalars_name = ''
        src.point_tensors_name = ''
        src.point_vectors_name = ''
    except AttributeError:
        pass
    src.start()
    src.stop()
    src.remove()

def clear_data(figures=None):
    """Workaround for Mayavi / VTK memory leak

    This is needed when Mayavi/VTK keeps a reference to source data
    when you would expect it to be freed like on a call to `mlab.clf()`
    or when removing sources from the pipeline.

    Note:
        This must be called when the pipeline still has the source, so
        before a call to `mlab.clf()`, etc.

    1. Set release_data_flag on all sources' data
    2. Remove reference to the data
    3. Remove the data source

    Args:
        figures (None, mayavi.core.scene.Scene, or 'all'): if None,
            gets current scene; if Scene object, just that one; if
            'all', act on all scenes in the current engine. Can also be
            a list of Scene objects
    """
    if figures is None:
        figures = [mlab.gcf()]
    elif figures == "all":
        figures = mlab.get_engine().scenes

    if not isinstance(figures, (list, tuple)):
        figures = [figures]
    if all(fig is None for fig in figures):
        return

    for fig in figures:
        # # fig stop / start kills mayavi now, not sure why
        # fig.stop()
        for child in list(fig.children):
            remove_source(child)
        # fig.start()
    return

def resize(size, figure=None):
    """Summary

    Args:
        size (tuple): width, height in pixels
        figure (mayavi.core.scene.Scene): specific figure, or None for
            :py:func:`mayavi.mlab.gcf`

    Returns:
        None
    """
    if figure is None:
        figure = mlab.gcf()

    try:
        # scene.set_size doesn't seem to work when rendering on screen, so
        # go into the backend and do it by hand
        if mlab.options.offscreen:
            figure.scene.set_size(size)
        elif figure.scene.off_screen_rendering:
            viscid.logger.warning("viscid.plot.vlab.resize doesn't work for "
                                  "figures that are off-screened this way. Try "
                                  "creating the figure with viscid.plot.vlab."
                                  "figure(size=(w, h), offscreen=True)")
        else:
            try:
                _ets_config = mayavi.ETSConfig
            except AttributeError:
                from traits.etsconfig.api import ETSConfig as _ets_config
            toolkit = _ets_config.toolkit

            if toolkit in ('qt', 'qt4'):
                sc = figure.scene
                widget11 = sc.control.childAt(1, 1)
                if 'toolbar' in widget11.__class__.__name__.lower():
                    toolbar_height = widget11.size().height()
                else:
                    # window_size = sc.control.parent().size()
                    # render_size = sc.control.centralWidget().size()
                    # toolbar_height = window_size.height() - render_size.height()
                    toolbar_height = 0
                sc.control.parent().resize(size[0], size[1] + toolbar_height)
            elif toolkit == 'wx':
                w, h = size[0], size[1]
                figure.scene.control.Parent.Parent.SetClientSizeWH(w, h)

            else:
                viscid.logger.warning("Unknown mayavi backend {0} (not qt4 or "
                                      "wx); not resizing.".format(toolkit))

    except Exception as e:  # pylint: disable=broad-except
        viscid.logger.warning("Resize didn't work:: {0}".format(repr(e)))

def hide_window(figure, debug=False):
    """Try to hide the window; only does something on Qt backend"""
    try:
        # fig.scene.control.parent().hide()
        figure.scene.control.parent().showMinimized()
    except Exception as e:  # pylint: disable=broad-except,unused-variable
        if debug:
            print("Window hide didn't work::", repr(e))

def savefig(*args, **kwargs):
    """Wrap mayavi.mlab.savefig with offscreen hack"""
    fig = mlab.gcf()
    prev_offscreen_state = fig.scene.off_screen_rendering
    if sys.platform != "darwin":
        fig.scene.off_screen_rendering = True

    mlab.savefig(*args, **kwargs)

    if fig.scene.off_screen_rendering != prev_offscreen_state:
        fig.scene.off_screen_rendering = prev_offscreen_state

def interact(stack_depth=0, **kwargs):
    viscid.vutil.interact(stack_depth=stack_depth + 1, mvi_ns=True, **kwargs)

plot3d_lines = plot_lines
plot_lines3d = plot_lines

##
## EOF
##
