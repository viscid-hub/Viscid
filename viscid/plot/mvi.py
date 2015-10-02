"""Convevience module for making 3d plots with Mayavi

Note:
    You can't set rc parameters for this module!
"""
from __future__ import print_function
import code
import os

import numpy as np
import mayavi
from mayavi import mlab
from mayavi.sources.builtin_surface import BuiltinSurface
from mayavi.sources.vtk_data_source import VTKDataSource
from tvtk.api import tvtk
import viscid
from viscid import field


def add_source(src, figure=None):
    """Add a vtk data source to a figure

    Args:
        src (VTKDataSource): Description
        figure (mayavi.core.scene.Scene): specific figure, or
            :py:func:`mayavi.mlab.gcf`

    Returns:
        None
    """
    if not figure:
        figure = mlab.gcf()

    if src not in figure.children:
        engine = figure.parent
        engine.add_source(src, scene=figure)

def add_lines(lines, scalars=None, figure=None, name="NoName"):
    """Add list of lines to a figure

    Args:
        lines (list): See :py:func:`lines2source`
        scalars (ndarray): See :py:func:`lines2source`
        figure (mayavi.core.scene.Scene): specific figure, or
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
        figure (mayavi.core.scene.Scene): specific figure, or
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
    r = viscid.vutil.prepare_vertices(vertices, scalars)
    verts, scalars, other = r

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

    src = mlab.pipeline.scalar_scatter(lines[0], lines[1], lines[2])
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
        grid.x_coordinates = fld.get_crd_cc('x')
        grid.y_coordinates = fld.get_crd_cc('y')
        grid.z_coordinates = fld.get_crd_cc('z')
    elif fld.iscentered("Node"):
        grid.dimensions = tuple(fld.crds.shape_nc)
        grid.x_coordinates = fld.get_crd_nc('x')
        grid.y_coordinates = fld.get_crd_nc('y')
        grid.z_coordinates = fld.get_crd_nc('z')
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
        grid.dimensions = tuple(fld.crds.shape_cc)
        grid.x_coordinates = fld.get_crd_nc('x')
        grid.y_coordinates = fld.get_crd_nc('y')
        grid.z_coordinates = fld.get_crd_nc('z')
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

def plot_mesh(vertices, scalars=None, color=None, name="NoName", **kwargs):
    # discover the 2d shape of vertices that indicates the connectivity
    # of vertices
    connective_shape = list(vertices.shape[1:])
    r = viscid.vutil.prepare_vertices(vertices.reshape(3, -1), scalars)
    verts, scalars, other = r  # pylint: disable=unused-variable

    # reshape verts, scalars, and color
    target_shape = [3] + connective_shape
    verts = verts.reshape(target_shape)

    if scalars is not None:
        if color is None and scalars.dtype == np.dtype('u1'):
            color = scalars.astype('f').reshape(target_shape) / 255.0
            scalars = None
        elif scalars.dtype == np.dtype('u1'):
            # color should have precidence over this since it was explicitly
            # given
            pass
        else:
            scalars = scalars.reshape(target_shape[1:])

    src = mlab.mesh(verts[0], verts[1], verts[2], scalars=scalars,
                    color=color, name=name, **kwargs)
    return src

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
            >>> from viscid.plot import mvi
            >>>
            >>> B = viscid.vlab.get_dipole()
            >>> seeds = viscid.Line([-4, 0, 0], [4, 0, 0])
            >>> lines, topology = viscid.calc_streamlines(B, seeds,
            >>>                                           ibound=0.05)
            >>> scalars = viscid.topology2color(topology)
            >>> mvi.plot_lines(lines, scalars, tube_radius=0.02)
            >>> mvi.mlab.savefig("dipole.x3d")
            >>> viscid.vutil.meshlab_convert("dipole.x3d", "dae")
            >>> mvi.show()

    Parameters:
        lines (list): See :py:func:`lines2source`
        scalar (ndarray): See :py:func:`lines2source`
        style (str): 'strip' or 'tube'
        figure (mayavi.core.scene.Scene): specific figure, or
            :py:func:`mayavi.mlab.gcf`
        name (str): Description
        tube_radius (float): Radius if style == 'tube'
        tube_sides (int): Angular resolution if style == 'tube'
        **kwargs: passed to :meth:`mayavi.mlab.pipeline.surface`. This
            is useful for setting a colormap among other things.

    Returns:
        Mayavi surface module

    Raises:
        ValueError: if style is neither tube nor strip
    """
    style = style.lower()
    if not figure:
        figure = mlab.gcf()

    src = lines2source(lines, scalars=scalars, name=name)

    if style == "tube":
        lines = mlab.pipeline.tube(src, figure=figure, tube_radius=tube_radius,
                                   tube_sides=tube_sides)
    elif style == "strip":
        lines = mlab.pipeline.stripper(src, figure=figure)
    else:
        raise ValueError("Unknown style for lines: {0}".format(style))

    surface = mlab.pipeline.surface(lines, **kwargs)
    return surface

def plot_ionosphere(fld, radius=1.063, crd_system="mhd", figure=None,
                    bounding_lat=0.0, **kwargs):
    """Plot an ionospheric field

    Args:
        fld (Field): Some spherical (phi, theta) / (lot, lat) field
        radius (float): Defaults to 1Re + 400km == 1.063Re
        crd_system (str): Either 'gse' or 'mhd'
        figure (mayavi.core.scene.Scene): specific figure, or
            :py:func:`mayavi.mlab.gcf`
        bounding_lat (float): Description
        **kwargs: passed to :py:func:`mayavi.mlab.mesh`

    Raises:
        ValueError: Description
    """
    if figure is None:
        figure = mlab.gcf()

    if crd_system == "mhd":
        roll = 0.0
    elif crd_system == "gse":
        roll = 180.0
    else:
        raise ValueError("crd_system == '{0}' not understood"
                         "".format(crd_system))

    nphi, ntheta = fld.shape
    sphere = viscid.Sphere([0, 0, 0], r=radius, ntheta=ntheta, nphi=nphi,
                           theta_phi=False, roll=roll)
    verts, arr = sphere.wrap_mesh(fld.data)

    if 'name' not in kwargs:
        kwargs['name'] = fld.name
    m = mlab.mesh(verts[0], verts[1], verts[2], scalars=arr, figure=figure,
                  **kwargs)

    if bounding_lat:
        rp = 1.5 * radius
        z = radius * np.cos((np.pi / 180.0) * bounding_lat)
        clip = mlab.pipeline.data_set_clipper(m.module_manager.parent)
        clip.widget.widget.place_widget(-rp, rp, -rp, rp, -z, z)
        clip.widget.update_implicit_function()
        clip.widget.widget.enabled = False
        insert_filter(clip, m.module_manager)
        # m.module_manager.parent.parent.filter.auto_orient_normals = True
    else:
        pass
        # m.module_manager.parent.filter.auto_orient_normals = True
    m.actor.mapper.interpolate_scalars_before_mapping = True

    m.module_manager.scalar_lut_manager.lut_mode = 'RdBu'
    m.module_manager.scalar_lut_manager.reverse_lut = True

    return m

def insert_filter(filtr, module_manager):
    """Insert a filter above an existing module_manager

    Args:
        filter (TYPE): Description
        module_manager (TYPE): Description
    """
    filtr.parent.children.remove(module_manager)
    filtr.children.append(module_manager)

def plot_blue_marble(r=1.0, orientation=None, figure=None):
    # make a plane, then deform it into a sphere
    eps = 1e-4
    ps = tvtk.PlaneSource(origin=(r, np.pi - eps, 0.0),
                          point1=(r, np.pi - eps, 2 * np.pi),
                          point2=(r, eps, 0.0),
                          x_resolution=32,
                          y_resolution=16)
    ps.update()
    transform = tvtk.SphericalTransform()
    tpoly = tvtk.TransformPolyDataFilter(transform=transform, input=ps.output)
    src = VTKDataSource(data=tpoly.output, name="blue_marble")
    surf = mlab.pipeline.surface(src)

    # now load a jpg, and use it to texture the sphere
    fname = os.path.realpath(os.path.dirname(__file__) + '/blue_marble.jpg')
    img = tvtk.JPEGReader(file_name=fname)
    texture = tvtk.Texture(interpolate=1)
    texture.input = img.output
    surf.actor.enable_texture = True
    surf.actor.texture = texture

    if orientation:
        surf.actor.actor.orientation = orientation
    else:
        surf.actor.actor.orientation = (0, 0, -45.0)

    add_source(src, figure=figure)

    return src

def plot_earth_3d(figure=None, daycol=(1, 1, 1), nightcol=(0, 0, 0),
                  radius=1.0, res=24, crd_system="mhd", night_only=False,
                  **kwargs):
    """Plot a black and white sphere (Earth) showing sunward direction

    Parameters:
        figure (mayavi.core.scene.Scene): specific figure, or
            :py:func:`mayavi.mlab.gcf`
        daycol (tuple, optional): color of dayside (RGB)
        nightcol (tuple, optional): color of nightside (RGB)
        res (optional): rosolution of teh sphere
        crd_system (str, optional): 'mhd' or 'gse', can be gotten from
            an openggcm field using ``fld.meta["crd_system"]``.

    Returns:
        Tuple (day, night) as vtk sources
    """
    if figure is None:
        figure = mlab.gcf()

    crd_system = crd_system.lower()
    if crd_system == "mhd":
        theta_dusk, theta_dawn = 270, 90
    elif crd_system == "gse":
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

def show(stop=False):
    """Calls :meth:`mayavi.mlab.show(stop=stop)`"""
    mlab.show(stop=stop)

def clf(figure=None):
    """Clear source data, then clear figure

    Args:
        figure (mayavi.core.scene.Scene): specific figure, or
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
        fig.stop()
        for child in list(fig.children):
            remove_source(child)
        fig.start()
    return

def resize(size, figure=None):
    """Summary

    Args:
        size (tuple): width, height in pixels
        figure (mayavi.core.scene.Scene): specific figure, or
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
        else:
            toolkit = mayavi.ETSConfig.toolkit

            if toolkit == 'qt4':
                sc = figure.scene
                window_height = sc.control.parent().size().height()
                render_height = sc.render_window.size[1]
                h = window_height - render_height
                sc.control.parent().resize(size[0], size[1] + h)

            elif toolkit == 'wx':
                w, h = size[0], size[1]
                figure.scene.control.Parent.Parent.SetClientSizeWH(w, h)

            else:
                viscid.logger.warn("Unknown mayavi backend {0} (not qt4 or "
                                   "wx); not resizing.".format(toolkit))

    except Exception as e:  # pylint: disable=broad-except
        viscid.logger.warn("Resize didn't work:: {0}".format(repr(e)))


def interact(local=None):
    """Switch to interactive interpreter

    This can be handy if you want to interact with a 3d plot using
    python.

    Parameters:
        local (dict, optional): Local namespace for the interactive
            session. Usually you want ``interact(local=locals())``
    """
    banner = """Have some fun with mayavi :)
    hints:
        - use locals() to explore the namespace
        - mlab.show(stop=True) or mvi.show() to interact with the plot/gui
        - mayavi objects all have a trait_names() method
        - Use Ctrl-D (eof) to end interaction """
    code.interact(banner, local=local)

##
## EOF
##
