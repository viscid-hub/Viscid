Just some notes...

 * for 1.0.0: more complete docstring coverage
 * for 1.0.0: be able to interpolate & streamline edge / face centered OpenGGCM fields
 * for 1.0.0: tidy up the parallel module

For better mayavi documentation
-------------------------------

Convenience functions:
 * plot_line / plot_lines
 * plot_ionosphere
 * plot_nulls
 * get_cmap
 * apply_cmap
 * plot_blue_marble
 * plot_earth_3d
 * interact

New mayavi wrappers
 * mlab.pipeline.scalar_cut_plane: from ScalarField
 * mlab.pipeline.vector_cut_plane: from VectorField
 * mlab.mesh: from Seeds
 * mlab.pipeline.streamline: from VectorField
 * mlab.pipeline.iso_surface: from ScalarField
 * mlab.points3d
 * mlab.quiver3d

Because Mayavi is buggy
 * clf: clear_data(), then mlab.clf()
 * remove_source: Safely remove a specific vtk source (and its memory)
 * clear_data: work around a memory leak where mlab.clf() doesn't free memory
 * resize: default resize is unreliable on OS X / Linux
 * savefig: offscreen rendering hack
