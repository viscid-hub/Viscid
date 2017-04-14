Custom Behavior (rc file)
=========================

Some classes have attributes that customize how specific files, datasets and grids behave. These attributes can be set globally, or on a per-instance basis. An instance picks up the value from the class when it's created, so changing things globally is recommended before any files are loaded. These attributes can also be set from `~/.viscidrc`.

.. note::

  By default, Viscid tries to parse the rc file with PyYaml. If PyYaml is not installed, it parses using a modified JSON format. Unlike standard JSON, you can include comments using '#' or '//' as well as trailing commas.

Here is an example rc file,

.. literalinclude:: ../resources/viscidrc
  :language: yaml

Common Customizations
---------------------

:py:mod:`viscid.plot.mpl_style`
-------------------------------

* **use_styles**: a list of style sheet names to activate
* **rc_params**: dictionary of parameters that get directly injected into
  matplotlib.rcParams
* **rc**: specify rc parameters through matplotlib.rc(...). Keys are groups
  and the values should be dictionaries that will be unpacked into the rc
  function call.

:py:mod:`viscid.plot.mpl_extra`
-------------------------------

These are deprecated and should be set using matplotlib rc parametrs
with the leading group name viscid, i.e., "viscid.symmetric_cmap". Note
that there is no "viscid.default_cmap" since that is already a matplotlib
rc parameter ("image.cmap")

* **default_cmap**: Changes the default color maps for plots made using
  :py:mod:`viscid.plot.vpyplot`
* **symmetric_cmap**: Changes the default color maps for plots that are
  symmetric about 0
* **default_cbarfmt**: Change the default tick formatter for colbars
* **default_majorfmt**: Change the default major tick formatter for axes
* **default_minorfmt**: Change the default major tick formatter for axes
* **default_majorloc**: Change the default major tick locator for axes
* **default_minorloc**: Change the default major tick locator for axes

:py:class:`viscid.grid.Grid`
----------------------------

* **force_vecter_layout** (``field.LAYOUT_*``): force all vectors to
  be of a certain layout when they're created (Default: LAYOUT_DEFAULT)
* **longterm_field_caches** (bool): If True, then when a field is
  cached, it must be explicitly removed from memory with
  "unload" or using the 'with' statemnt. If False,
  "shell copies" of fields are made so that memory is freed
  when the returned instance is garbage collected.
  (Default: False)

:py:class:`viscid.readers.openggcm.GGCMGrid`
--------------------------------------------

* **mhd_to_gse_on_read** (bool): flips arrays on load to be in
  GSE crds (Default: False)
* **copy_on_transform** (bool): True means array will be contiguous
  after transform (if one is done), but makes data load
  50\%-60\% slower (Default: True)

:py:class:`viscid.readers.openggcm.GGCMFile`
--------------------------------------------

* **read_log_file** (bool): search for a log file to load some of the
  libmrc runtime parameters. This does not read parameters
  from all libmrc classes, but can be customized with
  ``viscid.readers.ggcm_logfile.GGCMLogFile.watched_classes``.
  (Default: False due to performance)
