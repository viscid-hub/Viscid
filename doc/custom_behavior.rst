Custom Behavior (rc file)
=========================

Some classes have attributes that customize how specific files, datasets and grids behave. These attributes can be set globally, or on a per-instance basis. An instance picks up the value from the class when it's created, so changing things globally is recommended before any files are loaded. These attributes can also be set from `~/.viscidrc`. Here is an example::

    # ~/.viscidrc
    ### For Everything
    ## use shell copies so we don't have to call unload()
    grid.Grid.longterm_field_caches: false

    ### for OpenGGCM
    ## try to get extra run information
    readers.openggcm.GGCMFile.read_log_file: true
    ## everyone likes GSE coords :)
    readers.openggcm.GGCMGrid.mhd_to_gse_on_read: auto

    ### For Athena
    ## this doesn't work for some reason
    # readers.athena_bin.AthenaBinFileWrapper.var_type: prim

    # evaluator control for security
    calculator.evaluator.enabled: false

    # pretty plotting
    plot.vseaborn.enabled: true
    plot.vseaborn.context: poster
    plot.vseaborn.style: ticks
    plot.vseaborn.palette: husl, 8
    # plot.vseaborn.palette: [["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]]
    # plot.vseaborn.rc: {"lines.markeredgewidth": 0.01, "image.cmap": cubehelix}
    # plot.vseaborn.rc: {"lines.markeredgewidth": 0.01, "image.cmap": jet}
    plot.vseaborn.rc: {"image.cmap": cubehelix}

    ##
    ## EOF
    ##

Common Customizations
---------------------

viscid.grid.Grid
----------------

 * **force_vecter_layout** (``field.LAYOUT_*``): force all vectors to
   be of a certain layout when they're created
   (default: LAYOUT_DEFAULT)
 * **longterm_field_caches** (bool): If True, then when a field is
   cached, it must be explicitly removed from memory with
   "unload" or using the 'with' statemnt. If False,
   "shell copies" of fields are made so that memory is freed
   when the returned instance is garbage collected.
   Default: False

viscid.readers.openggcm.GGCMGrid
--------------------------------

 * **mhd_to_gse_on_read** (bool): flips arrays on load to be in
   GSE crds (default is False)
 * **copy_on_transform** (bool): True means array will be contiguous
   after transform (if one is done), but makes data load
   50\%-60\% slower (default is True)

viscid.readers.openggcm.GGCMFile
--------------------------------

 * **read_log_file** (bool): search for a log file to load some of the
   libmrc runtime parameters. This does not read parameters
   from all libmrc classes, but can be customized with
   :py:const`viscid.readers.ggcm_logfile.GGCMLogFile.
   watched_classes`. Defaults to False for performance.
