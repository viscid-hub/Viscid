Philosophy
==========

The idea behind Viscid is to get the python framework out of your way to help you work. To this end, data files are read into a Numpy ndarray styled object (:class:`viscid.field.Field`), and there are plenty of convenience functions to turn Fields into 2-D plots (:mod:`viscid.plot.vpyplot`) or 3-D plots (:mod:`viscid.plot.vlab`).

Here are some basic features that all readers try to implement to the best of their ability:

- Laziness: When using formats like xdmf and hdf5, we get a lot of meta data, so we can do a lot without reading the whole datafiles. Readers try their best to wait until the last possible moment to load any data. This translates into an order of magnitude better performance when loading > 100 time slices of high resolution data.

- Sane Temporal Data: If there is a way to assume files are parts of a time series, lazily consolidate everything into a :class:`viscid.dataset.DatasetTemporal`. This can be accessed like a dictionary just like Grids or Viscid Files, but they have ways to activate a specific time slice, or efficiently iterate through all time slices.

- :class:`viscid.field.Field` objects should be as much like Numpy ndarrays as possible. If for whatever reason, passing a field to a Numpy ufunc doesn't work, the underlying ndarray is always accessible via its ``data`` attribute.

The overall story goes something like this... :class:`viscid.readers.vfile.VFile` objects are :class:`viscid.dataset.Dataset` objects. Datasets contain :class:`viscid.grid.Grid` instances. Grids have :class:`viscid.field.Field` instances, :class:`viscid.coordinate.Coordinates` instances, and special methods (``_get_*``) to calculate derived quantities. At any of these levels, the object can be treated like a dictionary to get something sensible. As an example, all of the following return the same object

    >>> f = readers.load_file("some_file.xdmf")
    >>> bx = f["bx"]
    >>> f.activate_time(0)
    >>> bx = f['bx']
    >>> bx = f.get_grid(time=0)["bx"]

And to take a look into the coordinates (cell centered in this case)

    >>> x_cc = f["x_cc"]
    >>> x_cc = f.get_grid(time=0)["x_cc"]
    >>> x_cc = f['bx'].get_crd_cc('x')
