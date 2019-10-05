Indexing Fields
===============

The syntax and semantics for indexing are almost equivalent to Numpy's
basic indexing. The key difference is Viscid supports slice-by-location
(see below). Also, Viscid supports indexing by arrays (like Numpy's
advanced indexing), but these arrays must be one-dimensional.
Unfortunately, true advanced indexing is poorly defined when the result has
to be on a grid.

Indexing by a single index (or location) reduces that dimension from the
result. Use :py:func:`viscid.Field.slice_and_keep` to keep singly
selected dimensions.

.. warning::

    If not all slices are named, then all slices must be in the same
    order as ``fld.crds.axes``, otherwise index ordering would be poorly
    defined.

Slice-by-index
--------------

This syntax works exactly like in Numpy

    - ``fld[0]`` access the 0th element
    - ``fld[:-1]`` everything but the last element
    - ``fld[2:6:2]`` every other element from 2 to 6
    - ``fld[:5, :5]`` first five elements in x and y
    - ``fld[[0, 5, 7]]`` the 0th, 5th, and 7th elments

Slice-by-location
-----------------

Locations on the grid can be specified using imaginary numbers. The
semantics are such that the stop value is included if it matches
a point on the grid. This is different from most python indexing, but
it makes slice-by-location feel more natural.

    - ``fld[1.0j]`` access element closest to x = 1.0
    - ``fld[:0.0j]`` all elements where x <= 0.0
    - ``fld[0.0j::2]`` every other element where x >= 0.0
    - ``fld[0.0j:, 0.0j:]`` region x >= 0 and y >= 0
    - ``fld[[-1j, 0j, 1j]]`` elements closest to x = [-1, 0, 1]

Note that these slices don't necessarily take the closest grid points.
Rather, they specify a white-list region. Take the following example::

    >>> arr = numpy.array([-1, -0.5, 0.0, 0.5, 1.0])
    >>> slc = viscid.std_sel2index(numpy.s_[-0.49j:0.5j], arr)
    >>> arr[slc]
    array([0.0, 0.5])

Slice-by-mask
-------------

Slices can be a mask, just like in Numpy,

    - ``fld[[True, False, True]]`` 0th and 2nd elements

Named slices
------------

Slices can be encoded as strings to be explicit about which axis should
be indexed.

    - ``fld['y = 0j']`` select the y = 0.0 plane
    - ``fld['x = 0j', 'y = 0j']`` select the x = y = 0.0 plane
    - ``fld['x = 0j, y = 0j']`` select the x = y = 0.0 plane

Vector component slices
-----------------------

Vector components must be extracted by name. Anything that is not a
vector name is assumed to be a spatial slice. The reason for this is
that it is unambiguous. Also, the user does not have to care about the
data layout (array-of-struct, struct-of-array).

    - ``fld['x']`` extracts the x-component from a vector field
    - ``fld['x, y = 0j']`` x-component, y = 0.0 plane
    - ``fld['y = 0j', 'x']`` x-component, y = 0.0 plane
    - ``fld['x', 0j:, 0j]`` x-component, y = 0.0 plane, x >= 0.0
    - ``fld[0j:, 0j, 'x']`` x-component, y = 0.0 plane, x >= 0.0

Datetime64 / timedelta64 slices
-------------------------------

Slice-by-location does work with datetime64 / timedelta64 axes. The slices
can be specifed as normal python slices of strings or numpy datetime64
(or timedelta64) objects; or, the slice can be given as a string that looks
like ``fld["start:stop"]``. If you give the whole slice as a string, then
times must be preceeded by either "T" or "UT", otherwise the string would
be ambiguous.

    >>> t = viscid.linspace_datetime64('2010-01-01T12:00:00',
    >>>                                '2010-01-01T15:00:00', 8)
    >>> x = numpy.linspace(-1, 1, 12)
    >>> fld = viscid.zeros([t, x], crd_names='tx', center='node')
    >>>
    >>> fld[:'2010-01-01T13:30:00'].shape
    (4, 12)
    >>> fld['2010-01-01T12:30:00':'2010-01-01T13:30:00'].shape
    (2, 12)
    >>> fld[':UT2010-01-01T13:30:00'].shape
    (4, 12)
    >>> fld['UT2010-01-01T12:30:00:UT2010-01-01T13:30:00'].shape
    (2, 12)
    >>> fld[:'01:30:00'].shape
    (4, 12)
    >>> fld[':T01:30:00'].shape
    (4, 12)

Newaxis
-------

    - ``fld[numpy.newaxis]`` new axis in dimension 0
    - ``fld['u = newaxis']`` new axis in dimension 0 named 'u'
    - ``fld[:, numpy.newaxis]`` new axis in dimension 1
    - ``fld['y=:', numpy.newaxis]`` new axis after the y-axis
    - ``fld[..., numpy.newaxis]`` new axis in the last dimension

Ellipsis
--------

The ellipsis can be used in the standard way, but it can not be used
with named axes. The exception to this is that the ellipsis can be used
with named newaxes.

    - ``fld[0j:, ..., 0j:]`` select x >= 0.0 and z >= 0.0
    - ``fld[..., 'u = newaxis']`` new axis in the last dimension named u
