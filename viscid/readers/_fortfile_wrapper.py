# mostly stolen from pyggcm... thanks Matt

from threading import Lock

try:
    from viscid.readers import _jrrle
except ImportError as e:
    from viscid.verror import UnimportedModule
    msg = "Fortran readers not available since they were not built correctly"
    _jrrle = UnimportedModule(e, msg=msg)


# this lock is to prevent multiple threads from grabbing the same
# fortran file unit since checking for available units and opening
# the file may not be atomic. this should not be an issue for multiple
# processes
fortfile_open_lock = Lock()


class FortranFile(object):
    """
    small wrapper to allow the manipulation of fortran files from python
    """
    _unit = -1

    filename = None
    debug = None

    def __init__(self, name, debug=0):
        self.filename = name
        self.debug = debug

    # Make sure we close it when we're done
    def __del__(self):
        self.close()

    def open(self):
        if self.isopen:
            raise RuntimeError("Fortran file '{0}' already open"
                               "".format(self.filename))

        with fortfile_open_lock:
            self._unit = _jrrle.fopen(self.filename, uu=-1, debug=self.debug)

        if self._unit < 0:
            raise RuntimeError("Fortran open error ({0}) on '{1}'"
                               "".format(self._unit, self.filename))

    def close(self):
        if self.isopen:
            _jrrle.fclose(self._unit, debug=self.debug)
            self._unit = -1

    def seek(self, offset, whence=0):
        assert self.isopen
        status = _jrrle.seek(self._unit, offset, whence)
        if status != 0:
            raise AssertionError("status != 0: {0}".format(status))
        return status

    def tell(self):
        assert self.isopen
        pos = _jrrle.tell(self._unit)
        assert pos >= 0
        return pos

    @property
    def isopen(self):
        if self._unit > 0:
            if bool(_jrrle.fisopen(self._unit)):
                return True
            else:
                raise RuntimeError("File has a valid unit, but fortran says "
                                   "it's closed?")
        return False

    @property
    def unit(self):
        return self._unit

    def rewind(self):
        _jrrle.frewind(self._unit, debug=self.debug)

    def advance_one_line(self):
        return _jrrle.fadvance_one_line(self._unit, debug=self.debug)

    def backspace(self):
        _jrrle.fbackspace(self._unit, debug=self.debug)

    def __enter__(self):
        if not self.isopen:
            self.open()
        return self

    def __exit__(self, exc_type, value, traceback):
        if self.isopen:
            self.close()
